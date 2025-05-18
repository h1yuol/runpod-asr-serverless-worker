import os
import json
import tempfile
import boto3
import runpod
import whisperx
import torch
import torchaudio
from pyannote.audio import Pipeline
import logging
import warnings

# Suppress torchaudio backend warning
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')
bucket_name = os.environ.get('S3_BUCKET_NAME')
if not bucket_name:
    raise ValueError("S3_BUCKET_NAME environment variable is required")

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

TRANSCRIPTION_MODEL = "large-v3"

DEFAULT_SUFFIX = ".wav"

COMPUTE_TYPE = "float16"

def download_from_s3(s3_key):
    """Download file from S3 to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=DEFAULT_SUFFIX) as temp_file:
            s3_client.download_file(bucket_name, s3_key, temp_file.name)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error downloading file from S3: {str(e)}")
        raise

def upload_to_s3(file_path, s3_key):
    """Upload file to S3."""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        raise

def process_audio(audio_path):
    """Process audio file with WhisperX and speaker diarization."""
    try:
        # Load WhisperX model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model and compute type
        model = whisperx.load_model(TRANSCRIPTION_MODEL, device, compute_type=COMPUTE_TYPE)
        logger.info("WhisperX model loaded successfully")
        
        # Transcribe audio with proper parameters
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(
            audio,
            batch_size=16,
        )
        logger.info("Audio transcription completed")

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        logger.info("WhisperX alignment completed")

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        print(result['segments'])
        
        # Format results
        formatted_results = []
        for segment in result["segments"]:
            formatted_results.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment.get("speaker", "UNKNOWN"),
                "text": segment["text"]
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        # print stack trace
        import traceback
        traceback.print_exc()
        raise

def handler(event):
    """RunPod handler function."""
    try:
        # Get input parameters
        input_data = event["input"]
        s3_key = input_data.get("s3_key")
        
        if not s3_key:
            raise ValueError("s3_key is required in input")
        
        # Download audio file from S3
        audio_path = download_from_s3(s3_key)
        logger.info(f"Audio file downloaded from S3: {s3_key}")
        
        # Process audio
        results = process_audio(audio_path)
        logger.info("Audio processing completed")
        
        # Save results to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump(results, temp_file, ensure_ascii=False, indent=2)
            temp_file_path = temp_file.name
        
        # Upload results to S3
        output_key = s3_key.replace('audio/', 'text/').replace('.wav', '.json')
        upload_to_s3(temp_file_path, output_key)
        logger.info(f"Results uploaded to S3: {output_key}")
        
        # Clean up temporary files
        os.unlink(audio_path)
        os.unlink(temp_file_path)
        
        return {
            "status": "success",
            "output": {
                "s3_key": output_key,
                "results": results
            }
        }
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Initialize RunPod handler
runpod.serverless.start({"handler": handler}) 