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

def download_from_s3(s3_key):
    """Download file from S3 to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
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
        model = whisperx.load_model("large-v3", device, compute_type="float16")
        logger.info("WhisperX model loaded successfully")
        
        # Transcribe audio
        result = model.transcribe(
            audio_path,
            batch_size=16,
            language=None,  # Auto-detect language
            compute_type="float16"
        )
        logger.info("Audio transcription completed")
        
        # Load speaker diarization model
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        diarize_model.to(torch.device(device))
        logger.info("Speaker diarization model loaded successfully")
        
        # Perform diarization
        diarize_segments = diarize_model(audio_path)
        logger.info("Speaker diarization completed")
        
        # Align speaker labels with transcription
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker labels aligned with transcription")
        
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