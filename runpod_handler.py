import runpod
import os
import boto3
from botocore.exceptions import ClientError
import whisperx
import gc
import torch
import tempfile

# Configuration (expected to be set as environment variables in RunPod)
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1') # Default to a common region
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
HF_TOKEN = os.environ.get('HF_TOKEN') # For Pyannote diarization model

# Global cache for models to reuse them across calls if the worker persists
whisper_model_cache = None
diarization_model_cache = None

def load_whisper_model(model_name, device, compute_type):
    global whisper_model_cache
    if whisper_model_cache is None:
        print(f"Loading Whisper model: {model_name} (device: {device}, compute_type: {compute_type})")
        whisper_model_cache = whisperx.load_model(model_name, device, compute_type=compute_type)
    return whisper_model_cache

def load_diarization_model(device):
    global diarization_model_cache
    diarize_model_name = "pyannote/speaker-diarization-3.1"
    if diarization_model_cache is None:
        print(f"Loading diarization model: {diarize_model_name} (device: {device})")
        if HF_TOKEN:
            print("Using Hugging Face token for diarization model.")
            diarization_model_cache = whisperx.DiarizationPipeline(model_name=diarize_model_name, use_auth_token=HF_TOKEN, device=device)
        else:
            print("No Hugging Face token. Diarization model might fail if gated.")
            # This might raise an error if the model is gated and no token is provided
            diarization_model_cache = whisperx.DiarizationPipeline(model_name=diarize_model_name, device=device)
    return diarization_model_cache


def handler(job):
    job_input = job.get('input', {})
    s3_object_key = job_input.get('s3_object_key')

    if not s3_object_key:
        return {"error": "Missing 's3_object_key' in input"}

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET_NAME:
        return {"error": "AWS credentials or S3 bucket name not configured in worker environment."}

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16" # Good default for GPU
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # MPS might not be relevant in typical RunPod GPU envs
    #     device = "mps"
    #     compute_type = "float16"
    else:
        # Should ideally not happen on a GPU serverless worker, but as a fallback
        device = "cpu"
        compute_type = "int8" 
    
    print(f"Using device: {device}, Compute type: {compute_type}")

    # Download audio file from S3 to a temporary file
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    
    local_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp_audio_file:
            local_audio_path = tmp_audio_file.name
        
        print(f"Downloading s3://{S3_BUCKET_NAME}/{s3_object_key} to {local_audio_path}")
        s3_client.download_file(S3_BUCKET_NAME, s3_object_key, local_audio_path)
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        return {"error": f"Failed to download from S3: {str(e)}"}
    except Exception as e:
        print(f"An unexpected error occurred during S3 download: {e}")
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        return {"error": f"An unexpected error occurred during S3 download: {str(e)}"}

    try:
        # --- Transcription ---
        # Using "large-v3" as it's robust and CTranslate2 compatible.
        # The specific "openai/whisper-large-v3-turbo" might not be CTranslate2.
        # User's original intent was the "biggest and most performant". "large-v3" is that for faster-whisper.
        whisper_model_name = job_input.get("whisper_model", "large-v3") 
        batch_size = job_input.get("batch_size", 16)

        model = load_whisper_model(whisper_model_name, device, compute_type)
        
        print(f"Loading audio from: {local_audio_path}")
        audio = whisperx.load_audio(local_audio_path)
        
        print("Transcribing audio...")
        result = model.transcribe(audio, batch_size=batch_size)
        
        # No need to del model if using global cache, but good practice for GPU mem if not.
        # For serverless, workers might be reused, so caching is good.
        # If an error occurs before diarization, we can return transcription only
        transcription_segments = result["segments"]


        # --- Diarization ---
        try:
            diarize_model = load_diarization_model(device)
            print("Performing diarization...")
            diarize_segments = diarize_model(audio, min_speakers=job_input.get('min_speakers'), max_speakers=job_input.get('max_speakers'))
            
            print("Assigning word speakers...")
            result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
            segments_with_speakers = result_with_speakers["segments"]

        except Exception as e_diarize:
            print(f"Error during diarization: {e_diarize}")
            print("Proceeding with transcription results only.")
            # Fallback: structure segments to be similar to what the speaker output loop expects
            segments_with_speakers = []
            for seg in transcription_segments:
                segments_with_speakers.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "speaker": "UNKNOWN" # Add an UNKNOWN speaker if diarization fails
                })

        # --- Formatting Output ---
        output_segments = []
        current_speaker_out = None
        current_speech_out = ""
        current_start_time_out = None

        for segment in segments_with_speakers:
            if "words" in segment: # Word-level details available
                for word_info in segment["words"]:
                    word_text = word_info.get("word", "")
                    speaker = word_info.get("speaker", "UNKNOWN")
                    start_time = word_info.get("start")
                    
                    if current_speaker_out is None: # First word
                        current_speaker_out = speaker
                        current_start_time_out = start_time
                        current_speech_out = word_text + " "
                    elif speaker == current_speaker_out: # Same speaker
                        current_speech_out += word_text + " "
                    else: # Speaker change
                        if current_start_time_out is not None: # Ensure there was a start time
                             output_segments.append({
                                "speaker": current_speaker_out,
                                "text": current_speech_out.strip(),
                                "start_time": round(current_start_time_out, 2)
                            })
                        current_speaker_out = speaker
                        current_start_time_out = start_time
                        current_speech_out = word_text + " "
            else: # Segment-level details (e.g., if diarization failed and we only have Whisper segments)
                speaker = segment.get("speaker", "UNKNOWN") # Should be UNKNOWN if diarization failed
                text = segment.get("text", "").strip()
                start_time = segment.get("start")
                end_time = segment.get("end")

                if current_speaker_out == speaker and current_speech_out: # Continuation
                    current_speech_out += text + " "
                else: # New speaker or first segment after word-level section
                    if current_speech_out: # Print previous accumulated speech
                         output_segments.append({
                            "speaker": current_speaker_out,
                            "text": current_speech_out.strip(),
                            "start_time": round(current_start_time_out, 2) if current_start_time_out is not None else None
                        })
                    current_speaker_out = speaker
                    current_speech_out = text + " "
                    current_start_time_out = start_time
        
        # Add the last spoken segment
        if current_speech_out:
            output_segments.append({
                "speaker": current_speaker_out,
                "text": current_speech_out.strip(),
                "start_time": round(current_start_time_out, 2) if current_start_time_out is not None else None
            })
            
        final_output = {"segments": output_segments, "language_code": result.get("language")}
        
        return final_output

    except Exception as e:
        print(f"Error during ASR processing: {e}")
        # Attempt to return partial transcription if available
        if 'transcription_segments' in locals() and transcription_segments:
             return {"error": f"ASR processing failed: {str(e)}", "partial_transcription": transcription_segments}
        return {"error": f"ASR processing failed: {str(e)}"}
    finally:
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)
            print(f"Cleaned up temporary audio file: {local_audio_path}")
        
        # Clean up GPU memory (important for serverless if worker is reused)
        # Models are cached globally, so we don't delete them here, but clear cache
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        # elif device == "mps": # if MPS was enabled
        #     torch.mps.empty_cache()

runpod.serverless.start({"handler": handler}) 