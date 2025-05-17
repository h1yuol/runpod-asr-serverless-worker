import runpod
import os
import boto3
from botocore.exceptions import ClientError
import whisperx
import gc
import torch
import tempfile
import opencc # For Traditional to Simplified Chinese conversion

# Configuration (expected to be set as environment variables in RunPod)
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1') # Default to a common region
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
HF_TOKEN = os.environ.get('HF_TOKEN') # For Pyannote diarization model

# Global cache for models to reuse them across calls if the worker persists
whisper_model_cache = None
diarization_model_cache = None
opencc_converter_cache = None


def get_opencc_converter():
    global opencc_converter_cache
    if opencc_converter_cache is None:
        print("Initializing OpenCC converter (t2s.json)")
        opencc_converter_cache = opencc.OpenCC('t2s.json') # Traditional to Simplified
    return opencc_converter_cache

def load_whisper_model(model_name, device, compute_type, language=None): # Added language
    global whisper_model_cache
    # Cache key could be more sophisticated if more params change model loading
    cache_key = f"{model_name}-{device}-{compute_type}-{language}"
    # For simplicity, we'll just reload if language changes for now, or handle it in transcribe
    # A more robust cache would be a dict with cache_key.
    # For WhisperX, language is typically handled at transcribe time, but some model aspects might be language-dependent.
    # whisperx.load_model itself doesn't take language, but the underlying faster-whisper does.
    # Let's assume for now whisperx handles this by passing language to transcribe.

    if whisper_model_cache is None: # Simplified cache check for this example
        print(f"Loading Whisper model: {model_name} (device: {device}, compute_type: {compute_type})")
        # Language is not directly passed to whisperx.load_model
        # It's passed to model.transcribe()
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
            diarization_model_cache = whisperx.DiarizationPipeline(model_name=diarize_model_name, device=device)
    return diarization_model_cache


def handler(job):
    job_input = job.get('input', {})
    s3_object_key = job_input.get('s3_object_key')
    # New: Get language and initial_prompt from input
    target_language = job_input.get('language') # e.g., "en", "zh"
    initial_prompt_text = job_input.get('initial_prompt') # e.g., "以下是普通话的句子。"

    if not s3_object_key:
        return {"error": "Missing 's3_object_key' in input"}

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET_NAME:
        return {"error": "AWS credentials or S3 bucket name not configured in worker environment."}

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16" # Good default for GPU
    else:
        device = "cpu"
        compute_type = "int8" 
    
    print(f"Using device: {device}, Compute type: {compute_type}")
    if target_language:
        print(f"Target language specified: {target_language}")
    if initial_prompt_text:
        print(f"Using initial prompt: {initial_prompt_text}")

    # Download audio file from S3 to a temporary file
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    
    local_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp_audio_file: # Assuming M4A, suffix can be dynamic
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
        whisper_model_name = job_input.get("whisper_model", "large-v3") 
        batch_size = job_input.get("batch_size", 16)

        # Pass language to load_whisper_model if it affects model loading,
        # otherwise, it's mainly for the transcribe step.
        model = load_whisper_model(whisper_model_name, device, compute_type) 
        
        print(f"Loading audio from: {local_audio_path}")
        audio = whisperx.load_audio(local_audio_path)
        
        print("Transcribing audio...")
        # Pass language and initial_prompt to transcribe method
        transcribe_kwargs = {}
        if target_language:
            transcribe_kwargs['language'] = target_language
        if initial_prompt_text:
            transcribe_kwargs['initial_prompt'] = initial_prompt_text
        
        result = model.transcribe(audio, batch_size=batch_size, **transcribe_kwargs)
        
        transcription_segments = result["segments"]
        detected_language = result.get("language", target_language) # Use detected if not overridden by input

        # Convert to Simplified Chinese if language is Chinese
        if detected_language and detected_language.lower().startswith('zh'):
            print(f"Language is Chinese ({detected_language}). Converting to Simplified Chinese.")
            converter = get_opencc_converter()
            for seg in transcription_segments:
                if 'text' in seg:
                    seg['text'] = converter.convert(seg['text'])


        # --- Diarization ---
        try:
            diarize_model = load_diarization_model(device)
            # Check if diarization model loaded successfully
            if diarize_model is None:
                raise RuntimeError("Diarization model failed to load. Check HF_TOKEN and model availability.")

            print("Performing diarization...")
            diarize_segments = diarize_model(audio, min_speakers=job_input.get('min_speakers'), max_speakers=job_input.get('max_speakers'))
            
            print("Assigning word speakers...")
            result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result) # result already has simplified chinese if converted
            segments_with_speakers = result_with_speakers["segments"]

        except Exception as e_diarize:
            print(f"Error during diarization: {e_diarize}")
            print("Proceeding with transcription results only (speakers will be UNKNOWN).")
            # Fallback: structure segments from transcription_segments (which are already simplified if Chinese)
            segments_with_speakers = []
            for seg in transcription_segments: # transcription_segments already has simplified Chinese if applicable
                segments_with_speakers.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"), # This text is already simplified
                    "speaker": "UNKNOWN"
                })

        # --- Formatting Output ---
        output_segments = []
        current_speaker_out = None
        current_speech_out = ""
        current_start_time_out = None

        for segment in segments_with_speakers: # segments_with_speakers has simplified chinese in 'text' if applicable
            if "words" in segment: # Word-level details available
                for word_info in segment["words"]:
                    word_text = word_info.get("word", "")
                    # If word-level text also needs conversion (it should be covered by segment conversion if done right)
                    # For now, assume segment['text'] was the primary target for conversion.
                    # If whisperx.assign_word_speakers re-generates 'word' text from original audio features,
                    # then word-level conversion might be needed here too if not handled before assign_word_speakers.
                    # However, result["segments"] was modified before assign_word_speakers, so internal words should reflect that.

                    speaker = word_info.get("speaker", "UNKNOWN")
                    start_time = word_info.get("start")
                    
                    if current_speaker_out is None: # First word
                        current_speaker_out = speaker
                        current_start_time_out = start_time
                        current_speech_out = word_text + " "
                    elif speaker == current_speaker_out: # Same speaker
                        current_speech_out += word_text + " "
                    else: # Speaker change
                        if current_start_time_out is not None: 
                             output_segments.append({
                                "speaker": current_speaker_out,
                                "text": current_speech_out.strip(), # This is from word_text, already simplified
                                "start_time": round(current_start_time_out, 2)
                            })
                        current_speaker_out = speaker
                        current_start_time_out = start_time
                        current_speech_out = word_text + " "
            else: # Segment-level details (e.g., if diarization failed)
                speaker = segment.get("speaker", "UNKNOWN") 
                text = segment.get("text", "").strip() # This text is already simplified
                start_time = segment.get("start")
                
                if current_speaker_out == speaker and current_speech_out: 
                    current_speech_out += text + " "
                else: 
                    if current_speech_out: 
                         output_segments.append({
                            "speaker": current_speaker_out,
                            "text": current_speech_out.strip(), # This text is already simplified
                            "start_time": round(current_start_time_out, 2) if current_start_time_out is not None else None
                        })
                    current_speaker_out = speaker
                    current_speech_out = text + " "
                    current_start_time_out = start_time
        
        # Add the last spoken segment
        if current_speech_out:
            output_segments.append({
                "speaker": current_speaker_out,
                "text": current_speech_out.strip(), # This text is already simplified
                "start_time": round(current_start_time_out, 2) if current_start_time_out is not None else None
            })
            
        final_output = {"segments": output_segments, "language_code": detected_language} # Report the final language used/detected
        
        return final_output

    except Exception as e:
        print(f"Error during ASR processing: {e}")
        if 'transcription_segments' in locals() and transcription_segments:
             # Ensure partial transcription is also simplified if Chinese
            if detected_language and detected_language.lower().startswith('zh'):
                converter = get_opencc_converter()
                for seg in transcription_segments:
                    if 'text' in seg:
                        seg['text'] = converter.convert(seg['text'])
            return {"error": f"ASR processing failed: {str(e)}", "partial_transcription": transcription_segments}
        return {"error": f"ASR processing failed: {str(e)}"}
    finally:
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)
            print(f"Cleaned up temporary audio file: {local_audio_path}")
        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

runpod.serverless.start({"handler": handler}) 