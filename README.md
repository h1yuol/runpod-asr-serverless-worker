# Audio Transcription and Speaker Diarization Service

This service provides audio transcription and speaker diarization capabilities using WhisperX and Pyannote.audio. It's designed to run as a RunPod serverless endpoint.

## Features

- Audio transcription using WhisperX large-v3 model
- Speaker diarization using Pyannote.audio
- Support for Chinese and English mixed audio
- S3 integration for input/output file handling
- GPU-accelerated processing

## Usage

### Input Format

The service expects a JSON payload with the following structure:

```json
{
    "input": {
        "s3_key": "audio/your-audio-file.wav"
    }
}
```

### Output Format

The service returns a JSON response with the following structure:

```json
{
    "status": "success",
    "output": {
        "s3_key": "text/your-audio-file.json",
        "results": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "SPEAKER_1",
                "text": "Transcribed text here"
            }
        ]
    }
}
```

### S3 Structure

- Input audio files should be uploaded to: `s3://<your-bucket-name>/audio/`
- Output transcriptions will be saved to: `s3://<your-bucket-name>/text/`

## Environment Variables

The following environment variables are required:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region
- `HF_TOKEN`: Hugging Face access token
- `S3_BUCKET_NAME`: S3 bucket name (required)

## Error Handling

The service includes comprehensive error handling and logging. If an error occurs, the response will include:

```json
{
    "status": "error",
    "error": "Error message here"
}
```

## Notes

- The service automatically handles GPU acceleration when available
- Input audio files should be in WAV format
- The service supports mixed Chinese and English audio
- All sensitive information is stored in RunPod secrets 