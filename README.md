# Audio to Notes

Transcribes audio files using OpenAI Whisper API with optional speaker diarization.

Available in two modes:
- **CLI**: Command-line tool for local transcription
- **API**: FastAPI backend with job queue and retention for production use

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- [ffmpeg](https://ffmpeg.org/) (for audio processing)
- OpenAI API key
- HuggingFace token (optional, for speaker diarization)

## Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for diarization
```

### HuggingFace Token (for Speaker Diarization)

If you want to use speaker diarization, you need a HuggingFace token:

1. Create a free account at https://huggingface.co
2. Accept the model terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create an access token at https://huggingface.co/settings/tokens
4. Add `HF_TOKEN=your_token_here` to your `.env` file

## Usage

### CLI Mode

Basic transcription:

```bash
uv run --env-file .env main.py <audio_file>
```

With speaker diarization (identifies different speakers):

```bash
uv run --env-file .env main.py --diarize <audio_file>
```

### API Mode

Run the FastAPI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then use the REST API to create jobs, check status, and download transcripts. See [API_README.md](API_README.md) for complete API documentation.

**Quick example:**
```bash
# Create a job
curl -X POST "http://localhost:8000/jobs" \
  -H "X-OpenAI-API-Key: sk-..." \
  -F "file=@audio.mp3"

# Check status
curl http://localhost:8000/jobs/{job_id}

# Download transcript
curl -O -J http://localhost:8000/jobs/{job_id}/download
```

## How it works

**Basic mode**: The audio file is split into 5-minute chunks using ffmpeg, each chunk is transcribed via OpenAI Whisper, and the transcripts are combined into a single output file.

**Diarization mode**: Speaker diarization is performed using pyannote.audio to identify different speakers, then each speaker segment is transcribed separately with speaker labels.

## Output

### CLI Mode
Transcripts are saved to the `output/` directory:

- Basic: `<filename>_transcript.txt`
- Diarized: `<filename>_transcript_diarized.txt`

### API Mode
Transcripts are stored in `jobs/output/` and available via the download endpoint. Jobs are automatically deleted after 2 hours.

Diarized output format:
```
[00:00:00 - 00:00:05] SPEAKER_00: Hello, welcome to the meeting.
[00:00:05 - 00:00:12] SPEAKER_01: Thank you for having me.
```

## Cost

- **Transcription**: ~$0.36 per hour of audio ($0.006 per minute)
- **Diarization**: Free (runs locally using pyannote.audio)
