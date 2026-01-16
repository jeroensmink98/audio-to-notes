# Audio to Notes

Transcribes audio files using OpenAI Whisper API with optional speaker diarization.

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

Basic transcription:

```bash
uv run --env-file .env main.py <audio_file>
```

With speaker diarization (identifies different speakers):

```bash
uv run --env-file .env main.py --diarize <audio_file>
```

## How it works

**Basic mode**: The audio file is split into 5-minute chunks using ffmpeg, each chunk is transcribed via OpenAI Whisper, and the transcripts are combined into a single output file.

**Diarization mode**: Speaker diarization is performed using pyannote.audio to identify different speakers, then each speaker segment is transcribed separately with speaker labels.

## Output

Transcripts are saved to the `output/` directory:

- Basic: `<filename>_transcript.txt`
- Diarized: `<filename>_transcript_diarized.txt`

Diarized output format:
```
[00:00:00 - 00:00:05] SPEAKER_00: Hello, welcome to the meeting.
[00:00:05 - 00:00:12] SPEAKER_01: Thank you for having me.
```

## Programmatic Usage (Core Module)

The transcription functionality is available as a reusable Python library through the `core` module. This allows integration with FastAPI backends, web applications, or any Python project.

### Basic Transcription

```python
from openai import OpenAI
from core import transcribe_audio, TranscriptionResult

client = OpenAI()  # Uses OPENAI_API_KEY environment variable

result: TranscriptionResult = transcribe_audio(
    audio_file="path/to/audio.mp3",
    openai_client=client,
    chunks_dir="/tmp/chunks",
    progress_callback=print,  # Optional: for logging progress
)

print(f"Transcript: {result.text}")
print(f"Detected language: {result.language}")
```

### Transcription with Speaker Diarization

```python
from openai import OpenAI
from core import transcribe_audio_with_diarization, DiarizedTranscriptionResult

client = OpenAI()
hf_token = "your_huggingface_token"

result: DiarizedTranscriptionResult = transcribe_audio_with_diarization(
    audio_file="path/to/audio.mp3",
    openai_client=client,
    hf_token=hf_token,
    progress_callback=print,
)

# Get formatted transcript with speaker labels
print(result.format_transcript())

# Access individual parts
for part in result.parts:
    print(f"{part.speaker}: {part.text}")

# Get list of unique speakers
print(f"Speakers: {result.speakers}")
```

### Available Data Classes

- `TranscriptionResult`: Contains `text` (transcript) and `language` (detected language)
- `DiarizedTranscriptionResult`: Contains `parts` (list of `TranscriptPart`), `language`, and `speakers` (set of speaker IDs)
- `TranscriptPart`: Contains `speaker`, `start`, `end`, and `text` for each segment

### Available Functions

| Function | Description |
|----------|-------------|
| `transcribe_audio()` | Main function for basic transcription |
| `transcribe_audio_with_diarization()` | Transcription with speaker identification |
| `get_audio_duration()` | Get duration of audio file in seconds |
| `format_duration()` | Format seconds to HH:MM:SS.mmm |
| `chunk_audio()` | Split audio into chunks |
| `extract_audio_segment()` | Extract a time range from audio |
| `detect_language_from_chunk()` | Auto-detect language from audio |
| `diarize_audio()` | Identify speakers in audio |
| `group_speaker_segments()` | Merge consecutive same-speaker segments |

## Cost

- **Transcription**: ~$0.36 per hour of audio ($0.006 per minute)
- **Diarization**: Free (runs locally using pyannote.audio)
