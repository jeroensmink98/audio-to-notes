# AGENTS.md

This file provides guidance for AI agents working on the Audio to Notes codebase.

## Project Overview

Audio to Notes is a Python CLI tool that transcribes audio files using the OpenAI Whisper API. It supports:
- Basic transcription with automatic language detection
- Speaker diarization using pyannote.audio to identify different speakers

## Technology Stack

- **Python 3.10+** - Required Python version
- **uv** - Package manager (https://github.com/astral-sh/uv)
- **OpenAI Whisper API** - Cloud-based transcription
- **pyannote.audio** - Local speaker diarization
- **ffmpeg** - Audio processing and format conversion

## Project Structure

```
/
├── cli/               # CLI application
│   └── main.py       # Command-line interface
├── backend/          # Backend API
│   └── api/         # FastAPI application
│       ├── api.py   # API endpoints and worker
│       ├── database.py # SQLite models
│       ├── test_api.py # Test client
│       └── start_api.sh # Startup script
├── core.py          # Shared transcription logic
├── pyproject.toml   # Project dependencies and metadata
├── uv.lock          # Locked dependencies
├── .env             # Environment variables (not in git)
├── input/           # Default input directory for audio files (gitignored)
├── output/          # Transcription output directory (gitignored)
├── chunks/          # Temporary audio chunks during processing (gitignored)
└── jobs/            # API job storage (gitignored)
    ├── audio/       # Uploaded audio files
    └── output/      # Generated transcripts
```

## Development Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Create `.env` file** with required API keys:
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   HF_TOKEN=your_huggingface_token_here  # Optional, for diarization
   ```

4. **Ensure ffmpeg is installed** on the system

## Running the Application

### CLI Mode

Basic transcription:
```bash
uv run --env-file .env cli/main.py <audio_file>
```

With speaker diarization:
```bash
uv run --env-file .env cli/main.py --diarize <audio_file>
```

Process all files in `input/` directory:
```bash
uv run --env-file .env cli/main.py
```

### API Mode

Start the FastAPI server:
```bash
# Using the startup script
./backend/api/start_api.sh

# Or using uvicorn directly
uvicorn backend.api.api:app --host 0.0.0.0 --port 8000
```

See `backend/api/README.md` for API documentation.

## Key Code Patterns

### Configuration Constants
All configuration is defined at the top of `main.py`:
- `CHUNK_LENGTH` - Audio chunk size in seconds (5 minutes)
- `CHUNKS_DIR`, `AUDIO_DIR`, `OUTPUT_DIR` - Directory paths
- `INPUT_LANGUAGE` - Fallback language code
- `MIN_SEGMENT_DURATION` - Minimum segment duration for diarization

### Audio Processing Flow
1. **Basic mode**: Audio → Chunk (5 min) → Transcribe each → Combine
2. **Diarization mode**: Audio → Speaker segments → Transcribe each with labels

### Error Handling
- API calls use retry loops with 5-second delays on failure
- ffmpeg failures raise `RuntimeError` with detailed error messages
- Missing API keys are checked at startup

### Temporary Files
- Audio chunks are stored in `chunks/` directory
- Extracted segments use `tempfile.NamedTemporaryFile`
- Cleanup is performed after processing

## Common Development Tasks

### Adding a New Audio Format
1. Add the extension to the glob patterns in `main()` function (line ~414)
2. No other changes needed - ffmpeg handles format conversion automatically

### Modifying Chunk Size
Update `CHUNK_LENGTH` constant at the top of `main.py`

### Adding New CLI Arguments
Use `argparse` - see existing `--diarize` flag for reference

### Changing Output Format
Modify the formatting in `transcribe_audio_file()` or `transcribe_audio_file_with_diarization()` functions

## Important Notes

### API Keys
- `OPENAI_API_KEY` is **required** for all operations
- `HF_TOKEN` is only needed when using `--diarize` flag
- Never commit `.env` files

### GPU Support
The diarization code auto-detects GPU availability and falls back to CPU if:
- No GPU is available
- GPU is not compatible with PyTorch

### Supported Audio Formats
The tool supports: `.m4a`, `.mp3`, `.mp4`, `.wav`, `.aac`, `.flac`, `.amr`, `.mov`

### Output Files
- Basic: `output/<filename>_transcript.txt`
- Diarized: `output/<filename>_transcript_diarized.txt`

## Testing

There are currently no automated tests in this project. When adding tests:
- Use `pytest` as the testing framework
- Add test dependencies to `pyproject.toml`
- Mock external API calls (OpenAI, HuggingFace)

## Dependencies

When updating dependencies:
- Use `uv add <package>` to add new packages
- Run `uv sync` to update the lock file
- Keep versions pinned in `pyproject.toml` for reproducibility
