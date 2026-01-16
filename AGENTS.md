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
├── main.py           # CLI wrapper - argument parsing and file I/O only
├── core/             # Core transcription module (reusable library)
│   ├── __init__.py   # Public API exports
│   └── transcription.py  # Pure transcription logic
├── tests/            # Test suite
│   ├── inputs/       # Test audio files
│   ├── outputs/      # Expected test outputs
│   ├── test_core.py  # Unit tests for core module
│   └── test_integration.py  # Integration tests
├── pyproject.toml    # Project dependencies and metadata
├── uv.lock           # Locked dependencies
├── .env              # Environment variables (not in git)
├── input/            # Default input directory for audio files (gitignored)
├── output/           # Transcription output directory (gitignored)
└── chunks/           # Temporary audio chunks during processing (gitignored)
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

Basic transcription:
```bash
uv run --env-file .env main.py <audio_file>
```

With speaker diarization:
```bash
uv run --env-file .env main.py --diarize <audio_file>
```

Process all files in `input/` directory:
```bash
uv run --env-file .env main.py
```

## Key Code Patterns

### Architecture
The codebase follows a clean separation of concerns:
- **`core/` module**: Pure transcription logic with no CLI/UI/HTTP dependencies
- **`main.py`**: Thin CLI wrapper that handles argument parsing and file I/O

### Core Module API
The `core` module provides:

**Data Classes:**
- `TranscriptionResult`: Contains `text` and `language`
- `DiarizedTranscriptionResult`: Contains `parts`, `language`, `speakers`, and `format_transcript()` method
- `TranscriptPart`: Contains `speaker`, `start`, `end`, `text`

**Main Functions:**
- `transcribe_audio()`: Basic chunked transcription
- `transcribe_audio_with_diarization()`: Transcription with speaker identification

**Utility Functions:**
- `get_audio_duration()`, `format_duration()`, `chunk_audio()`, `extract_audio_segment()`
- `detect_language_from_chunk()`, `diarize_audio()`, `group_speaker_segments()`

### Configuration Constants
All configuration defaults are defined in `core/transcription.py` and exported via `core/__init__.py`:
- `DEFAULT_CHUNK_LENGTH` - Audio chunk size in seconds (5 minutes)
- `DEFAULT_MIN_SEGMENT_DURATION` - Minimum segment duration for diarization
- `DEFAULT_INPUT_LANGUAGE` - Fallback language code

```python
from core import DEFAULT_CHUNK_LENGTH, DEFAULT_MIN_SEGMENT_DURATION
```

CLI-specific paths are in `main.py`:
- `CHUNKS_DIR`, `AUDIO_DIR`, `OUTPUT_DIR` - Directory paths

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

### Using the Core Module Programmatically
```python
from openai import OpenAI
from core import transcribe_audio, transcribe_audio_with_diarization

client = OpenAI()

# Basic transcription
result = transcribe_audio(
    audio_file="audio.mp3",
    openai_client=client,
    chunks_dir="/tmp/chunks",
    progress_callback=print,  # Optional
)
print(result.text, result.language)

# With diarization
result = transcribe_audio_with_diarization(
    audio_file="audio.mp3",
    openai_client=client,
    hf_token="your_hf_token",
    progress_callback=print,
)
print(result.format_transcript())
```

### Adding a New Audio Format
1. Add the extension to the glob patterns in `main()` function in `main.py`
2. No other changes needed - ffmpeg handles format conversion automatically

### Modifying Chunk Size
Update `DEFAULT_CHUNK_LENGTH` constant in `core/transcription.py`

### Adding New CLI Arguments
Use `argparse` in `main.py` - see existing `--diarize` flag for reference

### Changing Output Format
Modify the `format_transcript()` method in `DiarizedTranscriptionResult` class in `core/transcription.py`

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

The project uses `pytest` for testing. Tests are located in the `tests/` directory.

### Running Tests
```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# Run only unit tests (no ffmpeg required)
PYTHONPATH=. python -m pytest tests/test_core.py -v

# Run integration tests (requires ffmpeg)
PYTHONPATH=. python -m pytest tests/test_integration.py -v
```

### Test Structure
- `tests/test_core.py`: Unit tests for pure functions and data classes (26 tests)
- `tests/test_integration.py`: Integration tests using real audio files (skipped if ffmpeg unavailable)
- `tests/inputs/`: Test audio files (e.g., `jfk_speech_input.mp3`)
- `tests/outputs/`: Expected test outputs

### Writing New Tests
- Mock external API calls (OpenAI, HuggingFace) for unit tests
- Use `@pytest.mark.skipif` to skip tests when dependencies are unavailable
- Test data classes and pure functions without mocking

## Dependencies

When updating dependencies:
- Use `uv add <package>` to add new packages
- Run `uv sync` to update the lock file
- Keep versions pinned in `pyproject.toml` for reproducibility
