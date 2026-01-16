"""CLI for audio transcription using core transcription module."""
import os
import sys
from glob import glob
import argparse

# Add parent directory to path to import core module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import (
    transcribe_audio_file,
    transcribe_audio_file_with_diarization,
    AUDIO_DIR,
)

def main():
    # Check if OPENAI_API_KEY environment variable is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    parser = argparse.ArgumentParser(description="Audio to Notes Script")
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to a single audio file to process (.mp3, .m4a, .amr, .mov, etc)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires HF_TOKEN environment variable)",
    )
    args = parser.parse_args()

    # Check for HF_TOKEN if diarization is requested
    if args.diarize and not os.getenv("HF_TOKEN"):
        print(
            "Error: HF_TOKEN environment variable is required for speaker diarization."
        )
        print("Get a token at https://huggingface.co/settings/tokens")
        return

    if args.audio_file:
        audio_files = [args.audio_file]
    else:
        # Find audio files in input directory (supporting .m4a, .mp3, .mp4, .wav, .aac, .flac, .amr, .mov)
        audio_files = []
        for ext in (
            "*.m4a",
            "*.mp3",
            "*.mp4",
            "*.wav",
            "*.aac",
            "*.flac",
            "*.amr",
            "*.mov",
        ):
            audio_files.extend(glob(os.path.join(AUDIO_DIR, ext)))
    if not audio_files:
        print("No audio files found to process.")
        return
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        if args.diarize:
            transcribe_audio_file_with_diarization(audio_file)
        else:
            transcribe_audio_file(audio_file)
    print("All done!")


if __name__ == "__main__":
    main()
