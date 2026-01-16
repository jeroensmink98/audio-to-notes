"""
Audio to Notes CLI - Command-line interface for transcribing audio files.

This CLI uses the core transcription module to provide audio transcription
with optional speaker diarization.
"""

import os
from glob import glob
import argparse

from openai import OpenAI

from core import (
    transcribe_audio,
    transcribe_audio_with_diarization,
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_MIN_SEGMENT_DURATION,
)

# CLI-specific configuration
CHUNKS_DIR = "chunks"
AUDIO_DIR = "input"  # Default input directory
OUTPUT_DIR = "output"  # Default output directory


def transcribe_audio_file(audio_file: str) -> None:
    """Transcribe an audio file and write the result to a file.

    This is a CLI wrapper around the core transcription function.
    """
    client = OpenAI()

    result = transcribe_audio(
        audio_file=audio_file,
        openai_client=client,
        chunks_dir=CHUNKS_DIR,
        chunk_length=DEFAULT_CHUNK_LENGTH,
        progress_callback=print,
    )

    # Write transcript to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_file))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_transcript.txt")
    with open(output_path, "w") as f:
        f.write(result.text)
    print(f"Transcript written to {output_path}")


def transcribe_audio_file_with_diarization(audio_file: str) -> None:
    """Transcribe an audio file with speaker diarization and write the result to a file.

    This is a CLI wrapper around the core diarization function.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required for speaker diarization"
        )

    client = OpenAI()

    result = transcribe_audio_with_diarization(
        audio_file=audio_file,
        openai_client=client,
        hf_token=hf_token,
        min_segment_duration=DEFAULT_MIN_SEGMENT_DURATION,
        progress_callback=print,
    )

    # Write transcript to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_file))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_transcript_diarized.txt")
    with open(output_path, "w") as f:
        f.write(result.format_transcript())
    print(f"Diarized transcript written to {output_path}")


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
