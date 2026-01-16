"""
Audio-to-Notes Core Module.

This module provides the core transcription functionality as a reusable library.
It contains no CLI, UI, or HTTP-specific code.

Example usage:
    from openai import OpenAI
    from core import transcribe_audio, TranscriptionResult

    client = OpenAI()
    result = transcribe_audio(
        audio_file="audio.mp3",
        openai_client=client,
        chunks_dir="/tmp/chunks"
    )
    print(result.text)
"""

from core.transcription import (
    # Data classes
    TranscriptPart,
    TranscriptionResult,
    DiarizedTranscriptionResult,
    # Configuration defaults
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_MIN_SEGMENT_DURATION,
    DEFAULT_INPUT_LANGUAGE,
    # Main functions
    transcribe_audio,
    transcribe_audio_with_diarization,
    # Utility functions
    get_audio_duration,
    format_duration,
    # Lower-level functions (for advanced usage)
    transcribe_chunk,
    chunk_audio,
    extract_audio_segment,
    detect_language_from_chunk,
    diarize_audio,
    group_speaker_segments,
)

__all__ = [
    # Data classes
    "TranscriptPart",
    "TranscriptionResult",
    "DiarizedTranscriptionResult",
    # Configuration defaults
    "DEFAULT_CHUNK_LENGTH",
    "DEFAULT_MIN_SEGMENT_DURATION",
    "DEFAULT_INPUT_LANGUAGE",
    # Main functions
    "transcribe_audio",
    "transcribe_audio_with_diarization",
    # Utility functions
    "get_audio_duration",
    "format_duration",
    # Lower-level functions
    "transcribe_chunk",
    "chunk_audio",
    "extract_audio_segment",
    "detect_language_from_chunk",
    "diarize_audio",
    "group_speaker_segments",
]
