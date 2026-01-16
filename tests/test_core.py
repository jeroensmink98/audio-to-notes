"""
Unit tests for the core transcription module.

These tests validate the core transcription logic without requiring
external API calls (OpenAI, HuggingFace).
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from core import (
    format_duration,
    group_speaker_segments,
    TranscriptPart,
    TranscriptionResult,
    DiarizedTranscriptionResult,
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_MIN_SEGMENT_DURATION,
    DEFAULT_INPUT_LANGUAGE,
)


class TestFormatDuration:
    """Tests for the format_duration function."""

    def test_zero_seconds(self):
        assert format_duration(0) == "00:00:00.000"

    def test_seconds_only(self):
        assert format_duration(45) == "00:00:45.000"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "00:02:05.000"

    def test_hours_minutes_seconds(self):
        assert format_duration(3725) == "01:02:05.000"

    def test_milliseconds(self):
        assert format_duration(1.5) == "00:00:01.500"

    def test_milliseconds_precision(self):
        # Test with different decimal values
        assert format_duration(0.123) == "00:00:00.123"
        assert format_duration(0.001) == "00:00:00.001"

    def test_large_duration(self):
        # 2 hours, 30 minutes, 45 seconds, 500ms
        assert format_duration(2 * 3600 + 30 * 60 + 45.5) == "02:30:45.500"


class TestGroupSpeakerSegments:
    """Tests for the group_speaker_segments function."""

    def test_empty_segments(self):
        """Empty input should return empty output."""
        assert group_speaker_segments([]) == []

    def test_single_segment(self):
        """Single segment should be returned as-is."""
        segments = [{"start": 0, "end": 10, "speaker": "SPEAKER_00"}]
        result = group_speaker_segments(segments)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 10, "speaker": "SPEAKER_00"}

    def test_merge_consecutive_same_speaker(self):
        """Consecutive segments from same speaker should be merged."""
        segments = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_00"},
        ]
        result = group_speaker_segments(segments, min_duration=20)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 10, "speaker": "SPEAKER_00"}

    def test_different_speakers_not_merged(self):
        """Different speakers should not be merged."""
        segments = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_01"},
        ]
        result = group_speaker_segments(segments)
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_gap_too_large_not_merged(self):
        """Segments with large gap should not be merged even if same speaker."""
        segments = [
            {"start": 0, "end": 15, "speaker": "SPEAKER_00"},  # 15s long
            {"start": 20, "end": 25, "speaker": "SPEAKER_00"},  # 5s gap
        ]
        # With min_duration=10, the first segment is long enough
        # and gap (5s) > min_gap (0.5s), so should not merge
        result = group_speaker_segments(segments, min_gap=0.5, min_duration=10)
        assert len(result) == 2

    def test_gap_small_same_speaker_merged(self):
        """Segments with small gap from same speaker should be merged."""
        segments = [
            {"start": 0, "end": 15, "speaker": "SPEAKER_00"},
            {"start": 15.3, "end": 20, "speaker": "SPEAKER_00"},  # 0.3s gap
        ]
        result = group_speaker_segments(segments, min_gap=0.5, min_duration=10)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 20, "speaker": "SPEAKER_00"}

    def test_multiple_speakers_alternating(self):
        """Alternating speakers should result in separate segments."""
        segments = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_01"},
            {"start": 10, "end": 15, "speaker": "SPEAKER_00"},
            {"start": 15, "end": 20, "speaker": "SPEAKER_01"},
        ]
        result = group_speaker_segments(segments)
        assert len(result) == 4

    def test_min_duration_respected(self):
        """Segments shorter than min_duration should merge with next same-speaker segment."""
        segments = [
            {"start": 0, "end": 3, "speaker": "SPEAKER_00"},  # 3s < 10s min_duration
            {"start": 5, "end": 10, "speaker": "SPEAKER_00"},  # 2s gap
        ]
        result = group_speaker_segments(segments, min_gap=0.5, min_duration=10)
        # Should merge because current_duration (3s) < min_duration (10s)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 10, "speaker": "SPEAKER_00"}


class TestTranscriptPart:
    """Tests for the TranscriptPart dataclass."""

    def test_create_transcript_part(self):
        part = TranscriptPart(
            speaker="SPEAKER_00",
            start=0.0,
            end=10.5,
            text="Hello, world!"
        )
        assert part.speaker == "SPEAKER_00"
        assert part.start == 0.0
        assert part.end == 10.5
        assert part.text == "Hello, world!"


class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_create_transcription_result(self):
        result = TranscriptionResult(
            text="This is a transcript.",
            language="en"
        )
        assert result.text == "This is a transcript."
        assert result.language == "en"


class TestDiarizedTranscriptionResult:
    """Tests for the DiarizedTranscriptionResult dataclass."""

    def test_create_diarized_result(self):
        parts = [
            TranscriptPart(
                speaker="SPEAKER_00",
                start=0.0,
                end=5.0,
                text="Hello"
            ),
            TranscriptPart(
                speaker="SPEAKER_01",
                start=5.0,
                end=10.0,
                text="Hi there"
            ),
        ]
        result = DiarizedTranscriptionResult(
            parts=parts,
            language="en",
            speakers={"SPEAKER_00", "SPEAKER_01"}
        )
        assert len(result.parts) == 2
        assert result.language == "en"
        assert result.speakers == {"SPEAKER_00", "SPEAKER_01"}

    def test_format_transcript(self):
        parts = [
            TranscriptPart(
                speaker="SPEAKER_00",
                start=0.0,
                end=5.5,
                text="Hello"
            ),
            TranscriptPart(
                speaker="SPEAKER_01",
                start=5.5,
                end=10.0,
                text="Hi there"
            ),
        ]
        result = DiarizedTranscriptionResult(
            parts=parts,
            language="en",
            speakers={"SPEAKER_00", "SPEAKER_01"}
        )
        formatted = result.format_transcript()
        assert "[00:00:00.000 - 00:00:05.500] SPEAKER_00: Hello" in formatted
        assert "[00:00:05.500 - 00:00:10.000] SPEAKER_01: Hi there" in formatted


class TestDefaultConfiguration:
    """Tests for default configuration values."""

    def test_default_chunk_length(self):
        assert DEFAULT_CHUNK_LENGTH == 300  # 5 minutes

    def test_default_min_segment_duration(self):
        assert DEFAULT_MIN_SEGMENT_DURATION == 10

    def test_default_input_language(self):
        assert DEFAULT_INPUT_LANGUAGE == "en"


class TestCoreModuleImports:
    """Tests to ensure all public API is importable."""

    def test_import_main_functions(self):
        from core import transcribe_audio, transcribe_audio_with_diarization
        assert callable(transcribe_audio)
        assert callable(transcribe_audio_with_diarization)

    def test_import_utility_functions(self):
        from core import get_audio_duration, format_duration
        assert callable(get_audio_duration)
        assert callable(format_duration)

    def test_import_lower_level_functions(self):
        from core import (
            transcribe_chunk,
            chunk_audio,
            extract_audio_segment,
            detect_language_from_chunk,
            diarize_audio,
            group_speaker_segments,
        )
        assert callable(transcribe_chunk)
        assert callable(chunk_audio)
        assert callable(extract_audio_segment)
        assert callable(detect_language_from_chunk)
        assert callable(diarize_audio)
        assert callable(group_speaker_segments)

    def test_import_data_classes(self):
        from core import TranscriptPart, TranscriptionResult, DiarizedTranscriptionResult
        # Verify they can be instantiated
        part = TranscriptPart(speaker="test", start=0, end=1, text="test")
        result = TranscriptionResult(text="test", language="en")
        diarized = DiarizedTranscriptionResult(parts=[], language="en", speakers=set())
        assert part is not None
        assert result is not None
        assert diarized is not None
