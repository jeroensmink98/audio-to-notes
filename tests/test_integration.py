"""
Integration tests for the core transcription module.

These tests use the test audio file from tests/inputs and
mock external API calls.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from core import (
    get_audio_duration,
    format_duration,
    chunk_audio,
    extract_audio_segment,
    transcribe_chunk,
    transcribe_audio,
    TranscriptionResult,
)


# Path to test audio file
TEST_AUDIO_FILE = os.path.join(
    os.path.dirname(__file__), "inputs", "jfk_speech_input.mp3"
)


def is_ffmpeg_available():
    """Check if ffmpeg/ffprobe is available."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def test_audio_exists():
    """Check if the test audio file exists."""
    if not os.path.exists(TEST_AUDIO_FILE):
        pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")
    return TEST_AUDIO_FILE


@pytest.fixture
def requires_ffmpeg():
    """Skip test if ffmpeg is not available."""
    if not is_ffmpeg_available():
        pytest.skip("ffmpeg not available")


@pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg not available")
class TestGetAudioDuration:
    """Tests for get_audio_duration using actual audio file."""

    def test_get_audio_duration_real_file(self, test_audio_exists):
        """Test getting duration from a real audio file."""
        duration = get_audio_duration(test_audio_exists)
        # The JFK speech is approximately 17 minutes (1020 seconds)
        assert duration > 900  # At least 15 minutes
        assert duration < 1200  # At most 20 minutes


@pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg not available")
class TestChunkAudio:
    """Tests for chunk_audio using actual audio file."""

    def test_chunk_audio_real_file(self, test_audio_exists):
        """Test chunking a real audio file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a short chunk length for testing
            chunk_length = 60  # 1 minute chunks
            chunks = chunk_audio(
                test_audio_exists,
                chunks_dir=temp_dir,
                chunk_length=chunk_length
            )
            
            # Should create multiple chunks
            assert len(chunks) > 1
            
            # All chunk files should exist
            for chunk in chunks:
                assert os.path.exists(chunk)
            
            # Chunks should be in the temp directory
            for chunk in chunks:
                assert chunk.startswith(temp_dir)


@pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg not available")
class TestExtractAudioSegment:
    """Tests for extract_audio_segment using actual audio file."""

    def test_extract_audio_segment_real_file(self, test_audio_exists):
        """Test extracting a segment from a real audio file."""
        segment_path = extract_audio_segment(test_audio_exists, 10, 20)
        
        try:
            # Segment file should exist
            assert os.path.exists(segment_path)
            
            # Segment should have reasonable size
            segment_size = os.path.getsize(segment_path)
            assert segment_size > 0
            
            # Segment duration should be approximately 10 seconds
            segment_duration = get_audio_duration(segment_path)
            assert segment_duration >= 9
            assert segment_duration <= 11
        finally:
            # Clean up
            if os.path.exists(segment_path):
                os.unlink(segment_path)


@pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg not available")
class TestTranscribeChunk:
    """Tests for transcribe_chunk with mocked OpenAI client."""

    def test_transcribe_chunk_mocked(self, test_audio_exists):
        """Test transcribe_chunk with mocked OpenAI API."""
        # Create mock OpenAI client
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = MagicMock(
            text="This is a test transcription."
        )
        
        # Use a small segment for testing
        segment_path = extract_audio_segment(test_audio_exists, 0, 5)
        
        try:
            result = transcribe_chunk(
                segment_path,
                openai_client=mock_client,
                language="en"
            )
            
            assert result == "This is a test transcription."
            
            # Verify API was called correctly
            mock_client.audio.transcriptions.create.assert_called_once()
            call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
            assert call_kwargs["model"] == "whisper-1"
            assert call_kwargs["language"] == "en"
        finally:
            if os.path.exists(segment_path):
                os.unlink(segment_path)


@pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg not available")
class TestTranscribeAudio:
    """Tests for the main transcribe_audio function with mocked API."""

    def test_transcribe_audio_mocked(self, test_audio_exists):
        """Test full transcription with mocked OpenAI API."""
        # Create mock OpenAI client
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = MagicMock(
            text="Test transcription text."
        )
        
        progress_messages = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock whisper model for language detection
            with patch("core.transcription.whisper") as mock_whisper:
                mock_model = MagicMock()
                mock_model.transcribe.return_value = {"language": "en"}
                mock_whisper.load_model.return_value = mock_model
                
                result = transcribe_audio(
                    audio_file=test_audio_exists,
                    openai_client=mock_client,
                    chunks_dir=temp_dir,
                    chunk_length=300,  # 5 minute chunks
                    progress_callback=progress_messages.append
                )
        
        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert "Test transcription text." in result.text
        assert result.language == "en"
        
        # Verify progress callback was called
        assert len(progress_messages) > 0
        # Should include duration message
        assert any("Duration" in msg for msg in progress_messages)
