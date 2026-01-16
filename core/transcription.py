"""
Core transcription module for audio-to-notes.

This module provides pure transcription logic as a reusable library,
without any CLI, UI, or HTTP concepts.
"""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import whisper
from openai import OpenAI

# Default configuration
DEFAULT_CHUNK_LENGTH = 300  # seconds (5 minutes)
DEFAULT_MIN_SEGMENT_DURATION = 10  # Minimum segment duration (seconds) for diarization
DEFAULT_INPUT_LANGUAGE = "en"  # Default input language (fallback)


@dataclass
class TranscriptPart:
    """Represents a transcribed segment with speaker information."""

    speaker: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    language: str


@dataclass
class DiarizedTranscriptionResult:
    """Result of a diarized transcription operation."""

    parts: list[TranscriptPart]
    language: str
    speakers: set[str]

    def format_transcript(self) -> str:
        """Format the diarized transcript with speaker labels and timestamps."""
        formatted = ""
        for part in self.parts:
            start_str = format_duration(part.start)
            end_str = format_duration(part.end)
            formatted += f"[{start_str} - {end_str}] {part.speaker}: {part.text}\n"
        return formatted


def get_audio_duration(filepath: str) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filepath,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return float(result.stdout.strip())


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def diarize_audio(
    audio_file: str,
    hf_token: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Identify speakers and their time segments in audio using pyannote.audio.

    Args:
        audio_file: Path to the audio file
        hf_token: HuggingFace API token for pyannote.audio
        progress_callback: Optional callback function for progress messages

    Returns:
        List of segments with start, end, and speaker information
    """
    from pyannote.audio import Pipeline
    import torchaudio

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    log("Loading speaker diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )

    # Auto-detect GPU/CPU with compatibility check
    device = torch.device("cpu")  # Default to CPU
    if torch.cuda.is_available():
        try:
            # Try to create a test tensor on GPU to check compatibility
            test_tensor = torch.zeros(1).cuda()
            device = torch.device("cuda")
            log(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
        except Exception as e:
            log(f"GPU detected but not compatible: {e}")
            log("Falling back to CPU")
            device = torch.device("cpu")
    else:
        log("No GPU available, using CPU")

    try:
        pipeline.to(device)
    except Exception as e:
        log(f"Failed to load pipeline on {device}: {e}")
        log("Falling back to CPU")
        device = torch.device("cpu")
        pipeline.to(device)

    # Preload audio using torchaudio to avoid torchcodec issues
    log("Loading audio file...")
    waveform, sample_rate = torchaudio.load(audio_file)
    # Convert to format pyannote expects: dict with 'waveform' and 'sample_rate'
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    log("Running speaker diarization...")
    diarization_output = pipeline(audio_input)

    # Access the Annotation object from DiarizeOutput
    # (newer pyannote.audio versions return DiarizeOutput instead of Annotation)
    if hasattr(diarization_output, "speaker_diarization"):
        diarization = diarization_output.speaker_diarization
    else:
        diarization = diarization_output

    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return segments


def group_speaker_segments(
    segments: list[dict],
    min_gap: float = 0.5,
    min_duration: float = DEFAULT_MIN_SEGMENT_DURATION,
) -> list[dict]:
    """Merge consecutive segments from the same speaker.

    Args:
        segments: List of {start, end, speaker} dicts
        min_gap: Maximum gap (seconds) between segments to merge
        min_duration: Minimum duration (seconds) for a segment before allowing a new segment

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    grouped = []
    current = segments[0].copy()

    for segment in segments[1:]:
        current_duration = current["end"] - current["start"]
        gap = segment["start"] - current["end"]

        # Merge if same speaker AND (duration < min_duration OR gap <= min_gap)
        if segment["speaker"] == current["speaker"]:
            if current_duration < min_duration or gap <= min_gap:
                current["end"] = segment["end"]
            else:
                # Same speaker but duration reached and gap is large enough
                grouped.append(current)
                current = segment.copy()
        else:
            # Different speaker - always start new segment
            grouped.append(current)
            current = segment.copy()

    grouped.append(current)
    return grouped


def extract_audio_segment(audio_file: str, start: float, end: float) -> str:
    """Extract a specific time range from audio file using ffmpeg.

    Args:
        audio_file: Path to the source audio file
        start: Start time in seconds
        end: End time in seconds

    Returns:
        Path to temporary file containing the segment

    Raises:
        RuntimeError: If ffmpeg command fails
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_file.close()

    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_file,
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-acodec",
        "libmp3lame",
        "-ar",
        "44100",
        "-ac",
        "2",
        temp_file.name,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        # Clean up the temporary file if ffmpeg failed
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg failed to extract audio segment from {start}s to {end}s. "
            f"Error: {result.stderr.decode('utf-8', errors='ignore')}"
        )

    return temp_file.name


def chunk_audio(
    filepath: str,
    chunks_dir: str,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    duration: Optional[float] = None,
) -> list[str]:
    """Split audio file into chunks using ffmpeg.

    Args:
        filepath: Path to the audio file
        chunks_dir: Directory to store chunk files
        chunk_length: Length of each chunk in seconds
        duration: Pre-computed duration (optional)

    Returns:
        List of paths to chunk files
    """
    basename, ext = os.path.splitext(os.path.basename(filepath))
    os.makedirs(chunks_dir, exist_ok=True)
    if duration is None:
        duration = get_audio_duration(filepath)
    chunk_paths = []
    # Always output chunks as .mp3 for OpenAI Whisper API compatibility
    output_ext = "mp3"
    for i, start in enumerate(range(0, int(duration), chunk_length), 1):
        chunk_path = os.path.join(chunks_dir, f"{basename}_chunk_{i:03d}.{output_ext}")
        # Use mp3 encoding for compatibility with OpenAI Whisper API
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            filepath,
            "-ss",
            str(start),
            "-t",
            str(chunk_length),
            "-acodec",
            "libmp3lame",
            "-ar",
            "44100",
            "-ac",
            "2",
            chunk_path,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunk_paths.append(chunk_path)
    return chunk_paths


def detect_language_from_chunk(
    chunk_path: str,
    detection_duration: int = 30,
    fallback_language: str = DEFAULT_INPUT_LANGUAGE,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Detect language from first N seconds of audio chunk using local Whisper model.

    Args:
        chunk_path: Path to the audio chunk
        detection_duration: Number of seconds to use for detection
        fallback_language: Language to use if detection fails
        progress_callback: Optional callback function for progress messages

    Returns:
        Detected language code (e.g., 'en', 'nl', 'de')
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    temp_file = None
    try:
        log(f"Detecting language from first {detection_duration} seconds of {chunk_path}...")
        # Extract first N seconds to a temporary file for faster detection
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_file.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            chunk_path,
            "-t",
            str(detection_duration),
            "-acodec",
            "libmp3lame",
            temp_file.name,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        model = whisper.load_model("tiny")  # Use 'tiny' for fastest detection
        result = model.transcribe(temp_file.name, language=None)  # None = auto-detect
        detected_language = result["language"]
        log(f"Detected language: {detected_language}")
        return detected_language
    except Exception as e:
        log(f"Language detection failed: {e}. Using fallback language: {fallback_language}")
        return fallback_language
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def transcribe_chunk(
    chunk_path: str,
    openai_client: OpenAI,
    language: Optional[str] = None,
) -> str:
    """Transcribe a single audio chunk using OpenAI Whisper API.

    Args:
        chunk_path: Path to the audio chunk
        openai_client: Initialized OpenAI client
        language: Optional language code for transcription

    Returns:
        Transcribed text
    """
    with open(chunk_path, "rb") as audio_file:
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language
        transcript = openai_client.audio.transcriptions.create(**kwargs)
    return transcript.text


def transcribe_audio(
    audio_file: str,
    openai_client: OpenAI,
    chunks_dir: str,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    retry_delay: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TranscriptionResult:
    """Transcribe an audio file by chunking and transcribing each part.

    Args:
        audio_file: Path to the audio file
        openai_client: Initialized OpenAI client
        chunks_dir: Directory to store temporary chunk files
        chunk_length: Length of each chunk in seconds
        retry_delay: Delay in seconds between retry attempts on failure
        progress_callback: Optional callback function for progress messages

    Returns:
        TranscriptionResult containing the full transcript and detected language
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    duration = get_audio_duration(audio_file)
    log(f"Duration: {format_duration(duration)}")
    chunk_paths = chunk_audio(audio_file, chunks_dir, chunk_length, duration)
    log(f"Created {len(chunk_paths)} chunks.")

    # Detect language from first chunk
    detected_language = DEFAULT_INPUT_LANGUAGE
    if chunk_paths:
        detected_language = detect_language_from_chunk(
            chunk_paths[0], progress_callback=progress_callback
        )

    transcript = ""
    for chunk_path in chunk_paths:
        while True:
            try:
                log(f"Transcribing {chunk_path} ...")
                text = transcribe_chunk(chunk_path, openai_client, detected_language)
                transcript += text + "\n"
                break
            except Exception as e:
                log(f"Error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    return TranscriptionResult(text=transcript, language=detected_language)


def transcribe_audio_with_diarization(
    audio_file: str,
    openai_client: OpenAI,
    hf_token: str,
    min_segment_duration: float = DEFAULT_MIN_SEGMENT_DURATION,
    retry_delay: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> DiarizedTranscriptionResult:
    """Transcribe an audio file with speaker diarization.

    Args:
        audio_file: Path to the audio file
        openai_client: Initialized OpenAI client
        hf_token: HuggingFace API token for pyannote.audio
        min_segment_duration: Minimum segment duration for diarization
        retry_delay: Delay in seconds between retry attempts on failure
        progress_callback: Optional callback function for progress messages

    Returns:
        DiarizedTranscriptionResult containing transcript parts with speaker labels
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    duration = get_audio_duration(audio_file)
    log(f"Duration: {format_duration(duration)}")

    # Step 1: Run speaker diarization
    segments = diarize_audio(audio_file, hf_token, progress_callback)
    unique_speakers = set(s["speaker"] for s in segments)
    log(f"Found {len(unique_speakers)} speakers: {', '.join(sorted(unique_speakers))}")

    # Step 2: Group consecutive segments from same speaker
    grouped_segments = group_speaker_segments(
        segments, min_duration=min_segment_duration
    )
    log(f"Grouped into {len(grouped_segments)} speech segments")

    # Step 3: Detect language from first 30 seconds of the audio
    first_segment_path = extract_audio_segment(audio_file, 0, 30)
    detected_language = detect_language_from_chunk(
        first_segment_path, progress_callback=progress_callback
    )
    os.unlink(first_segment_path)
    log(f"Using detected language: {detected_language}")

    # Step 4: Transcribe each segment
    transcript_parts = []
    temp_files = []

    try:
        for i, segment in enumerate(grouped_segments):
            log(
                f"Transcribing segment {i + 1}/{len(grouped_segments)} "
                f"({segment['speaker']}: {format_duration(segment['start'])} - "
                f"{format_duration(segment['end'])})..."
            )

            # Extract segment audio
            segment_path = extract_audio_segment(
                audio_file, segment["start"], segment["end"]
            )
            temp_files.append(segment_path)

            # Transcribe with retry
            while True:
                try:
                    text = transcribe_chunk(
                        segment_path, openai_client, detected_language
                    )
                    transcript_parts.append(
                        TranscriptPart(
                            speaker=segment["speaker"],
                            start=segment["start"],
                            end=segment["end"],
                            text=text.strip(),
                        )
                    )
                    break
                except Exception as e:
                    log(f"Error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    return DiarizedTranscriptionResult(
        parts=transcript_parts,
        language=detected_language,
        speakers=unique_speakers,
    )
