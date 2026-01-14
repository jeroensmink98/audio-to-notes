import os
import subprocess
import time
import tempfile
from glob import glob
import argparse
from openai import OpenAI
import whisper
import torch

# CONFGIURATION
CHUNK_LENGTH = 300  # seconds (5 minutes)
CHUNKS_DIR = "chunks"
AUDIO_DIR = "input"  # Default input directory
OUTPUT_DIR = "output"  # Default output directory
INPUT_LANGUAGE = "en"  # Default input language (fallback)
MIN_SEGMENT_DURATION = (
    10  # Minimum segment duration (seconds) for diarized transcription
)


def get_audio_duration(filepath):
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


def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def diarize_audio(audio_file):
    """Identify speakers and their time segments in audio using pyannote.audio."""
    from pyannote.audio import Pipeline
    import torchaudio

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required for speaker diarization"
        )

    print("Loading speaker diarization model...")
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
            print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
        except Exception as e:
            print(f"GPU detected but not compatible: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    else:
        print("No GPU available, using CPU")

    try:
        pipeline.to(device)
    except Exception as e:
        print(f"Failed to load pipeline on {device}: {e}")
        print("Falling back to CPU")
        device = torch.device("cpu")
        pipeline.to(device)

    # Preload audio using torchaudio to avoid torchcodec issues
    print("Loading audio file...")
    waveform, sample_rate = torchaudio.load(audio_file)
    # Convert to format pyannote expects: dict with 'waveform' and 'sample_rate'
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    print("Running speaker diarization...")
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


def group_speaker_segments(segments, min_gap=0.5, min_duration=MIN_SEGMENT_DURATION):
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


def extract_audio_segment(audio_file, start, end):
    """Extract a specific time range from audio file using ffmpeg.

    Returns path to temporary file containing the segment.
    Raises RuntimeError if ffmpeg command fails.
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


def chunk_audio(filepath, chunk_length=CHUNK_LENGTH, duration=None):
    """Split audio file into chunks using ffmpeg."""
    basename, ext = os.path.splitext(os.path.basename(filepath))
    ext = ext.lstrip(".")  # Remove leading dot
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    if duration is None:
        duration = get_audio_duration(filepath)
    chunk_paths = []
    # Always output chunks as .mp3 for OpenAI Whisper API compatibility
    # This ensures .amr and other formats are converted properly
    output_ext = "mp3"
    for i, start in enumerate(range(0, int(duration), chunk_length), 1):
        chunk_path = os.path.join(CHUNKS_DIR, f"{basename}_chunk_{i:03d}.{output_ext}")
        # Use mp3 encoding for compatibility with OpenAI Whisper API
        # This handles .amr and other formats that need conversion
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


def detect_language_from_chunk(chunk_path, detection_duration=30):
    """Detect language from first N seconds of audio chunk using local Whisper model."""
    import tempfile

    temp_file = None
    try:
        print(
            f"Detecting language from first {detection_duration} seconds of {chunk_path}..."
        )
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
        print(f"Detected language: {detected_language}")
        return detected_language
    except Exception as e:
        print(
            f"Language detection failed: {e}. Using fallback language: {INPUT_LANGUAGE}"
        )
        return INPUT_LANGUAGE
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def transcribe_audio_file(audio_file):
    """Chunk the audio file, transcribe each chunk, and combine into a single text file."""
    duration = get_audio_duration(audio_file)
    print(f"Duration: {format_duration(duration)}")
    chunk_paths = chunk_audio(audio_file, duration=duration)
    print(f"Created {len(chunk_paths)} chunks.")

    # Detect language from first chunk
    detected_language = None
    if chunk_paths:
        detected_language = detect_language_from_chunk(chunk_paths[0])
    else:
        detected_language = INPUT_LANGUAGE

    transcript = ""
    for chunk_path in chunk_paths:
        while True:
            try:
                print(f"Transcribing {chunk_path} ...")
                text = transcribe_chunk(chunk_path, language=detected_language)
                transcript += text + "\n"
                break
            except Exception as e:
                print(f"Error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
    # Write transcript to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_file))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_transcript.txt")
    with open(output_path, "w") as f:
        f.write(transcript)
    print(f"Transcript written to {output_path}")


def transcribe_chunk(chunk_path, language=None):
    client = OpenAI()  # Automatically reads from OPENAI_API_KEY environment variable
    with open(chunk_path, "rb") as audio_file:
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language
        transcript = client.audio.transcriptions.create(**kwargs)
    return transcript.text


def transcribe_audio_file_with_diarization(audio_file):
    """Transcribe audio file with speaker diarization."""
    duration = get_audio_duration(audio_file)
    print(f"Duration: {format_duration(duration)}")

    # Step 1: Run speaker diarization
    segments = diarize_audio(audio_file)
    unique_speakers = set(s["speaker"] for s in segments)
    print(
        f"Found {len(unique_speakers)} speakers: {', '.join(sorted(unique_speakers))}"
    )

    # Step 2: Group consecutive segments from same speaker
    grouped_segments = group_speaker_segments(segments)
    print(f"Grouped into {len(grouped_segments)} speech segments")

    # Step 3: Detect language from first 30 seconds of the audio
    # (not from the first speaker segment, which may be too short)
    first_segment_path = extract_audio_segment(audio_file, 0, 30)
    detected_language = detect_language_from_chunk(first_segment_path)
    os.unlink(first_segment_path)
    print(f"Using detected language: {detected_language}")

    # Step 4: Transcribe each segment
    transcript_parts = []
    temp_files = []

    try:
        for i, segment in enumerate(grouped_segments):
            print(
                f"Transcribing segment {i + 1}/{len(grouped_segments)} ({segment['speaker']}: {format_duration(segment['start'])} - {format_duration(segment['end'])})..."
            )

            # Extract segment audio
            segment_path = extract_audio_segment(
                audio_file, segment["start"], segment["end"]
            )
            temp_files.append(segment_path)

            # Transcribe with retry
            while True:
                try:
                    text = transcribe_chunk(segment_path, language=detected_language)
                    transcript_parts.append(
                        {
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": text.strip(),
                        }
                    )
                    break
                except Exception as e:
                    print(f"Error: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
    finally:
        # Clean up temp files (always executed, even if exception occurs)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    # Step 5: Format output with speaker labels
    formatted_transcript = ""
    for part in transcript_parts:
        start_str = format_duration(part["start"])
        end_str = format_duration(part["end"])
        formatted_transcript += (
            f"[{start_str} - {end_str}] {part['speaker']}: {part['text']}\n"
        )

    # Write transcript to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_file))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_transcript_diarized.txt")
    with open(output_path, "w") as f:
        f.write(formatted_transcript)
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
