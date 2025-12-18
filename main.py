import os
import subprocess
import time
from glob import glob
import argparse
from openai import OpenAI
import whisper



# CONFGIURATION
CHUNK_LENGTH = 300  # seconds (5 minutes)
CHUNKS_DIR = "chunks"
AUDIO_DIR = "input"  # Default input directory
OUTPUT_DIR = "output"  # Default output directory
INPUT_LANGUAGE = "nl"  # Default input language


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
    """Format duration in seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        print(f"Detecting language from first {detection_duration} seconds of {chunk_path}...")
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
        print(f"Language detection failed: {e}. Using fallback language: {INPUT_LANGUAGE}")
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
    args = parser.parse_args()

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
        transcribe_audio_file(audio_file)
    print("All done!")





if __name__ == "__main__":
    main()
