import os
import subprocess
import time
import openai
from glob import glob
import argparse
from lib.openai import transcribe_chunk

# CONFGIURATION
CHUNK_LENGTH = 300  # seconds (5 minutes)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHUNKS_DIR = 'chunks'
AUDIO_DIR = 'input'  # Default input directory
OUTPUT_DIR = 'output'  # Default output directory
INPUT_LANGUAGE = 'nl'  # Default input language


def get_audio_duration(filepath):
    """Get duration of audio file in seconds using ffprobe."""
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', filepath
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def chunk_audio(filepath, chunk_length=CHUNK_LENGTH):
    """Split audio file into chunks using ffmpeg."""
    basename, ext = os.path.splitext(os.path.basename(filepath))
    ext = ext.lstrip('.')  # Remove leading dot
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    duration = get_audio_duration(filepath)
    chunk_paths = []
    for i, start in enumerate(range(0, int(duration), chunk_length), 1):
        chunk_path = os.path.join(
            CHUNKS_DIR, f"{basename}_chunk_{i:03d}.{ext}"
        )
        cmd = [
            'ffmpeg', '-y', '-i', filepath, '-ss', str(start), '-t', str(chunk_length),
            '-c', 'copy', chunk_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunk_paths.append(chunk_path)
    return chunk_paths

def transcribe_audio_file(audio_file):
    """Chunk the audio file, transcribe each chunk, and combine into a single text file."""
    chunk_paths = chunk_audio(audio_file)
    print(f"Created {len(chunk_paths)} chunks.")
    transcript = ''
    for chunk_path in chunk_paths:
        while True:
            try:
                print(f"Transcribing {chunk_path} ...")
                text = transcribe_chunk(chunk_path, language=INPUT_LANGUAGE)
                transcript += text + '\n'
                break
            except Exception as e:
                print(f"Error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
    # Write transcript to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_file))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_transcript.txt")
    with open(output_path, 'w') as f:
        f.write(transcript)
    print(f"Transcript written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Audio to Notes Script')
    parser.add_argument('audio_file', nargs='?', help='Path to a single audio file to process (.mp3, .m4a, etc)')
    args = parser.parse_args()

    if args.audio_file:
        audio_files = [args.audio_file]
    else:
        # Find audio files in input directory (supporting .m4a, .mp3, .mp4, .wav, .aac, .flac)
        audio_files = []
        for ext in ('*.m4a', '*.mp3', '*.mp4', '*.wav', '*.aac', '*.flac'):
            audio_files.extend(glob(os.path.join(AUDIO_DIR, ext)))
    if not audio_files:
        print('No audio files found to process.')
        return
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        transcribe_audio_file(audio_file)
    print("All done!")

if __name__ == '__main__':
    main() 
