# Audio to Notes

Transcribes audio files using OpenAI Whisper API.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- [ffmpeg](https://ffmpeg.org/) (for audio processing)
- OpenAI API key

## Usage

Create a `.env` file with your `OPENAI_API_KEY`:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

Run the script:

```bash
uv run --env-file .env main.py <audio_file>
```

## How it works

The audio file is split into 5-minute chunks using ffmpeg, each chunk is transcribed via OpenAI Whisper, and the transcripts are combined into a single output file.

## Cost & Output

- **Cost**: ~$0.36 per hour of audio ($0.006 per minute)
- **Output**: Transcripts are saved to the `output/` directory
