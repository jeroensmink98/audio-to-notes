# Audio to Notes

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
