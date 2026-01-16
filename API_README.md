# FastAPI Backend for Audio to Notes

This is the FastAPI backend that provides a REST API for audio transcription with job queue management and automatic retention.

## Features

- **Job Queue**: Asynchronous background processing of transcription jobs
- **SQLite Persistence**: Jobs are tracked in a SQLite database
- **Retention Policy**: Automatic cleanup of jobs older than 2 hours
- **Secure**: API keys are never stored or logged, only kept in memory during processing
- **Speaker Diarization**: Optional speaker identification support

## API Endpoints

### POST /jobs
Create a new transcription job.

**Headers:**
- `X-OpenAI-API-Key`: Your OpenAI API key (required, not stored)

**Form Data:**
- `file`: Audio file to transcribe (required)
- `diarize`: Enable speaker diarization (optional, default: false)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "created_at": "2026-01-16T14:00:00"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/jobs" \
  -H "X-OpenAI-API-Key: sk-..." \
  -F "file=@audio.mp3" \
  -F "diarize=false"
```

### GET /jobs/{job_id}
Get the status of a transcription job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "created_at": "2026-01-16T14:00:00",
  "diarize": false,
  "transcript_available": true
}
```

Possible statuses: `queued`, `processing`, `completed`, `failed`

**Example:**
```bash
curl http://localhost:8000/jobs/{job_id}
```

### GET /jobs/{job_id}/download
Download the transcript for a completed job.

**Response:** Text file containing the transcript

**Example:**
```bash
curl -O -J http://localhost:8000/jobs/{job_id}/download
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "queue_size": 0
}
```

## Running the Server

### Option 1: Using uvicorn directly
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Option 2: With auto-reload for development
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: With environment variables
```bash
# For diarization support
export HF_TOKEN=your_huggingface_token
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Configuration

The backend uses the following configuration:

- **Database**: `jobs.db` (SQLite)
- **Audio Files**: Stored in `jobs/audio/`
- **Transcripts**: Stored in `jobs/output/`
- **Retention**: Jobs are automatically deleted after 2 hours
- **Cleanup Interval**: Every 30 minutes

## Security

- **API Keys**: OpenAI API keys are passed via the `X-OpenAI-API-Key` header and are NEVER stored in the database or logs
- **In-Memory Only**: API keys are only kept in memory during job processing and immediately deleted afterwards
- **Safe Logging**: All logging is configured to avoid leaking sensitive data

## Requirements

All dependencies are listed in `pyproject.toml`:
- FastAPI
- Uvicorn
- SQLAlchemy
- aiosqlite
- APScheduler
- python-multipart

## Architecture

1. **Job Creation**: Client uploads audio file and provides API key
2. **Queueing**: Job is added to in-memory queue and database
3. **Background Processing**: Worker task processes jobs asynchronously
4. **Status Polling**: Client can check job status at any time
5. **Download**: Once completed, transcript can be downloaded
6. **Cleanup**: Jobs older than 2 hours are automatically deleted

## Example Workflow

```bash
# 1. Create a job
RESPONSE=$(curl -X POST "http://localhost:8000/jobs" \
  -H "X-OpenAI-API-Key: sk-..." \
  -F "file=@audio.mp3" \
  -F "diarize=false")
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

# 2. Check status
curl http://localhost:8000/jobs/$JOB_ID

# 3. Download when completed
curl -O -J http://localhost:8000/jobs/$JOB_ID/download
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200 OK`: Success
- `400 Bad Request`: Missing or invalid parameters
- `404 Not Found`: Job not found
- `500 Internal Server Error`: Server error

Error responses include a detail message:
```json
{
  "detail": "Error description"
}
```
