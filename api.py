"""FastAPI backend for audio transcription with job queue and retention."""
import os
import uuid
import shutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler

from database import init_db, get_db, Job
from core import transcribe_audio_file, transcribe_audio_file_with_diarization

# Configure logging to avoid leaking sensitive data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories for job files
JOBS_DIR = Path("jobs")
JOBS_AUDIO_DIR = JOBS_DIR / "audio"
JOBS_OUTPUT_DIR = JOBS_DIR / "output"

# Create directories
JOBS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
JOBS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Audio to Notes API", version="1.0.0")

# In-memory job queue and worker state
job_queue: asyncio.Queue = asyncio.Queue()
api_keys_cache: Dict[str, str] = {}  # job_id -> api_key (memory only)
worker_task: Optional[asyncio.Task] = None


async def process_job(job_id: str, audio_path: str, diarize: bool, openai_api_key: str, db: Session):
    """Process a transcription job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found in database")
        return
    
    try:
        logger.info(f"Processing job {job_id} (diarize={diarize})")
        job.status = "processing"
        db.commit()
        
        # Run transcription (API key passed directly, never stored)
        if diarize:
            transcript_path = transcribe_audio_file_with_diarization(
                audio_path,
                output_dir=str(JOBS_OUTPUT_DIR),
                openai_api_key=openai_api_key
            )
        else:
            transcript_path = transcribe_audio_file(
                audio_path,
                output_dir=str(JOBS_OUTPUT_DIR),
                openai_api_key=openai_api_key
            )
        
        # Update job status
        job.status = "completed"
        job.transcript_filename = os.path.basename(transcript_path)
        db.commit()
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
    finally:
        # Remove API key from cache once job is done
        if job_id in api_keys_cache:
            del api_keys_cache[job_id]


async def worker():
    """Background worker to process jobs from the queue."""
    logger.info("Worker started")
    while True:
        try:
            # Get next job from queue
            job_data = await job_queue.get()
            job_id = job_data["job_id"]
            audio_path = job_data["audio_path"]
            diarize = job_data["diarize"]
            openai_api_key = job_data["openai_api_key"]
            
            # Get a new DB session for this job
            db = next(get_db())
            try:
                await process_job(job_id, audio_path, diarize, openai_api_key, db)
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            job_queue.task_done()


def cleanup_old_jobs():
    """Clean up jobs older than 2 hours (retention policy)."""
    logger.info("Running cleanup task for jobs older than 2 hours")
    db = next(get_db())
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=2)
        old_jobs = db.query(Job).filter(Job.created_at < cutoff_time).all()
        
        for job in old_jobs:
            logger.info(f"Cleaning up job {job.id} (created at {job.created_at})")
            
            # Delete audio file
            audio_path = JOBS_AUDIO_DIR / job.audio_filename
            if audio_path.exists():
                audio_path.unlink()
                logger.info(f"Deleted audio file: {audio_path}")
            
            # Delete transcript file if exists
            if job.transcript_filename:
                transcript_path = JOBS_OUTPUT_DIR / job.transcript_filename
                if transcript_path.exists():
                    transcript_path.unlink()
                    logger.info(f"Deleted transcript file: {transcript_path}")
            
            # Delete job from database
            db.delete(job)
        
        db.commit()
        logger.info(f"Cleanup complete: removed {len(old_jobs)} old jobs")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        db.rollback()
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    """Initialize database and start background worker."""
    global worker_task
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Start background worker
    worker_task = asyncio.create_task(worker())
    logger.info("Background worker started")
    
    # Start cleanup scheduler (runs every 30 minutes)
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_old_jobs, 'interval', minutes=30)
    scheduler.start()
    logger.info("Cleanup scheduler started (runs every 30 minutes)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global worker_task
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    logger.info("Application shutdown complete")


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    diarize: bool = False,
    x_openai_api_key: Optional[str] = Header(None, alias="X-OpenAI-API-Key"),
    db: Session = Depends(get_db)
):
    """Create a new transcription job.
    
    Args:
        file: Audio file to transcribe
        diarize: Whether to enable speaker diarization (default: False)
        x_openai_api_key: OpenAI API key (passed in header, NEVER stored)
        
    Returns:
        Job ID and initial status
    """
    # Validate API key is provided
    if not x_openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required. Pass it in X-OpenAI-API-Key header."
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    audio_filename = f"{job_id}_{file.filename}"
    audio_path = JOBS_AUDIO_DIR / audio_filename
    
    try:
        with audio_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio file")
    
    # Create job in database (API key is NOT stored)
    job = Job(
        id=job_id,
        status="queued",
        audio_filename=audio_filename,
        diarize=diarize,
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    
    # Store API key in memory cache only (not persisted)
    api_keys_cache[job_id] = x_openai_api_key
    
    # Add job to queue
    await job_queue.put({
        "job_id": job_id,
        "audio_path": str(audio_path),
        "diarize": diarize,
        "openai_api_key": x_openai_api_key  # Passed to worker, not stored
    })
    
    logger.info(f"Created job {job_id} for file {file.filename}")
    
    return {
        "job_id": job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat()
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status of a transcription job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Job status information
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "diarize": job.diarize
    }
    
    if job.status == "failed" and job.error_message:
        response["error"] = job.error_message
    
    if job.status == "completed" and job.transcript_filename:
        response["transcript_available"] = True
    
    return response


@app.get("/jobs/{job_id}/download")
async def download_transcript(job_id: str, db: Session = Depends(get_db)):
    """Download the transcript for a completed job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Transcript file
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    if not job.transcript_filename:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    transcript_path = JOBS_OUTPUT_DIR / job.transcript_filename
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    
    return FileResponse(
        path=str(transcript_path),
        filename=job.transcript_filename,
        media_type="text/plain"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "queue_size": job_queue.qsize()
    }
