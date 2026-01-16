#!/usr/bin/env python3
"""Test client for the FastAPI backend."""
import sys
import time
import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def create_job(audio_file: str, api_key: str, diarize: bool = False):
    """Create a transcription job."""
    print(f"\nCreating job for {audio_file}...")
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file {audio_file} not found")
        return None
    
    headers = {"X-OpenAI-API-Key": api_key}
    data = {"diarize": str(diarize).lower()}
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_BASE_URL}/jobs", headers=headers, files=files, data=data)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        job_data = response.json()
        print(f"Job created: {json.dumps(job_data, indent=2)}")
        return job_data["job_id"]
    else:
        print(f"Error: {response.text}")
        return None


def get_job_status(job_id: str):
    """Get job status."""
    response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting status: {response.text}")
        return None


def wait_for_job(job_id: str, max_wait: int = 300):
    """Wait for job to complete."""
    print(f"\nWaiting for job {job_id} to complete...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status_data = get_job_status(job_id)
        if not status_data:
            return False
        
        status = status_data["status"]
        print(f"Status: {status}")
        
        if status == "completed":
            print("Job completed successfully!")
            return True
        elif status == "failed":
            print(f"Job failed: {status_data.get('error', 'Unknown error')}")
            return False
        
        time.sleep(5)  # Poll every 5 seconds
    
    print("Timeout waiting for job")
    return False


def download_transcript(job_id: str, output_file: str = None):
    """Download transcript."""
    print(f"\nDownloading transcript for job {job_id}...")
    response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/download")
    
    if response.status_code == 200:
        if not output_file:
            # Get filename from Content-Disposition header
            output_file = f"transcript_{job_id}.txt"
        
        with open(output_file, "w") as f:
            f.write(response.text)
        
        print(f"Transcript saved to {output_file}")
        print(f"Preview (first 500 chars):\n{response.text[:500]}")
        return True
    else:
        print(f"Error downloading transcript: {response.text}")
        return False


def main():
    """Run test workflow."""
    if len(sys.argv) < 3:
        print("Usage: python test_api.py <audio_file> <openai_api_key> [diarize]")
        print("Example: python test_api.py audio.mp3 sk-... false")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    api_key = sys.argv[2]
    diarize = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    
    # Test health
    if not test_health():
        print("Health check failed. Is the server running?")
        sys.exit(1)
    
    # Create job
    job_id = create_job(audio_file, api_key, diarize)
    if not job_id:
        sys.exit(1)
    
    # Wait for completion
    if not wait_for_job(job_id):
        sys.exit(1)
    
    # Download transcript
    if download_transcript(job_id):
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
