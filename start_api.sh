#!/bin/bash
# Startup script for the FastAPI backend

echo "Starting Audio to Notes API Server..."
echo "======================================="
echo ""
echo "Environment variables:"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}... (${#OPENAI_API_KEY} chars)"
echo "  HF_TOKEN: ${HF_TOKEN:0:10}... (${#HF_TOKEN} chars)"
echo ""

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set"
    echo "Jobs requiring transcription will need to provide the API key in the request header"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set"
    echo "Speaker diarization will not be available"
fi

echo ""
echo "Starting server on http://0.0.0.0:8000"
echo "API documentation available at http://0.0.0.0:8000/docs"
echo ""

# Start the server
exec uvicorn api:app --host 0.0.0.0 --port 8000
