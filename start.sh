#!/bin/bash
# Startup script for Render.com

# If chroma_data is missing, we could run ingest here, 
# but for now we assume it's part of the repo/bundle.
echo "Starting Slack Draft Agent..."
python app.py
