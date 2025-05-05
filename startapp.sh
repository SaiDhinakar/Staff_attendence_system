#!/bin/bash

# Script to start both the backend Python service and Django web application
# for the Staff Face Recognition System

# Start the backend Python process
echo "Starting backend service..."
python backend/Updated.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to initialize properly
echo "Waiting for backend to initialize..."
sleep 5  # Increased wait time to ensure backend is ready

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "Backend started successfully. Starting Django server..."
    
    # Start the Django application
    python StaffFaceRecognition/manage.py runserver 0.0.0.0:8000
    
    # When Django server stops, also kill the backend process
    echo "Stopping backend process..."
    kill $BACKEND_PID 2>/dev/null || echo "Backend process already stopped"
else
    echo "Backend failed to start. Check backend.log for details."
    exit 1
fi
