#!/bin/bash

# Setup script for Staff Attendance System

set -e

echo "=== Staff Attendance System Setup ==="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p media/profile_pics
mkdir -p StaffFaceRecognition/backend/dataset

# Create .env file if it doesn't exist
if [ ! -f "StaffFaceRecognition/.env" ]; then
    echo "Creating .env file..."
    echo "IP='127.0.0.1'" > StaffFaceRecognition/.env
    echo "✓ .env file created"
else
    echo "✓ .env file already exists"
fi

# Create face_embeddings.json if it doesn't exist
if [ ! -f "face_embeddings.json" ]; then
    echo "Creating face_embeddings.json..."
    echo "{}" > face_embeddings.json
    echo "✓ face_embeddings.json created"
else
    echo "✓ face_embeddings.json already exists"
fi

# Create empty database if it doesn't exist
if [ ! -f "StaffFaceRecognition/db.sqlite3" ]; then
    echo "Database will be created on first run"
else
    echo "✓ Database already exists"
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo ""
echo "For Docker (Recommended):"
echo "  ./docker.sh start"
echo ""
echo "For Manual Installation:"
echo "  pip install -r StaffFaceRecognition/requirements.txt"
echo "  ./startapp.sh"
echo ""
echo "Access the application at: http://localhost:8000"
echo "Default login: admin / admin123"
