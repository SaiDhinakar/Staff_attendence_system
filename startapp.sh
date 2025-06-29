#!/bin/bash

# Script to start both the backend Python service and Django web application
# for the Staff Face Recognition System

echo "=== Staff Attendance System Startup ==="
echo "Checking dependencies..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "StaffFaceRecognition/manage.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to Django directory
cd StaffFaceRecognition

# Run Django migrations
echo "Running Django migrations..."
python manage.py makemigrations --noinput 2>/dev/null || echo "No new migrations to make"
python manage.py migrate --noinput || echo "Migration completed"

# Create superuser if needed
echo "Checking for superuser..."
python manage.py shell -c "
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin/admin123')
else:
    print('Superuser already exists')
" 2>/dev/null || echo "Superuser check completed"

# Start the backend Python process
echo "Starting FastAPI backend..."
cd backend
python Updated.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to initialize properly
echo "Waiting for backend to initialize..."
sleep 8

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "✓ Backend started successfully (PID: $BACKEND_PID)"
    echo "✓ Backend logs: logs/backend.log"
    
    # Start the Django application
    echo "Starting Django frontend on http://0.0.0.0:8000..."
    echo "Default login: admin / admin123"
    echo ""
    echo "Press Ctrl+C to stop both services"
    
    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo "Stopping services..."
        kill $BACKEND_PID 2>/dev/null || echo "Backend already stopped"
        echo "All services stopped."
        exit 0
    }
    
    # Set up signal handler
    trap cleanup SIGINT SIGTERM
    
    # Start Django server
    python manage.py runserver 0.0.0.0:8000
    
    # If we get here, Django stopped, so clean up
    cleanup
else
    echo "✗ Backend failed to start. Check logs/backend.log for details."
    cat ../logs/backend.log
    exit 1
fi
