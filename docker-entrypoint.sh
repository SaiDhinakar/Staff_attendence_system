#!/bin/bash
set -e

echo "=== Staff Attendance System Docker Container ==="
echo "Starting initialization..."

# Change to the correct directory
cd /app/StaffFaceRecognition

# Run Django migrations
echo "Running Django migrations..."
python manage.py makemigrations --noinput || echo "No new migrations to make"
python manage.py migrate --noinput || echo "Migration completed"

# Create superuser if it doesn't exist
echo "Creating superuser if needed..."
python manage.py shell -c "
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin/admin123')
else:
    print('Superuser already exists')
" || echo "Superuser creation skipped"

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput || echo "Static files collection completed"

# Start the FastAPI backend in the background
echo "Starting FastAPI backend on port 5600..."
cd /app/StaffFaceRecognition/backend
python Updated.py > /app/logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to initialize
echo "Waiting for backend to initialize..."
sleep 10

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "✓ Backend started successfully (PID: $BACKEND_PID)"
else
    echo "✗ Backend failed to start. Check /app/logs/backend.log for details."
    exit 1
fi

# Start Django frontend
echo "Starting Django frontend on port 8000..."
cd /app/StaffFaceRecognition
python manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!

# Wait for Django to initialize
sleep 5

if ps -p $DJANGO_PID > /dev/null; then
    echo "✓ Django frontend started successfully (PID: $DJANGO_PID)"
else
    echo "✗ Django frontend failed to start."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "=== All services started successfully ==="
echo "Access the application at: http://localhost:8000"
echo "FastAPI backend available at: http://localhost:5600"
echo "Default login: admin / admin123"
echo ""
echo "Logs:"
echo "  Backend: /app/logs/backend.log"
echo "  Django: stdout"
echo ""

# Function to cleanup processes on exit
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || echo "Backend already stopped"
    kill $DJANGO_PID 2>/dev/null || echo "Django already stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Keep the container running and wait for processes
wait $DJANGO_PID
