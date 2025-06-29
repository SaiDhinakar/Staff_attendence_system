#!/bin/bash

# Docker management script for Staff Attendance System

set -e

IMAGE_NAME="staff-attendance-system"
CONTAINER_NAME="staff-attendance-app"

show_help() {
    echo "Staff Attendance System - Docker Management"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  start       Start the application"
    echo "  stop        Stop the application"
    echo "  restart     Restart the application"
    echo "  logs        Show application logs"
    echo "  clean       Remove containers and images"
    echo "  shell       Open shell in container"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start      # Start the application"
    echo "  $0 logs       # View logs"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed or not in PATH"
        echo ""
        echo "To install Docker:"
        echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "  sudo sh get-docker.sh"
        echo "  sudo usermod -aG docker \$USER"
        echo "  # Log out and log back in"
        exit 1
    fi
    
    # Test Docker permissions
    if ! docker ps >/dev/null 2>&1; then
        echo "Error: Docker permission denied"
        echo ""
        echo "This usually means you need to:"
        echo "1. Add your user to the docker group:"
        echo "   sudo usermod -aG docker \$USER"
        echo "2. Log out and log back in"
        echo ""
        echo "Or run this script with sudo (not recommended):"
        echo "   sudo $0 $1"
        exit 1
    fi
}

build_image() {
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
    echo "Build completed successfully!"
}

start_app() {
    # Stop existing container if running
    stop_app 2>/dev/null || true
    
    echo "Starting Staff Attendance System..."
    
    # Create necessary directories on host
    mkdir -p logs media/profile_pics
    
    docker run -d \
        --name $CONTAINER_NAME \
        -p 8000:8000 \
        -p 5600:5600 \
        -v "$(pwd)/media:/app/media" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/StaffFaceRecognition/db.sqlite3:/app/StaffFaceRecognition/db.sqlite3" \
        -v "$(pwd)/face_embeddings.json:/app/face_embeddings.json" \
        -v "$(pwd)/StaffFaceRecognition/backend/dataset:/app/StaffFaceRecognition/backend/dataset" \
        --device=/dev/video0:/dev/video0 \
        -e PYTHONUNBUFFERED=1 \
        -e IP=127.0.0.1 \
        --restart unless-stopped \
        $IMAGE_NAME
    
    echo ""
    echo "✓ Application started successfully!"
    echo ""
    echo "Access the application:"
    echo "  Web Interface: http://localhost:8000"
    echo "  API Backend:   http://localhost:5600"
    echo "  Default login: admin / admin123"
    echo ""
    echo "To view logs: $0 logs"
    echo "To stop:      $0 stop"
}

stop_app() {
    echo "Stopping Staff Attendance System..."
    docker stop $CONTAINER_NAME 2>/dev/null || echo "Container not running"
    docker rm $CONTAINER_NAME 2>/dev/null || echo "Container already removed"
    echo "Application stopped."
}

restart_app() {
    echo "Restarting Staff Attendance System..."
    stop_app
    start_app
}

show_logs() {
    echo "Showing application logs (Press Ctrl+C to exit)..."
    docker logs -f $CONTAINER_NAME
}

clean_docker() {
    echo "Cleaning up Docker resources..."
    read -p "This will remove containers and images. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        echo "Cleanup completed."
    else
        echo "Cleanup cancelled."
    fi
}

open_shell() {
    echo "Opening shell in container..."
    docker exec -it $CONTAINER_NAME bash
}

check_status() {
    if docker ps | grep -q $CONTAINER_NAME; then
        echo "✓ Application is running"
        echo "  Container: $CONTAINER_NAME"
        echo "  Web Interface: http://localhost:8000"
        echo "  API Backend: http://localhost:5600"
    else
        echo "✗ Application is not running"
        echo "  Use '$0 start' to start the application"
    fi
}

# Main script logic
check_docker

case "${1:-help}" in
    build)
        build_image
        ;;
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    logs)
        show_logs
        ;;
    status)
        check_status
        ;;
    clean)
        clean_docker
        ;;
    shell)
        open_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
