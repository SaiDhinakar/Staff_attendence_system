# Staff Attendance System

A face recognition-based staff attendance system with Django frontend and FastAPI backend.

## Quick Start with Docker (Recommended)

### Prerequisites

- Docker installed on your system
- Camera connected (for face recognition)
- User added to docker group (Linux/Mac)

### Docker Installation & Setup

#### 1. Install Docker
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

#### 2. Verify Docker Installation
```bash
docker --version
docker run hello-world
```

### Running with Docker

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd Staff_attendence_system
   ```

2. **Build and start the application:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Web Interface: http://localhost:8000
   - API Backend: http://localhost:5600
   - Default login: `admin` / `admin123`

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Docker Commands

```bash
# Build and start in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild without cache
docker-compose build --no-cache

# Enter the container shell
docker-compose exec staff-attendance-app bash
```

## Manual Installation (Alternative)

### Prerequisites
- Python 3.8+
- pip
- Camera connected to the system

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r StaffFaceRecognition/requirements.txt
   ```

2. **Start the application:**
   ```bash
   chmod +x startapp.sh
   ./startapp.sh
   ```

3. **Access the application:**
   - Web Interface: http://localhost:8000
   - Default login: `admin` / `admin123`

## Features

- 🎥 **Real-time Face Recognition**: Live camera feed with face detection
- 👥 **Employee Management**: Add/edit employee profiles
- ⏰ **Attendance Tracking**: Automatic check-in/check-out
- 📊 **Reports**: View attendance reports and statistics
- 🔒 **Authentication**: Secure admin panel
- 🐳 **Docker Support**: Easy deployment with Docker

## Architecture

- **Frontend**: Django web application (Port 8000)
- **Backend**: FastAPI service for face recognition (Port 5600)
- **Database**: SQLite for data storage
- **Camera**: OpenCV for video capture

## Configuration

### Environment Variables

The application uses a `.env` file in the `StaffFaceRecognition` directory:

```env
IP=127.0.0.1
```

### Camera Setup

- The application uses the default camera (`/dev/video0`)
- For different cameras, modify the Docker Compose file device mapping
- For USB cameras, ensure proper permissions

## Usage

1. **Add Employees:**
   - Login to admin panel
   - Navigate to Employee section
   - Add employee profiles with photos

2. **Face Recognition:**
   - Employees face the camera
   - System automatically detects and recognizes faces
   - Click "Check In" or "Check Out" buttons

3. **View Reports:**
   - Access attendance reports from the admin panel
   - Export data as needed

## Troubleshooting

### Docker Issues

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs staff-attendance-app

# Restart services
docker-compose restart
```

### Camera Issues

- Ensure camera permissions: `sudo usermod -a -G video $USER`
- Check camera device: `ls /dev/video*`
- Test camera: `ffplay /dev/video0`

### Common Problems

1. **Backend fails to start**: Check camera permissions and dependencies
2. **Face recognition not working**: Ensure proper lighting and camera quality
3. **Database errors**: Check file permissions and disk space

## Development

### Project Structure

```
Staff_attendence_system/
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── startapp.sh
├── requirements-docker.txt
└── StaffFaceRecognition/
    ├── manage.py
    ├── requirements.txt
    ├── backend/
    │   └── Updated.py
    ├── Home/
    ├── Authentication/
    └── Frontend/
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test with Docker
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review logs: `docker-compose logs`
