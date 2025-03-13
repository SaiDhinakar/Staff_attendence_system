# Staff Attendance System with Face Recognition

A robust attendance management system that uses facial recognition to automatically track staff attendance in real-time. Built with Python, Django, FastAPI, and deep learning.

## 🌟 Features

- Real-time face detection and recognition
- Automated attendance marking
- Web-based admin interface
- Employee management (add/delete/update)
- Attendance reports and analytics
- Multi-face detection support
- High accuracy face recognition using deep learning
- Background task processing for face embeddings
- Profile photo management

## 🏗️ System Architecture

### Components:
1. **Web Interface (Django)**
   - Employee management
   - Attendance records
   - Report generation
   - Profile photo management

2. **Face Recognition Backend (FastAPI)**
   - Real-time video processing
   - Face detection using MTCNN
   - Face recognition using ResNet
   - Embedding generation and matching

3. **Database**
   - SQLite for storing employee and attendance records
   - JSON file for storing face embeddings

## 🛠️ Technical Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Django, FastAPI
- **Face Recognition**: 
  - MTCNN (Multi-task Cascaded Convolutional Networks)
  - ResNet (Deep Residual Learning)
  - FaceNet PyTorch
- **Database**: SQLite3
- **Camera Interface**: OpenCV with NVIDIA Jetson support
- **Background Tasks**: Asynchronous processing

## 📝 Process Flow

1. **Employee Registration**
   - Admin adds new employee with details
   - Upload multiple face photos
   - System processes photos in background
   - Generates and stores face embeddings

2. **Face Detection**
   - Camera captures real-time video feed
   - MTCNN detects faces in frames
   - Largest face is selected for recognition
   - Face alignment and preprocessing

3. **Face Recognition**
   - Convert detected face to embeddings
   - Compare with stored employee embeddings
   - Match found -> Mark attendance
   - Rate limiting prevents duplicate entries

4. **Attendance Processing**
   - Automatic attendance marking
   - Timestamp recording
   - Status tracking (IN/OUT)
   - Attendance report generation

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/staff_attendance_system.git
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Database setup:
```bash
python manage.py migrate
```

5. Create admin user:
```bash
python manage.py createsuperuser
```

## 🚀 Running the System

1. Start Django server:
```bash
python manage.py runserver
```

2. Start Face Recognition backend:
```bash
python backend/face_detect.py
```

## 📂 Project Structure

```
Staff_attendance_system/
├── StaffFaceRecognition/
│   ├── backend/
│   │   ├── face_detect.py
│   │   └── staff_embeddings.json
│   ├── Home/
│   │   ├── views.py
│   │   ├── models.py
│   │   └── urls.py
│   └── media/
│       └── employee_photos/
└── requirements.txt
```

## ⚙️ Configuration

Key settings can be modified in:
- `settings.py`: Django configuration
- `face_detect.py`: Recognition parameters
- `staff_embeddings.json`: Embedding storage

## 🔐 Security Features

- Authentication required for admin access
- Secure embedding storage
- Rate limiting for attendance marking
- Transaction-safe database operations

## 📈 Performance Optimization

- Batch processing for embeddings
- Async video processing
- Background task handling
- Efficient embedding storage
- Camera pipeline optimization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.