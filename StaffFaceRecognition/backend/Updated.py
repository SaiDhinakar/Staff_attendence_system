import cv2
import torch
import json
import os
import time
import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from datetime import datetime, timedelta
from pydantic import BaseModel
from collections import defaultdict
import uvicorn
from typing import Dict
import asyncio
import logging
from fastapi import BackgroundTasks

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_detection_data = None
latest_detection_times = {}
latest_frame = None
latest_detected_ids = []
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self.last_check = {}
    
    def can_process(self, identity):
        now = time.time()
        if identity not in self.last_check:
            self.last_check[identity] = now
            return True
        
        if now - self.last_check[identity] >= self.interval:
            self.last_check[identity] = now
            return True
        return False

attendance_limiter = RateLimiter(60)  # 60 second interval

class FaceDetect:
    def __init__(self, db_file="face_embeddings.json"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')

        # Initialize models
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load known embeddings
        self.db_file = db_file
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return json.load(f)
        return {}
    def reload_embeddings(self):
        """Reload face embeddings from the JSON file."""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, "r") as f:
                    self.embeddings = json.load(f)
                logger.info("Face embeddings reloaded successfully")
                return True
            else:
                logger.error(f"Embedding file {self.db_file} not found")
                return False
        except Exception as e:
            logger.error(f"Error reloading embeddings: {e}")
            return False
        
    async def process_frame(self, frame):
        global latest_detection_data, latest_detection_times, latest_frame, latest_detected_ids
        
        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = self.mtcnn.detect(img)
            
            # Initialize with Unknown detection
            detection_data = {
                "identity": "Unknown", 
                "confidence": None,
                "time": datetime.now().strftime("%H:%M:%S")
            }

            if boxes is not None and len(boxes) > 0:
                # Find largest face in frame
                best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = map(int, best_box)
                
                # Crop and process the face
                face_img = img.crop((x1, y1, x2, y2))
                
                # Get face tensor but DON'T use unsqueeze - this causes the dimension error
                face_tensor = self.mtcnn(face_img)
                
                if face_tensor is not None:
                    # Important: Log the tensor shape for debugging
                    logger.debug(f"Face tensor shape: {face_tensor.shape}")
                    
                    # Recognize the face - face_tensor is already properly batched
                    identity, dist = self.recognize_face(face_tensor)
                    confidence = 1 - dist if identity != "Unknown" else None
                    
                    # Update detection data with results
                    detection_data = {
                        "identity": identity, 
                        "confidence": confidence,
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                    
                    # Draw bounding box on frame for visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, identity, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update global variables
            latest_detection_data = detection_data
            latest_frame = frame
            latest_detected_ids = detection_data["identity"]
            
            # Update detection times properly
            if detection_data["identity"] != "Unknown":
                # Store as dict with identity as key and detection data as value
                latest_detection_times = {detection_data["identity"]: detection_data}
            else:
                latest_detection_times = {}
                
            return detection_data
                
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return {"identity": "Unknown", "confidence": None, "time": datetime.now().strftime("%H:%M:%S")}

    def recognize_face(self, face_tensor):
        if face_tensor is None:
            return "Unknown", None
            
        try:
            # Make sure tensor has correct shape
            if len(face_tensor.shape) == 3:  # Single image, shape [C, H, W]
                face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
            elif len(face_tensor.shape) > 4:  # Too many dimensions
                logger.warning(f"Face tensor has unexpected shape: {face_tensor.shape}")
                face_tensor = face_tensor.squeeze()  # Remove extra dimensions
                if len(face_tensor.shape) == 3:
                    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
                    
            # Get embedding
            embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()
            
            min_dist = float('inf')
            identity = "Unknown"

            for name, encodings in self.embeddings.items():
                for db_enc in encodings:
                    dist = torch.nn.functional.pairwise_distance(
                        embedding, torch.tensor(db_enc).unsqueeze(0)
                    ).item()
                    if dist < min_dist:
                        min_dist = dist
                        identity = name

            return (identity, min_dist) if min_dist <= 0.6 else ("Unknown", min_dist)
            
        except Exception as e:
            logger.error(f"Error in recognize_face: {e}")
            return "Unknown", 1.0

face_detector = FaceDetect()

latest_frame = None
latest_detected_ids = []
latest_detection_times = {}
last_attendance_time: Dict[str, datetime] = {}
MIN_ATTENDANCE_INTERVAL = timedelta(minutes=1) 

checked_status = None

employee_check_status  = {}

async def process_frame_wrapper(frame):
    try:
        await face_detector.process_frame(frame)
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")

async def video_capture():
    global latest_frame, latest_detected_ids, latest_detection_times

    pipeline = (
        "nvarguscamerasrc sensor-id=0 sensor-mode=3 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not open camera. Exiting.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame, attempting reset.")
            # reset_camera()
            await asyncio.sleep(2)
            # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            cap = cv2.VideoCapture(0)
            continue

        await process_frame_wrapper(frame)
        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


def reset_camera():
    os.system("sudo systemctl restart nvargus-daemon")
    time.sleep(2)
    logger.info("Camera reset completed.")

@app.get('/video_stream')
async def video_stream():
    return StreamingResponse(generate_video_stream(), media_type='text/event-stream')

def generate_video_stream():
    last_heartbeat = time.time()
    
    while True:
        current_time = time.time()
        
        if latest_frame is not None:
            try:
                # Resize for efficiency
                frame_small = cv2.resize(latest_frame, (640, 480))
                
                # Compress image
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                _, buffer = cv2.imencode('.jpg', frame_small, encode_param)
                encoded_frame = base64.b64encode(buffer.tobytes()).decode('utf-8')
                
                # Get detection ID safely
                detection = "Unknown"
                if latest_detected_ids is not None:
                    if isinstance(latest_detected_ids, list):
                        detection = latest_detected_ids[0] if latest_detected_ids else "Unknown"
                    else:
                        detection = latest_detected_ids
                
                # Create frame data
                frame_data = {
                    "type": "frame_update",
                    "data": {
                        "image": encoded_frame,
                        "detection": detection,
                        "timestamp": current_time
                    }
                }
                
                # Send frame
                yield f"data: {json.dumps(frame_data)}\n\n"
                last_heartbeat = current_time
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in video stream: {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                
            # Limit frame rate
            time.sleep(0.03)  # ~30 fps
            
        else:
            # No frame available, send heartbeat
            if current_time - last_heartbeat > 2.0:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                last_heartbeat = current_time
                
            # Wait longer when no frames
            time.sleep(0.5)

def save_attendance(emp_id, detection_time, check_type):
    try:
        conn = sqlite3.connect(r"../db.sqlite3")
        cursor = conn.cursor()

        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Check if the employee already has an attendance record for today
        cursor.execute("SELECT emp_id, time_in_list, time_out_list FROM Home_attendance WHERE emp_id = ? AND date = ?",
                    (emp_id, current_date))
        record = cursor.fetchone()

        if record:
            # Record exists, update time_in or time_out by appending new time
            attendance_id, time_in, time_out = record

            if check_type == "check_in":
                if time_in:
                    updated_time_in = f"{time_in},{detection_time}"
                else:
                    updated_time_in = detection_time
                cursor.execute("UPDATE Home_attendance SET time_in_list = ? WHERE id = ?", (updated_time_in, attendance_id))
            elif check_type == "check_out":
                if time_out:
                    updated_time_out = f"{time_out},{detection_time}"
                else:
                    updated_time_out = detection_time
                cursor.execute("UPDATE Home_attendance SET time_out_list = ? WHERE id = ?", (updated_time_out, attendance_id))

        else:
            # Insert a new record with the first detection time
            if check_type == "check_in":
                cursor.execute(
                    "INSERT INTO Home_attendance (date, emp_id, time_in_list, time_out_list) VALUES (?, ?, ?, ?)",
                    (current_date, emp_id, detection_time, ''))
            elif check_type == "check_out":
                cursor.execute(
                    "INSERT INTO Home_attendance (date, emp_id, time_in_list, time_out_list) VALUES (?, ?, ?, ?)",
                    (current_date, emp_id, '', detection_time))

        conn.commit()
    except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        if conn:
            conn.close()

class EmbeddingRequest(BaseModel):
    db_path: str
    output_file: str = "face_embeddings.json"

def store_embeddings(db_path, output_file):
    if not os.path.exists(db_path):
        return {"error": "Dataset path does not exist"}

    existing_embeddings = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_embeddings = json.load(f)

    embeddings = defaultdict(list, existing_embeddings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    for identity in os.listdir(db_path):
        # Skip the profile_pics directory
        if identity == 'profile_pics':
            continue
            
        identity_path = os.path.join(db_path, identity)
        if os.path.isdir(identity_path):
            for image_name in os.listdir(identity_path):
                image_path = os.path.join(identity_path, image_name)
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_cropped = mtcnn(img)
                    if img_cropped is not None:
                        img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).detach().cpu().numpy().tolist()
                        embeddings[identity].append(img_embedding)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue

    with open(output_file, "w") as f:
        json.dump(dict(embeddings), f, indent=4)

    return {"message": "Embeddings stored successfully", "output_file": output_file}

@app.post("/store_embeddings/")
def api_store_embeddings(request: EmbeddingRequest):
    return store_embeddings(request.db_path, request.output_file)

@app.get("/load_embeddings/")
def load_embeddings(input_file: str = "face_embeddings.json"):
    if not os.path.exists(input_file):
        return {"error": "Embeddings file not found"}

    with open(input_file, "r") as f:
        embeddings = json.load(f)
    return embeddings

@app.get('/check-in')
async def check_in():
    global latest_detection_times

    print(f"latest_detection_times: {latest_detection_times}")  # Debugging
    if not latest_detection_times:
        raise HTTPException(status_code=400, detail="No face detected")

    # ✅ Fix: Properly get the latest detected employee ID
    emp_id, detection_data = next(iter(latest_detection_times.items()))
    employee_check_status[str(emp_id)]=True

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db.sqlite3'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (emp_id,))
    employee = cursor.fetchone()
    conn.close()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    name, department = employee

    print(f"Detected emp_id: {emp_id}, Confidence: {detection_data['confidence']}")  # Debugging

    if detection_data["confidence"] < 0.4:
        raise HTTPException(status_code=400, detail="Face recognition confidence too low")

    save_attendance(emp_id, detection_data["time"], "check_in")

    return {
        "status": "success",
        "message": "Checked in successfully",
        "employee": {
            "name":name,
            "id": emp_id,
            "department":department,
            "confidence": f"{detection_data['confidence']:.2%}",
            "time": detection_data["time"]
        }
    }


@app.get('/check-out')
async def check_out():
    global latest_detection_times

    print(f"latest_detection_times: {latest_detection_times}")  # Debugging

    if not latest_detection_times:
        raise HTTPException(status_code=400, detail="No face detected")

    # ✅ Fix: Properly get the latest detected employee ID
    emp_id, detection_data = next(iter(latest_detection_times.items()))
    employee_check_status[str(emp_id)]=False

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db.sqlite3'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (emp_id,))
    employee = cursor.fetchone()
    conn.close()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    name, department = employee

    print(f"Detected emp_id: {emp_id}, Confidence: {detection_data['confidence']}")  # Debugging

    if detection_data["confidence"] < 0.4:  # Higher threshold for check-out
        raise HTTPException(status_code=400, detail="Face recognition confidence too low")

    save_attendance(emp_id, detection_data["time"], "check_out")

    return {
        "status": "success",
        "message": "Checked out successfully",
        "employee": {
           "name":name,
            "id": emp_id,
            "department":department,
            "confidence": f"{detection_data['confidence']:.2%}",
            "time": detection_data["time"]
        }
    }
@app.get('/video_stream')
async def video_stream():
    """Stream video with face detection results to the client."""
    return StreamingResponse(generate_video_stream(), media_type='text/event-stream')

async def process_automatic_attendance(identity: str, detection_data: dict):
    """Process automatic attendance based on face detection"""
    try:
        # Validate attendance first
        if not validate_attendance(identity, "check_in"):
            logger.warning(f"Attendance validation failed for {identity}")
            return
            
        current_time = datetime.now()
    
        # Check if enough time has passed since last attendance
        if identity in last_attendance_time:
            time_diff = current_time - last_attendance_time[identity]
            if time_diff < MIN_ATTENDANCE_INTERVAL:
                return
    
        try:
            # Get employee details
            db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db.sqlite3'))
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (identity,))
            employee = cursor.fetchone()
            
            if not employee:
                return
                
            name, department = employee
            
            # Determine check type based on time
            current_hour = current_time.hour
            check_type = "check_in" if current_hour < 12 else "check_out"
            
            # Save attendance
            detection_time = current_time.strftime("%H:%M:%S")
            save_attendance(identity, detection_time, check_type)
            
            # Update last attendance time
            last_attendance_time[identity] = current_time
            
            # Prepare attendance record for frontend
            attendance_record = {
                "emp_id": identity,
                "emp_name": name,
                "department": department,
                "time_in_list": detection_time if check_type == "check_in" else None,
                "time_out_list": detection_time if check_type == "check_out" else None,
                "working_hours": calculate_working_hours(identity)
            }
            
            # Send SSE event with attendance update
            data = {
                "type": "attendance_update",
                "data": attendance_record
            }
            yield f"data: {json.dumps(data)}\n\n"
            
        except Exception as e:
            print(f"Error in automatic attendance: {e}")
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error in automatic attendance: {e}")
        # Send error event to frontend
        error_data = {
            "type": "attendance_error",
            "data": {"message": str(e)}
        }
        yield f"data: {json.dumps(error_data)}\n\n"

def calculate_working_hours(emp_id: str) -> str:
    """Calculate working hours for an employee"""
    try:
        conn = sqlite3.connect(r"../db.sqlite3")
        cursor = conn.cursor()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT time_in_list, time_out_list 
            FROM Home_attendance 
            WHERE emp_id = ? AND date = ?
        """, (emp_id, current_date))
        
        record = cursor.fetchone()
        if not record:
            return "--:--"
            
        time_in, time_out = record
        
        if not time_in or not time_out:
            return "--:--"
            
        # Get latest check-in and check-out times
        latest_in = time_in.split(',')[-1]
        latest_out = time_out.split(',')[-1]
        
        # Convert to datetime
        time_in_dt = datetime.strptime(latest_in, "%H:%M:%S")
        time_out_dt = datetime.strptime(latest_out, "%H:%M:%S")
        
        # Calculate duration
        duration = time_out_dt - time_in_dt
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        
        return f"{hours:02d}:{minutes:02d}"
        
    except Exception as e:
        print(f"Error calculating working hours: {e}")
        return "--:--"
    finally:
        if conn:
            conn.close()

class AttendanceError(Exception):
    """Custom exception for attendance-related errors"""
    pass


def validate_attendance(emp_id: str, check_type: str) -> bool:
    """Validate attendance marking conditions"""
    try:
        conn = sqlite3.connect(r"../db.sqlite3")
        cursor = conn.cursor()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Check if employee exists
        cursor.execute("SELECT emp_id FROM Home_employee WHERE emp_id = ?", (emp_id,))
        if not cursor.fetchone():
            raise AttendanceError("Employee not found")
            
        # Check for duplicate check-in/out in short time
        cursor.execute("""
            SELECT time_in_list, time_out_list 
            FROM Home_attendance 
            WHERE emp_id = ? AND date = ?
        """, (emp_id, current_date))
        
        record = cursor.fetchone()
        if record:
            time_list = record[0] if check_type == "check_in" else record[1]
            if time_list:
                last_time = datetime.strptime(time_list.split(',')[-1], "%H:%M:%S")
                time_diff = datetime.now() - last_time
                if time_diff < MIN_ATTENDANCE_INTERVAL:
                    raise AttendanceError("Too soon for another attendance mark")
                    
        return True
        
    except AttendanceError as e:
        print(f"Attendance validation failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_db_connection(max_retries=3, retry_delay=1):
    """Get database connection with retry mechanism"""
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(r"../db.sqlite3")
            return conn
        except sqlite3.Error as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to connect to database after {max_retries} attempts")
                raise
            logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
            time.sleep(retry_delay)


@app.post("/reload-embeddings")
async def reload_embeddings(background_tasks: BackgroundTasks):
    """Endpoint to reload face embeddings"""
    try:
        success = face_detector.reload_embeddings()
        if success:
            return {"status": "success", "message": "Face embeddings reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload embeddings")
    except Exception as e:
        logger.error(f"Error in reload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    threading.Thread(target=lambda: asyncio.run(video_capture()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=5600)
