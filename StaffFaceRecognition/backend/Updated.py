
import cv2
import queue
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
from datetime import datetime
from pydantic import BaseModel
from collections import defaultdict, deque
import uvicorn
import asyncio
import logging
from fastapi import BackgroundTasks
from PIL import Image, ImageEnhance  


app = FastAPI()

# Queue for recent activity events (for SSE)
recent_activity_queue = queue.Queue()

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

# Face recognition history buffer - stores recognized faces for 5 minutes
face_recognition_history = {}
RECOGNITION_HISTORY_TIMEOUT = 300  # Increased to 5 minutes

# manage the face recognition history
def update_recognition_history(identity, confidence, face_image=None):
    """
    Update the face recognition history buffer.
    Maintains recognized identities for 2 minutes.
    """
    global face_recognition_history
    current_time = datetime.now()
    
    # Remove expired entries (older than 2 minutes)
    expired_ids = []
    for face_id in face_recognition_history:
        if (current_time - face_recognition_history[face_id]['timestamp']).total_seconds() > RECOGNITION_HISTORY_TIMEOUT:
            expired_ids.append(face_id)
    
    for face_id in expired_ids:
        del face_recognition_history[face_id]
    
    # Add or update the current detection if it passes threshold
    if identity != "Unknown" and confidence is not None and confidence >= 0.3:  # Lowered threshold for better detection
        if identity not in face_recognition_history:
            face_recognition_history[identity] = {
                'identity': identity,
                'confidence': confidence,
                'timestamp': current_time,
                'face_image': face_image,
                'time': current_time.strftime("%H:%M:%S")
            }
        else:
            # Update with higher confidence if found, or refresh timestamp
            if confidence > face_recognition_history[identity]['confidence']:
                face_recognition_history[identity]['confidence'] = confidence
                face_recognition_history[identity]['face_image'] = face_image
            face_recognition_history[identity]['timestamp'] = current_time  # Always refresh timestamp
            face_recognition_history[identity]['time'] = current_time.strftime("%H:%M:%S")
    
    return face_recognition_history

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

        # Initialize models with improved parameters
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=50,
            thresholds=[0.6, 0.7, 0.9],
            factor=0.709,
            post_process=True,
            keep_all=True, 
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load known embeddings
        self.db_file = db_file
        self.embeddings = self.load_embeddings()
        
        # Add batch processing capabilities
        self.batch_size = 32  # Adjust based on your GPU memory
        
        # Add adaptive threshold capabilities
        self.initial_threshold = 0.7  # Start with more permissive threshold
        self.current_threshold = self.initial_threshold
        self.confidence_history = []

    def load_embeddings(self):
        """Load face embeddings from a JSON file."""
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
            
    # Add this new method for batch processing
    async def process_embeddings(self, image_paths):
        """Process multiple images in batches"""
        embeddings = []

        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []

            # Load and preprocess images
            for path in batch_paths:
                try:
                    img = Image.open(path)
                    face = self.mtcnn(img)
                    if face is not None:
                        batch_images.append(face)
                except Exception as e:
                    logger.error(f"Error processing image {path}: {e}")
                    continue

            if batch_images:
                # Stack tensors for batch processing
                batch_tensor = torch.stack(batch_images)

                # Get embeddings for batch
                with torch.no_grad():  # Disable gradient calculation
                    batch_embeddings = self.resnet(batch_tensor.to(self.device))

                embeddings.extend(batch_embeddings.cpu().numpy())

        return embeddings

    async def process_frame(self, frame):
        global latest_detection_data, latest_detection_times, latest_frame, latest_detected_ids
        
        try:
            # Convert frame to RGB for better processing
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Enhance image quality
            img = ImageEnhance.Contrast(img).enhance(1.1)
            img = ImageEnhance.Brightness(img).enhance(1.05)
            
            # Detect faces
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
                
                # Add a margin around the face for better recognition
                margin = int((x2 - x1) * 0.2)  # 20% margin
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(img.width, x2 + margin)
                y2 = min(img.height, y2 + margin)
                
                # Crop and process the face
                face_img = img.crop((x1, y1, x2, y2))
                
                # Run face through MTCNN again for better alignment
                face_tensor = self.mtcnn(face_img)
                
                if face_tensor is not None:
                    # Multiple recognition attempts with slightly different parameters
                    identity1, dist1 = self.recognize_face(face_tensor)
                    
                    # Try a second time with increased contrast
                    enhanced_face = ImageEnhance.Contrast(face_img).enhance(1.3)
                    face_tensor2 = self.mtcnn(enhanced_face)
                    identity2, dist2 = self.recognize_face(face_tensor2) if face_tensor2 is not None else ("Unknown", 1.0)
                    
                    # Choose the better result
                    if dist1 <= dist2:
                        identity, confidence = identity1, 1 - dist1
                    else:
                        identity, confidence = identity2, 1 - dist2
                    
                    if identity != "Unknown":
                        # Update detection data with results
                        detection_data = {
                            "identity": identity, 
                            "confidence": confidence,
                            "time": datetime.now().strftime("%H:%M:%S")
                        }
                        
                        # Update recognition history
                        update_recognition_history(identity, confidence, face_img)
                        
                        # Draw bounding box on frame for visualization (darker color, thicker line, no text)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 0), 4)

                        # Generate overlay data for recognized face
                        if user_preference["show_overlay"]:
                            overlay_data = generate_overlay_data(identity)
                            if overlay_data:
                                # Store overlay data to be sent with next frame update
                                latest_overlay_data = overlay_data

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
                
            # Update recognition history
            update_recognition_history(
                detection_data["identity"], 
                detection_data["confidence"], 
                face_img if detection_data["identity"] != "Unknown" else None
            )
            
            return detection_data
                
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return {"identity": "Unknown", "confidence": None, "time": datetime.now().strftime("%H:%M:%S")}

    def recognize_face(self, face_tensor):
        """Compares face embeddings to known faces in the database."""
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
                    # Ensure both embeddings are properly shaped for distance calculation
                    db_embedding = torch.tensor(db_enc).flatten().unsqueeze(0)  # [1, 512]
                    current_embedding = embedding.flatten().unsqueeze(0)       # [1, 512]
                    
                    # Calculate distance - now both tensors have compatible shapes
                    dist = torch.nn.functional.pairwise_distance(current_embedding, db_embedding).item()
                    
                    if dist < min_dist:
                        min_dist = dist
                        identity = name

            threshold = 0.6
            return (identity, min_dist) if min_dist <= threshold else ("Unknown", min_dist)
            
        except Exception as e:
            logger.error(f"Error in recognize_face: {e}")
            return "Unknown", 1.0

face_detector = FaceDetect()

latest_frame = None
latest_detected_ids = []
latest_detection_times = {}

checked_status = None

# Add user preference tracking
user_preference = {
    "action": "check_in",  # Default to check-in
    "show_overlay": True   # Default to showing overlay
}

# Add function to generate overlay data
def generate_overlay_data(emp_id):
    """Generate data for UI overlay based on employee ID or using face recognition history"""
    try:
        global face_recognition_history
        
        # If we should display a list of detections
        if isinstance(emp_id, str) and (emp_id == "Unknown" or not emp_id):
            # Check if we have face recognition history to display
            if face_recognition_history:
                # Return all recognized faces in history
                employees_data = []
                conn = get_db_connection()
                
                for identity, data in face_recognition_history.items():
                    try:
                        cursor = conn.cursor()
                        cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (identity,))
                        employee = cursor.fetchone()
                        
                        if employee:
                            name, department = employee
                            employees_data.append({
                                "name": name,
                                "id": identity,
                                "department": department,
                                "confidence": f"{data['confidence']:.2%}",
                                "time": data['time']
                            })
                    except Exception as e:
                        logger.error(f"Error getting employee data for {identity}: {e}")
                
                conn.close()
                
                # Get current date
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                if employees_data:
                    return {
                        "employees": employees_data,
                        "date": current_date,
                        "action": user_preference["action"],
                        "multi_detection": True
                    }
                return None
            else:
                return None
            
        # For a specific employee ID    
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (emp_id,))
        employee = cursor.fetchone()
        conn.close()
        
        if not employee:
            return None
            
        name, department = employee
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "name": name,
            "id": emp_id,
            "department": department,
            "time": current_time,
            "date": current_date,
            "action": user_preference["action"],
            "multi_detection": False
        }
    except Exception as e:
        logger.error(f"Error generating overlay data: {e}")
        return None

async def process_frame_wrapper(frame):
    try:
        await face_detector.process_frame(frame)
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")

def reset_camera():
    """Force reset the Jetson camera pipeline to fix stream issues."""
    os.system("sudo systemctl restart nvargus-daemon")
    time.sleep(2)  # Give time for the daemon to restart
    logger.info("Camera reset completed.")

def video_capture():
    """Continuously capture and process video frames."""
    global latest_frame, latest_detected_ids, latest_detection_times

    # Create event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # pipeline = (
    #     "nvarguscamerasrc sensor-id=0 sensor-mode=3 ! "
    #     "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    #     "nvvidconv ! video/x-raw, format=BGRx ! "
    #     "videoconvert ! video/x-raw, format=BGR ! "
    #     "appsink drop=1"
    # )

    # Use either GSTREAMER or fallback to regular webcam
    try:
        # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except:
        logger.warning("Gstreamer pipeline failed, falling back to regular webcam")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not open camera. Exiting.")
        return

    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame, attempting reset.")
                    reset_camera()
                    cap.release()
                    time.sleep(2)
                    # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    cap = cv2.VideoCapture(0)
                    continue

                # Use the event loop to run async functions
                loop.run_until_complete(process_frame_wrapper(frame))

            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping video capture...")
    except Exception as e:
        logger.error(f"Fatal error during capture: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
            cv2.destroyAllWindows()
        loop.close()
        logger.info("Camera resources released.")

@app.get('/video_stream')
async def video_stream():
    return StreamingResponse(generate_video_stream(), media_type='text/event-stream')

def generate_video_stream():
    last_heartbeat = time.time()
    frame_counter = 0
    frame_buffer = None
    last_frame_time = time.time()
    
    # Create thread-local storage for frame processing
    # This helps avoid GIL contention and memory copying
    thread_local = threading.local()
    
    while True:
        current_time = time.time()
        frame_counter += 1
        
        if latest_frame is not None:
            try:
                # Process every frame but alternate between sending and skipping
                # This maintains fluidity while reducing bandwidth
                if frame_counter % 2 == 0:
                    # Calculate FPS for logging/debugging
                    fps = 1.0 / (current_time - last_frame_time) if current_time != last_frame_time else 30.0
                    last_frame_time = current_time
                    
                    # Use a copy to avoid race conditions with the main thread
                    # Only create a new copy if we're actually going to use it
                    if not hasattr(thread_local, 'working_frame'):
                        thread_local.working_frame = latest_frame.copy()
                    else:
                        np.copyto(thread_local.working_frame, latest_frame)
                    
                    # Resize for efficiency - maintain aspect ratio
                    # More efficient resize with fixed dimensions to avoid recalculations
                    if not hasattr(thread_local, 'resize_dims'):
                        height, width = thread_local.working_frame.shape[:2]
                        new_width = 480  # Reduced from 640 for better performance
                        new_height = int(height * (new_width / width))
                        thread_local.resize_dims = (new_width, new_height)
                    
                    frame_small = cv2.resize(thread_local.working_frame, thread_local.resize_dims, 
                                             interpolation=cv2.INTER_NEAREST)  # Faster interpolation
                    
                    # Compress image more efficiently - trade quality for performance
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Reduced quality from 80 to 70
                    
                    # Re-use buffer if possible
                    if frame_buffer is None:
                        _, buffer = cv2.imencode('.jpg', frame_small, encode_param)
                        frame_buffer = buffer
                    else:
                        # Try to reuse buffer for less memory allocation
                        try:
                            success = cv2.imencode('.jpg', frame_small, encode_param, frame_buffer)
                            if not success:
                                _, frame_buffer = cv2.imencode('.jpg', frame_small, encode_param)
                        except:
                            _, frame_buffer = cv2.imencode('.jpg', frame_small, encode_param)
                    
                    # Encode frame only once
                    encoded_frame = base64.b64encode(frame_buffer.tobytes()).decode('utf-8')
                    
                    # Get detection ID efficiently
                    detection = "Unknown"
                    if latest_detected_ids is not None:
                        if isinstance(latest_detected_ids, list):
                            detection = latest_detected_ids[0] if latest_detected_ids else "Unknown"
                        else:
                            detection = latest_detected_ids
                    
                    # Create frame data - simplified for better performance
                    frame_data = {
                        "type": "frame_update",
                        "data": {
                            "image": encoded_frame,
                            "detection": detection,
                            "timestamp": int(current_time * 1000)  # Use integer milliseconds
                        }
                    }
                    
                    # Send frame
                    yield f"data: {json.dumps(frame_data)}\n\n"
                    last_heartbeat = current_time
                
                # No delay between frames - the natural processing time provides rate limiting
                # This allows the system to run as fast as it can handle
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in video stream: {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                time.sleep(0.05)  # Reduced delay if there's an error
                
        else:
            # No frame available, send heartbeat
            if current_time - last_heartbeat > 2.0:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                last_heartbeat = current_time
                
            # Wait shorter time when no frames to check more frequently
            time.sleep(0.1)  # Reduced from 0.2

def save_attendance(emp_id, detection_time, check_type):
    try:
        print(f"DEBUG - save_attendance called with: emp_id={emp_id}, detection_time={detection_time}, check_type={check_type}")
        
        # Use the same get_db_connection function for consistency
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        print(f"DEBUG - Current date: {current_date}")

        # Check if the employee already has an attendance record for today
        cursor.execute("SELECT id, time_in_list, time_out_list FROM Home_attendance WHERE emp_id = ? AND date = ?",
                    (emp_id, current_date))
        record = cursor.fetchone()
        print(f"DEBUG - Existing record: {record}")

        if record:
            # Record exists, update time_in or time_out by appending new time
            attendance_id, time_in, time_out = record
            
            print(f"DEBUG - Updating attendance: ID={attendance_id}, check_type={check_type}")
            print(f"DEBUG - Before update: time_in={time_in}, time_out={time_out}")

            if check_type == "check_in":
                if time_in:
                    updated_time_in = f"{time_in},{detection_time}"
                else:
                    updated_time_in = detection_time
                cursor.execute("UPDATE Home_attendance SET time_in_list = ? WHERE id = ?", 
                              (updated_time_in, attendance_id))
                print(f"DEBUG - Updated time_in to: {updated_time_in}")
            elif check_type == "check_out":
                if time_out:
                    updated_time_out = f"{time_out},{detection_time}"
                else:
                    updated_time_out = detection_time
                cursor.execute("UPDATE Home_attendance SET time_out_list = ? WHERE id = ?", 
                              (updated_time_out, attendance_id))
                print(f"DEBUG - Updated time_out to: {updated_time_out}")

        else:
            # Insert a new record with the first detection time
            print(f"DEBUG - Creating new attendance record")
            if check_type == "check_in":
                cursor.execute(
                    "INSERT INTO Home_attendance (date, emp_id, time_in_list, time_out_list) VALUES (?, ?, ?, ?)",
                    (current_date, emp_id, detection_time, ''))
                print(f"DEBUG - Created new check-in record for {emp_id}")
            elif check_type == "check_out":
                cursor.execute(
                    "INSERT INTO Home_attendance (date, emp_id, time_in_list, time_out_list) VALUES (?, ?, ?, ?)",
                    (current_date, emp_id, '', detection_time))
                print(f"DEBUG - Created new check-out record for {emp_id}")

        conn.commit()
        print(f"DEBUG - Database committed successfully")

        # Broadcast recent activity to SSE clients
        activity = {
            "emp_id": emp_id,
            "detection_time": detection_time,
            "check_type": check_type,
            "timestamp": datetime.now().isoformat()
        }
        recent_activity_queue.put(activity)
        return True
    except sqlite3.Error as e:
        print(f"DEBUG - Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    except Exception as e:
        print(f"DEBUG - General error in save_attendance: {e}")
        raise
    finally:
        if conn:
            conn.close()
            print(f"DEBUG - Database connection closed")

class EmbeddingRequest(BaseModel):
    db_path: str
    output_file: str = "face_embeddings.json"

def store_embeddings(db_path, output_file):
    if not os.path.exists(db_path):
        return {"error": "Dataset path does not exist"}

    existing_embeddings = {}
    if (os.path.exists(output_file)):
        with open(output_file, 'r') as f:
            existing_embeddings = json.load(f)

    embeddings = defaultdict(list, existing_embeddings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use higher quality settings for
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=50, thresholds=[0.6, 0.7, 0.9], 
                  factor=0.709, post_process=True, keep_all=False, device=device)
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
                    # Apply image enhancement
                    img = ImageEnhance.Contrast(img).enhance(1.2)
                    img = ImageEnhance.Brightness(img).enhance(1.1)
                    
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
    global face_recognition_history, latest_detection_times

    print(f"DEBUG - Check-in called")
    print(f"DEBUG - face_recognition_history: {len(face_recognition_history) if face_recognition_history else 0} items")
    print(f"DEBUG - latest_detection_times: {len(latest_detection_times) if latest_detection_times else 0} items")
    print(f"DEBUG - latest_detected_ids: {latest_detected_ids}")

    # Check if we have recent face data
    if not face_recognition_history and not latest_detection_times:
        print("DEBUG - No face recognition history or latest detection times")
        # If both are empty, check if there's any recent detection at all
        if latest_detected_ids and latest_detected_ids != "Unknown":
            print(f"DEBUG - Using latest_detected_ids: {latest_detected_ids}")
            # Create a basic detection record to use
            detection_data = {
                "identity": latest_detected_ids,
                "confidence": 0.5,  # Default confidence
                "time": datetime.now().strftime("%H:%M:%S")
            }
            
            # Use direct processing instead of process_single_checkin
            try:
                # Get employee details
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (latest_detected_ids,))
                employee = cursor.fetchone()
                
                print(f"DEBUG - Employee query result: {employee}")
                
                if not employee:
                    conn.close()
                    raise HTTPException(status_code=404, detail="Employee not found")

                name, department = employee
                conn.close()
                
                print(f"DEBUG - About to save attendance for {latest_detected_ids}")
                # Save attendance
                save_attendance(latest_detected_ids, detection_data["time"], "check_in")
                print(f"DEBUG - Attendance saved successfully")

                return {
                    "status": "success",
                    "message": "Checked in successfully",
                    "action_type": "check_in",
                    "employee": {
                        "name": name,
                        "id": latest_detected_ids,
                        "department": department,
                        "confidence": f"{detection_data['confidence']:.2%}",
                        "time": detection_data["time"]
                    }
                }
            except Exception as e:
                print(f"DEBUG - Exception in check-in: {str(e)}")
                logger.error(f"Check-in error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Check-in failed: {str(e)}")
        else:
            print("DEBUG - No latest_detected_ids available")
            raise HTTPException(status_code=400, detail="No faces detected. Please ensure your face is visible to the camera.")

    # If we have face recognition history, use it
    if face_recognition_history:
        print(f"DEBUG - Processing face_recognition_history with {len(face_recognition_history)} items")
        # Process all faces in the recognition history
        results = []
        current_time = datetime.now()
        
        for identity, data in face_recognition_history.items():
            print(f"DEBUG - Processing identity: {identity} with confidence: {data.get('confidence', 'N/A')}")
            try:
                # Get employee details from DB
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (identity,))
                employee = cursor.fetchone()
                
                if not employee:
                    print(f"DEBUG - Employee {identity} not found in database")
                    conn.close()
                    continue  # Skip if employee not found
                    
                name, department = employee
                conn.close()
                
                print(f"DEBUG - Saving attendance for {identity}")
                # Save attendance
                save_attendance(identity, data['time'], "check_in")
                
                # Add to results
                results.append({
                    "name": name,
                    "id": identity,
                    "department": department,
                    "confidence": f"{data['confidence']:.2%}",
                    "time": data['time']
                })
                
            except Exception as e:
                print(f"DEBUG - Error processing check-in for {identity}: {e}")
                logger.error(f"Error processing check-in for {identity}: {e}")
                # Continue processing other employees even if one fails
        
        # Clear history after processing
        face_recognition_history.clear()
        
        if results:
            print(f"DEBUG - Returning {len(results)} successful check-ins")
            # Return multiple results if available
            return {
                "status": "success",
                "message": f"Checked in {len(results)} employee(s) successfully",
                "action_type": "check_in",
                "employees": results,
                # Include the first employee separately for backward compatibility
                "employee": results[0] if results else None
            }
    
    # Fallback to latest detection if available
    if latest_detection_times:
        print(f"DEBUG - Using latest_detection_times: {latest_detection_times}")
        emp_id, detection_data = next(iter(latest_detection_times.items()))
        
        # Use direct processing instead of process_single_checkin
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (emp_id,))
            employee = cursor.fetchone()

            if not employee:
                conn.close()
                raise HTTPException(status_code=404, detail="Employee not found")

            name, department = employee
            conn.close()

            print(f"DEBUG - Saving attendance for {emp_id} from latest_detection_times")
            # Save attendance
            save_attendance(emp_id, detection_data["time"], "check_in")

            return {
                "status": "success",
                "message": "Checked in successfully",
                "action_type": "check_in",
                "employee": {
                    "name": name,
                    "id": emp_id,
                    "department": department,
                    "confidence": f"{detection_data['confidence']:.2%}",
                    "time": detection_data["time"]
                }
            }
        except Exception as e:
            print(f"DEBUG - Exception in latest_detection_times processing: {str(e)}")
            logger.error(f"Check-in error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Check-in failed: {str(e)}")
        
    # If we get here, we couldn't find any valid faces
    print("DEBUG - No recognizable faces found")
    raise HTTPException(status_code=400, detail="No recognizable faces found. Please ensure your face is visible to the camera.")

@app.get('/check-out')
async def check_out():
    global face_recognition_history, latest_detection_times

    # Check if we have recent face data
    if not face_recognition_history and not latest_detection_times:
        # If both are empty, check if there's any recent detection at all
        if latest_detected_ids and latest_detected_ids != "Unknown":
            # Create a basic detection record to use
            detection_data = {
                "identity": latest_detected_ids,
                "confidence": 0.5,  # Default confidence
                "time": datetime.now().strftime("%H:%M:%S")
            }
            return await process_single_checkout(latest_detected_ids, detection_data)
        else:
            raise HTTPException(status_code=400, detail="No faces detected. Please ensure your face is visible to the camera.")

    # If we have face recognition history, use it
    if face_recognition_history:
        # Process all faces in the recognition history
        results = []
        current_time = datetime.now()
        
        for identity, data in face_recognition_history.items():
            try:
                # Get employee details from DB
                db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db.sqlite3'))
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (identity,))
                employee = cursor.fetchone()
                conn.close()
                
                if not employee:
                    continue  # Skip if employee not found
                    
                name, department = employee
                
                # Allow check-out with any confidence level when explicitly initiated
                # Remove this line: if data['confidence'] < 0.4:
                    
                # Save attendance
                save_attendance(identity, data['time'], "check_out")
                
                # Add to results
                results.append({
                    "name": name,
                    "id": identity,
                    "department": department,
                    "confidence": f"{data['confidence']:.2%}",
                    "time": data['time']
                })
                
            except Exception as e:
                logger.error(f"Error processing check-out for {identity}: {e}")
                # Continue processing other employees even if one fails
        
        # Clear history after processing
        face_recognition_history.clear()
        
        if results:
            # Return multiple results if available
            return {
                "status": "success",
                "message": f"Checked out {len(results)} employee(s) successfully",
                "action_type": "check_out",
                "employees": results,
                # Include the first employee separately for backward compatibility
                "employee": results[0] if results else None
            }
    
    # Fallback to latest detection if available
    if latest_detection_times:
        emp_id, detection_data = next(iter(latest_detection_times.items()))
        return await process_single_checkout(emp_id, detection_data)
        
    # If we get here, we couldn't find any valid faces
    raise HTTPException(status_code=400, detail="No recognizable faces found. Please ensure your face is visible to the camera.")

# Add this helper function for single check-outs
async def process_single_checkout(emp_id, detection_data):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT emp_name, department FROM Home_employee WHERE emp_id = ?", (emp_id,))
        employee = cursor.fetchone()
        conn.close()

        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        name, department = employee

        if detection_data["confidence"] < 0.4:
            raise HTTPException(status_code=400, detail="Face recognition confidence too low")

        save_attendance(emp_id, detection_data["time"], "check_out")

        return {
            "status": "success",
            "message": "Checked out successfully",
            "action_type": "check_out",
            "employee": {
                "name": name,
                "id": emp_id,
                "department": department,
                "confidence": f"{detection_data['confidence']:.2%}",
                "time": detection_data["time"]
            }
        }
    except Exception as e:
        logger.error(f"Check-out error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Check-out failed: {str(e)}")

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

def get_db_connection(max_retries=3, retry_delay=1):
    """Get database connection with retry mechanism"""
    for attempt in range(max_retries):
        try:
            db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db.sqlite3'))
            conn = sqlite3.connect(db_path)
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

@app.get('/debug-attendance/{emp_id}')
async def debug_attendance(emp_id: str):
    """Debug endpoint to check attendance records."""
    try:
        conn = sqlite3.connect(r"../db.sqlite3")
        cursor = conn.cursor()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get attendance record
        cursor.execute("""
            SELECT id, emp_id, date, time_in_list, time_out_list 
            FROM Home_attendance 
            WHERE emp_id = ? AND date = ?
        """, (emp_id, current_date))
        
        record = cursor.fetchone()
        if record:
            id, emp_id, date, time_in, time_out = record
            return {
                "status": "found",
                "record": {
                    "id": id,
                    "emp_id": emp_id,
                    "date": date,
                    "time_in_list": time_in,
                    "time_out_list": time_out
                }
            }
        else:
            return {"status": "not_found", "message": "No attendance record found for today"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()

@app.post('/set-preference')
async def set_preference(preference: dict):
    """Update user preference for check-in/check-out and overlay display"""
    global user_preference
    
    try:
        # Update action preference if provided
        if "action" in preference:
            if preference["action"] in ["check_in", "check_out"]:
                user_preference["action"] = preference["action"]
            else:
                raise HTTPException(status_code=400, detail="Invalid action value. Must be 'check_in' or 'check_out'")
        
        # Update overlay display preference if provided
        if "show_overlay" in preference:
            user_preference["show_overlay"] = bool(preference["show_overlay"])
        
        return {"status": "success", "preference": user_preference}
    except Exception as e:
        logger.error(f"Error setting preference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/get-overlay-data')
async def get_overlay_data(emp_id: str = None):
    """Get overlay data for a specific employee or the last detected employee"""
    try:
        if emp_id is None:
            # Use the latest detected ID if no specific ID is provided
            if latest_detected_ids and latest_detected_ids != "Unknown":
                emp_id = latest_detected_ids
            else:
                return {"status": "error", "message": "No employee detected"}
        
        # Find the employee's detection data
        # confidence = None
        # if latest_detection_times and emp_id in latest_detection_times:
        #     confidence = latest_detection_times[emp_id].get("confidence", None)
        # elif face_recognition_history and emp_id in face_recognition_history:
        #     confidence = face_recognition_history[emp_id].get("confidence", None)
        
        # Generate overlay data
        overlay_data = generate_overlay_data(emp_id)
        
        if overlay_data:
            return {"status": "success", "data": overlay_data}
        else:
            return {"status": "error", "message": "Could not generate overlay data"}
    except Exception as e:
        logger.error(f"Error getting overlay data: {e}")
        return {"status": "error", "message": str(e)}

# SSE endpoint for recent activity updates
@app.get('/recent-activity-stream')
async def recent_activity_stream():
    """SSE endpoint for recent activity updates."""
    from starlette.responses import StreamingResponse
    import asyncio

    async def event_generator():
        while True:
            try:
                # Wait for new activity
                activity = await asyncio.get_event_loop().run_in_executor(None, recent_activity_queue.get)
                yield f"data: {json.dumps(activity)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    threading.Thread(target=lambda: loop.run_until_complete(video_capture()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=5600)
