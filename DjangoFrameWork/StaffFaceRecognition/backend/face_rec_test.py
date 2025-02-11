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
from datetime import datetime
from pydantic import BaseModel
from collections import defaultdict
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this to specific origins if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaceDetect:
    def __init__(self, db_file="face_embeddings.json"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')

        # Initialize models
        self.mtcnn = MTCNN(keep_all=True, device=self.device)  # Detect multiple faces
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load known embeddings
        self.db_file = db_file
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        """Load face embeddings from a JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return json.load(f)
        else:
            print(f"Embedding file {self.db_file} not found.")
            return {}

    def recognize_face(self, face_tensor):
        """Compares face embeddings to known faces in the database."""
        if face_tensor is None:
            return "Unknown", None

        if len(face_tensor.shape) == 3:  # Ensure correct shape
            face_tensor = face_tensor.unsqueeze(0)

        embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()

        min_dist = float('inf')
        identity = "Unknown"

        for name, encodings in self.embeddings.items():
            for db_enc in encodings:
                dist = torch.nn.functional.pairwise_distance(
                    embedding,
                    torch.tensor(db_enc).unsqueeze(0)
                )

                if dist.numel() == 1:
                    dist = dist.item()
                else:
                    dist = dist.mean().item()

                if dist < min_dist:
                    min_dist = dist
                    identity = name

        threshold = 0.6
        return (identity, min_dist) if min_dist <= threshold else ("Unknown", min_dist)


    def process_frame(self, frame):
        """Process a video frame, detect faces, and recognize identities."""
        global latest_detection_data, latest_detection_times  # Ensure both are updated

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)

        detection_data = {
            "identity": "Unknown",
            "detection_times": {},
            "confidence": None
        }

        if boxes is not None and len(boxes) > 0:
            largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            x1, y1, x2, y2 = map(int, largest_box)

            face_img = img.crop((x1, y1, x2, y2))
            face_tensor = self.mtcnn(face_img)

            if face_tensor is not None:
                face_tensor = face_tensor.unsqueeze(0) if len(face_tensor.shape) == 3 else face_tensor
                identity, dist = self.recognize_face(face_tensor)

                if identity != "Unknown":
                    detection_time = time.strftime("%H:%M:%S", time.localtime())
                    confidence = 1 - dist

                    detection_data = {
                        "identity": identity,
                        "detection_times": {identity: {"time": detection_time, "confidence": confidence}},
                        "confidence": confidence
                    }

                    # ✅ Fix: Update latest_detection_times properly
                    latest_detection_times[identity] = {
                        "time": detection_time,
                        "confidence": confidence
                    }

                    # Debugging Print Statements
                    print(f"Detected: {identity}, Confidence: {confidence}")
                    print(f"Updated latest_detection_times: {latest_detection_times}")

        # ✅ Update global latest_detection_data
        latest_detection_data = {
            "frame": frame,
            "identity": detection_data["identity"],
            "detection_times": detection_data["detection_times"],
            "confidence": detection_data["confidence"],
            "last_detection": time.strftime("%H:%M:%S", time.localtime())
        }

        return frame, detection_data["identity"], detection_data["detection_times"]

    def encode_image(self, frame):
        """Convert frame to base64 encoded string."""
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')

face_detector = FaceDetect()

# Global variables to store latest frame & detection results
latest_frame = None
latest_detected_ids = []
latest_detection_times = {}

def video_capture():
    """Continuously capture and process video frames."""
    global latest_frame, latest_detected_ids, latest_detection_times

    cap = cv2.VideoCapture(0)  # Open default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue  # Skip iteration if no frame is read

        # Process frame for face detection and recognition
        modified_frame, detected_ids, detection_time = face_detector.process_frame(frame)

        # Store latest results in global variables
        latest_frame = modified_frame
        latest_detected_ids = detected_ids
        latest_detection_time = detection_time

    cap.release()

def generate_video_stream():
    """Generate a continuous video stream with detection results."""
    global latest_frame, latest_detected_ids, latest_detection_times

    while True:
        if latest_frame is not None:
            # Encode the latest frame to base64
            encoded_frame = face_detector.encode_image(latest_frame)

            # Send the frame and detection results in correct SSE format
            data = {
                "image": encoded_frame
            }
            yield f"data: {json.dumps(data)}\n\n"

        time.sleep(0.1)  # Control frame rate

def save_attendance(emp_id, detection_time, check_type):
    conn = sqlite3.connect(r"/mnt/data/PROJECTS/Staff_attendence_system/DjangoFrameWork/StaffFaceRecognition/db.sqlite3")  # Connect to the database
    cursor = conn.cursor()

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Check if the employee already has an attendance record for today
    cursor.execute("SELECT id, time_in_list, time_out_list FROM Home_attendance WHERE emp_id = ? AND date = ?",
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
    conn.close()

class EmbeddingRequest(BaseModel):
    db_path: str
    output_file: str = "backend/face_embeddings.json"


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
def load_embeddings(input_file: str = "backend/face_embeddings.json"):
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

    print(f"Detected emp_id: {emp_id}, Confidence: {detection_data['confidence']}")  # Debugging

    if detection_data["confidence"] < 0.4:
        raise HTTPException(status_code=400, detail="Face recognition confidence too low")

    save_attendance(emp_id, detection_data["time"], "check_in")

    return {
        "status": "success",
        "message": "Checked in successfully",
        "employee": {
            "id": emp_id,
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

    print(f"Detected emp_id: {emp_id}, Confidence: {detection_data['confidence']}")  # Debugging

    if detection_data["confidence"] < 0.5:  # Higher threshold for check-out
        raise HTTPException(status_code=400, detail="Face recognition confidence too low")

    save_attendance(emp_id, detection_data["time"], "check_out")

    return {
        "status": "success",
        "message": "Checked out successfully",
        "employee": {
            "id": emp_id,
            "confidence": f"{detection_data['confidence']:.2%}",
            "time": detection_data["time"]
        }
    }


@app.get('/video_stream')
async def video_stream():
    """Stream video with face detection results to the client."""
    return StreamingResponse(generate_video_stream(), media_type='text/event-stream')

if __name__ == '__main__':
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=video_capture, daemon=True)
    video_thread.start()

    # Start the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5600)