import cv2
import torch
import json
import os
import time
import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import base64
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from datetime import datetime


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
        """Detect faces in the frame and trigger recognition if needed."""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _ = self.mtcnn.detect(img)
        detected_identities = []
        detection_times = {}

        if boxes is None:
            return frame, detected_identities, detection_times  # No faces detected

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_width = x2 - x1

            min_face_size = 120
            max_face_size = 250

            identity = "Unknown"

            if min_face_size < face_width < max_face_size:
                face_img = img.crop((x1, y1, x2, y2))
                face_tensor = self.mtcnn(face_img)

                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0) if len(face_tensor.shape) == 3 else face_tensor
                    identity, dist = self.recognize_face(face_tensor)

                    if identity != "Unknown":
                        detected_identities.append(identity)
                        detection_time = time.strftime("%H:%M:%S", time.gmtime())
                        detection_times[identity] = detection_time

                        # Save attendance to SQLite
                        save_attendance(identity, detection_time)
                        cv2.putText(frame, f"{identity}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame, detected_identities, detection_times

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
        modified_frame, detected_ids, detection_times = face_detector.process_frame(frame)

        # Store latest results in global variables
        latest_frame = modified_frame
        latest_detected_ids = detected_ids
        latest_detection_times = detection_times

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


def save_attendance(emp_id, detection_time):
    conn = sqlite3.connect("db.sqlite3")  # Connect to the database
    cursor = conn.cursor()

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Check if the employee already has an attendance record for today
    cursor.execute("SELECT id, time_in, time_out FROM Home_attendance WHERE emp_id = ? AND date = ?",
                   (emp_id, current_date))
    record = cursor.fetchone()

    if record:
        # Record exists, update time_out by appending new time
        attendance_id, time_in, time_out = record

        if time_out:
            updated_time_out = f"{time_out},{detection_time}"
        else:
            updated_time_out = detection_time

        cursor.execute("UPDATE Home_attendance SET time_out = ? WHERE id = ?", (updated_time_out, attendance_id))

    else:
        # Insert a new record with the first detection time
        cursor.execute(
            "INSERT INTO Home_attendance (date, time_in, time_out, emp_id, status) VALUES (?, ?, ?, ?, NULL)",
            (current_date, detection_time, detection_time, emp_id))

    conn.commit()
    conn.close()



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
