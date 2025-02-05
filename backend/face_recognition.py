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

app = FastAPI()

class FaceDetect:
    def __init__(self, db_file="backend/face_embeddings.json"):
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

        # Ensure tensor has the correct shape before passing to the model
        if len(face_tensor.shape) == 3:  # If shape is [3, 160, 160], add batch dim
            face_tensor = face_tensor.unsqueeze(0)  # Shape becomes [1, 3, 160, 160]

        embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()

        min_dist = float('inf')
        identity = "Unknown"

        for name, encodings in self.embeddings.items():
            for db_enc in encodings:
                dist = torch.nn.functional.pairwise_distance(
                    embedding,
                    torch.tensor(db_enc).unsqueeze(0)
                ).item()

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
                    face_tensor = face_tensor.unsqueeze(0) if len(
                        face_tensor.shape) == 3 else face_tensor
                    identity, dist = self.recognize_face(face_tensor)

                    if identity != "Unknown":
                        detected_identities.append(identity)
                        detection_times[identity] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

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
            break

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

            # Send the frame and detection results
            yield f"data: {json.dumps({'image': encoded_frame, 'detected_ids': latest_detected_ids, 'detection_times': latest_detection_times})}\n\n"

        time.sleep(0.1)  # Control frame rate

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
