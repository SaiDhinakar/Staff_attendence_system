import cv2
import torch
import json
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from ultralytics import YOLO  # Import YOLOv8


class FaceDetect:
    def __init__(self, db_file="backend/face_embeddings.json"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')

        # Initialize models
        self.yolo = YOLO("backend/yolo_model/yolo11n-face.pt")  # Load YOLOv8 face detection model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load known embeddings
        self.db_file = db_file
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        """Load face embeddings from a JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return {k: [torch.tensor(e) for e in v] for k, v in json.load(f).items()}
        else:
            print(f"Embedding file {self.db_file} not found.")
            return {}

    def recognize_face(self, face_tensor):
        """Compares face embeddings to known faces in the database."""
        if face_tensor is None:
            return "Unknown", None

        face_tensor = face_tensor.unsqueeze(0) if len(face_tensor.shape) == 3 else face_tensor
        embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()

        min_dist = float('inf')
        identity = "Unknown"

        for name, encodings in self.embeddings.items():
            for db_enc in encodings:
                dist = torch.nn.functional.pairwise_distance(
                    embedding, db_enc.unsqueeze(0)
                ).item()
                if dist < min_dist:
                    min_dist = dist
                    identity = name

        threshold = 0.6
        return (identity, min_dist) if min_dist <= threshold else ("Unknown", min_dist)

    def process_frame(self, frame):
        """Detect faces using YOLO and perform recognition."""
        results = self.yolo(frame)[0]  # Run YOLOv8 on the frame
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_width = x2 - x1

            # Define face size limits
            min_face_size = 80
            max_face_size = 250

            identity = "Unknown"

            if min_face_size < face_width < max_face_size:
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                face_img = cv2.resize(face_img, (160, 160))  # Resize to fit the recognition model
                face_tensor = torch.tensor(face_img).permute(2, 0, 1).float().div(255).unsqueeze(0)
                identity, dist = self.recognize_face(face_tensor)

                # Display results
                print(f"Recognized: {identity} | Distance: {dist:.4f}")
                cv2.putText(frame, f"{identity}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Assign color based on identity status
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    def run_camera(self):
        """Starts live face detection and recognition."""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Face Recognition System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example Usage:
face_detector = FaceDetect()
face_detector.run_camera()