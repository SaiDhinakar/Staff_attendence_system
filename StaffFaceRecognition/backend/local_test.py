import cv2
import torch
import json
import os
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

class FaceRecognizer:
    def __init__(self, db_file="face_embeddings.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize models
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Load stored embeddings
        self.db_file = db_file
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        """Load face embeddings from a JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return json.load(f)
        else:
            print(f"Embedding file {self.db_file} not found. No stored faces.")
            return {}

    def recognize_face(self, face_tensor):
        """Compare face embeddings with stored embeddings."""
        if face_tensor is None:
            return "Unknown", None

        face_tensor = face_tensor.unsqueeze(0) if len(face_tensor.shape) == 3 else face_tensor
        embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()

        min_dist = float("inf")
        identity = "Unknown"

        for name, encodings in self.embeddings.items():
            for db_enc in encodings:
                db_enc = torch.tensor(db_enc)
                dist = torch.nn.functional.pairwise_distance(embedding, db_enc.unsqueeze(0)).item()

                if dist < min_dist:
                    min_dist = dist
                    identity = name

        threshold = 0.6
        return (identity, min_dist) if min_dist <= threshold else ("Unknown", min_dist)

    def process_frame(self, frame):
        """Detect, extract, and recognize face."""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)
        class_names = ['sai']  # Add more names if needed

        if boxes is None or len(boxes) == 0:
            return frame, "Unknown"  # No faces detected

        # Pick the largest detected face
        largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = map(int, largest_box)

        face_width = x2 - x1
        min_face_size, max_face_size = 120, 250
        identity = "Unknown"
        confidence = 0.0

        if min_face_size < face_width < max_face_size:
            face_img = img.crop((x1, y1, x2, y2))
            face_tensor = self.mtcnn(face_img)

            if face_tensor is not None:
                identity, dist = self.recognize_face(face_tensor)

                # Confidence calculation (Ensure dist is valid)
                confidence = round((1 - dist) * 100, 2) if dist is not None and dist <= 1.0 else 0.0

                # Display Name and Confidence
                label = f"{identity} ({confidence}%)" if identity != "Unknown" else "Unknown"

                # Draw name and confidence on frame
                text_color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            # Draw bounding box
            box_color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        return frame, identity


def main():
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue  # Skip frame if there's an issue

        # Process frame
        frame, detected_name = recognizer.process_frame(frame)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
