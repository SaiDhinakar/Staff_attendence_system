import cv2
import torch
import json
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


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
        if boxes is None:
            return frame  # No faces detected

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_width = x2 - x1

            # Define face distance threshold
            min_face_size = 120
            max_face_size = 250

            # Initialize identity to avoid UnboundLocalError
            identity = "Unknown"

            if min_face_size < face_width < max_face_size:
                face_img = img.crop((x1, y1, x2, y2))
                face_tensor = self.mtcnn(face_img)

                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0) if len(
                        face_tensor.shape) == 3 else face_tensor  # Ensure correct shape
                    identity, dist = self.recognize_face(face_tensor)

                    # 🔹 Print the recognized identity in the console
                    print(f"Recognized: {identity} | Distance: {dist:.4f}")

                    # 🔹 Display identity on the video frame
                    cv2.putText(frame, f"{identity}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Assign color based on identity status
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    def run_camera(self):
        """Starts live face detection-based attendance system."""
        cap = cv2.VideoCapture(0)  # Open webcam (change to camera index if needed)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)

            cv2.imshow("Face Recognition System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()


# Example Usage:
face_detector = FaceDetect()
face_detector.run_camera()
