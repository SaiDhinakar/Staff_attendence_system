from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import json
from PIL import Image

class FaceDetect:
    def __init__(self, db_file="face_embeddings.json"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')
        self.mtcnn = None
        self.resnet = None
        self.db_file = db_file
        self.embeddings = self.load_embeddings()

    def load_model(self):
        """Load MTCNN and ResNet models for face detection and recognition."""
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print("Models loaded successfully.")

    def load_embeddings(self):
        """Load face embeddings from a JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return json.load(f)
        else:
            print(f"Embedding file {self.db_file} not found.")
            return {}

    def img_to_encoding(self, img_path):
        """Detects a face in an image and returns the face embedding."""
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

        img_aligned = self.mtcnn(img)
        if img_aligned is not None:
            embedding = self.resnet(img_aligned.unsqueeze(0).to(self.device)).detach().cpu()
            return embedding.squeeze()
        else:
            print("No face detected")
            return None

    def face_recognition(self, image_path):
        """Recognizes a face by comparing to stored embeddings."""
        encoding = self.img_to_encoding(image_path)
        if encoding is None:
            return "Unknown", None

        min_dist = float('inf')
        identity = "Unknown"

        for name, encodings in self.embeddings.items():
            for db_enc in encodings:
                dist = torch.nn.functional.pairwise_distance(
                    encoding.unsqueeze(0),
                    torch.tensor(db_enc).unsqueeze(0)
                ).item()
                if dist < min_dist:
                    min_dist = dist
                    identity = name

        threshold = 0.6
        if min_dist > threshold:
            print(f"Face detected but not recognized. Distance: {min_dist:.4f}")
            return "Unknown", min_dist
        else:
            print(f"Recognized as {identity}. Distance: {min_dist:.4f}")
            return identity, min_dist

# Example usage:
face_detector = FaceDetect()
face_detector.load_model()
identity, distance = face_detector.face_recognition("test_m.jpg")
