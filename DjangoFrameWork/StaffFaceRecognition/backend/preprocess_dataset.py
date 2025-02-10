import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm  # Progress bar

class FacePreprocessor:
    def __init__(self, input_dir, output_dir="processed_dataset"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def process_images(self):
        """Process dataset: Detect faces, crop, and save."""
        for class_name in tqdm(os.listdir(self.input_dir), desc="Processing Classes"):
            class_path = os.path.join(self.input_dir, class_name)
            if not os.path.isdir(class_path):
                continue  # Skip non-folder files

            # Create corresponding output class folder
            output_class_path = os.path.join(self.output_dir, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for image_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}", leave=False):
                image_path = os.path.join(class_path, image_name)

                # Load image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Skipping invalid image: {image_path}")
                    continue

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                # Detect faces
                face = self.mtcnn(img_pil)
                if face is None:
                    print(f"No face detected: {image_path}")
                    continue

                # Convert tensor to image and save
                face = face.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
                face = (face * 255).astype("uint8")  # Convert from [0,1] to [0,255]
                output_path = os.path.join(output_class_path, image_name)

                # Save cropped face
                cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        print(f"\n✅ Dataset processed! Faces saved in '{self.output_dir}'")


if __name__ == "__main__":
    dataset_path = "dataset"
    processor = FacePreprocessor(dataset_path)
    processor.process_images()
