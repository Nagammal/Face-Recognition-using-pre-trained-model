import sys
import cv2
import numpy as np
import face_recognition
from PIL import Image
import dlib

def print_versions():
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pillow: {Image.__version__}")
    print(f"face_recognition: {face_recognition.__version__}")
    print(f"dlib: {dlib.__version__}")

def test_face_detection(image_path):
    print(f"\nTesting image: {image_path}")
    
    # Load with PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    print(f"Image shape: {image_np.shape}")
    print(f"Image dtype: {image_np.dtype}")
    print(f"Is contiguous: {image_np.flags['C_CONTIGUOUS']}")
    
    try:
        # Try face detection
        face_locations = face_recognition.face_locations(image_np)
        print(f"Found {len(face_locations)} faces!")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== System Information ===")
    print_versions()
    
    # Test images
    test_images = [
        "data/Nagammal/image7.jpeg",
        "data/Ananth/image4.jpg",
        "data/Shivdev/image6.jpg"
    ]
    
    for img_path in test_images:
        success = test_face_detection(img_path)
        print(f"Test {'succeeded' if success else 'failed'} for {img_path}")