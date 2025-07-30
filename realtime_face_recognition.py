import os
from PIL import Image
import numpy as np
import face_recognition
import cv2
from datetime import datetime, timezone
import time

def load_face_encodings(data_dir="data"):
    """Loads all images for each person and returns lists of encodings and names."""
    known_face_encodings = []
    known_face_names = []

    print("\nLoading face data from all folders...")

    # Loop over each person's directory in data/
    for person_name in os.listdir(data_dir):
        person_folder = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_folder):
            continue  # Skip files

        # Loop over each image file in the person's folder
        for filename in os.listdir(person_folder):
            file_path = os.path.join(person_folder, filename)
            if not (filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
                continue  # Skip non-image files

            try:
                # Load with PIL, convert to RGB
                image = Image.open(file_path).convert('RGB')
                image_np = np.ascontiguousarray(np.array(image))

                # Detect faces and get encodings
                face_locations = face_recognition.face_locations(image_np)
                if not face_locations:
                    print(f"No face found in {file_path}, skipping.")
                    continue

                face_encodings = face_recognition.face_encodings(image_np, face_locations)
                for encoding in face_encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(person_name)
                    print(f"Added encoding for {person_name} from {filename}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"\nTotal faces loaded: {len(known_face_encodings)}")
    return known_face_encodings, known_face_names

def initialize_camera(preferred_index=0):
    camera_indices = [preferred_index, 0, 1, -1]
    for idx in camera_indices:
        print(f"Trying camera index: {idx}")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Failed to open camera {idx}")
            continue
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {idx}")
            cap.release()
            continue
        print(f"Successfully opened camera {idx}")
        return cap, idx
    return None, None

def main():
    print("=== Face Recognition System ===")
    print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Load all known faces
    known_face_encodings, known_face_names = load_face_encodings("data")

    if not known_face_encodings:
        print("\nError: No faces were loaded. Please check your image files.")
        return

    print("\nInitializing video capture...")
    video_capture, camera_index = initialize_camera()

    if video_capture is None:
        print("Error: Could not initialize any camera!")
        print("Please check your webcam connection and permissions")
        return

    print(f"\nUsing camera index: {camera_index}")
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to grab frame")
                video_capture.release()
                time.sleep(1)
                video_capture, _ = initialize_camera(camera_index)
                if video_capture is None:
                    break
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during recognition: {e}")

    finally:
        print("\nCleaning up...")
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"End Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

if __name__ == "__main__":
    main()