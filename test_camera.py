import cv2

def test_webcam():
    # Try different camera indices
    for camera_index in [0, 1, -1]:
        print(f"\nTrying camera index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            continue
            
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_index}")
            cap.release()
            continue
            
        print(f"Successfully opened camera {camera_index}")
        print(f"Frame shape: {frame.shape}")
        
        # Show the frame
        cv2.imshow('Camera Test', frame)
        cv2.waitKey(2000)  # Wait for 2 seconds
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        return camera_index
    
    return None

if __name__ == "__main__":
    print("Testing webcam access...")
    working_camera = test_webcam()
    
    if working_camera is not None:
        print(f"\nWorking camera found at index: {working_camera}")
        print("Use this index in your face recognition script")
    else:
        print("\nNo working camera found!")
        print("Please check your webcam connection and permissions")