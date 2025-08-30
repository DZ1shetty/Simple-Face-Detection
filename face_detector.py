# Import the necessary libraries
import cv2
import time

# --- Configuration ---
# You can tweak these parameters for performance vs. accuracy
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)
BLUR_STRENGTH = (23, 23) # Must be odd numbers
SKIP_FRAMES = 3 # Process every 3rd frame to improve performance

# --- State Variables ---
blur_faces = False # Start with rectangles, not blurring
frame_counter = 0
faces = [] # Store the last known face locations

# Load the pre-trained Haar Cascade classifier for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading cascade file: {e}")
    exit()

# Initialize video capture from the default webcam (camera index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Starting video stream...")
print("Press 'b' to toggle blur effect.")
print("Press 's' to save snapshot.")
print("Press 'q' to quit.")

# Loop to capture frames from the webcam continuously
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was not captured successfully, break the loop
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame_counter += 1
    # Only run detection on specified frames to save resources
    if frame_counter % SKIP_FRAMES == 0:
        # Convert the captured frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

    # Process each detected face using the last known locations
    for (x, y, w, h) in faces:
        if blur_faces:
            # Apply a Gaussian blur to the face region
            face_roi = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, BLUR_STRENGTH, 0)
            frame[y:y+h, x:x+w] = blurred_face
        else:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # --- Display On-Screen Information ---
    # Display the face count
    face_count_text = f'Faces Detected: {len(faces)}'
    cv2.putText(frame, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the current mode (Rectangle or Blur)
    mode_text = f"Mode: {'Blur' if blur_faces else 'Rectangle'}"
    cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Display the resulting frame
    cv2.imshow('Improved Face Detector', frame)

    # --- Handle Key Presses ---
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if the 'q' key is pressed
    if key == ord('q'):
        break
    # Toggle blur effect if 'b' is pressed
    elif key == ord('b'):
        blur_faces = not blur_faces
    # Save a snapshot if 's' is pressed
    elif key == ord('s'):
        # Create a unique filename with a timestamp
        filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved snapshot as {filename}")

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Video stream stopped.")
