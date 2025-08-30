
import cv2
import os
import urllib.request
from fer import FER

# --- Function to Download Models ---
def download_models():
    """
    Checks for the age and gender models and downloads them if they are missing.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # URLs for the models from a reliable GitHub repository
    age_proto_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/age_deploy.prototxt"
    age_model_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/age_net.caffemodel"
    gender_proto_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/gender_deploy.prototxt"
    gender_model_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/gender_net.caffemodel"

    # File paths
    age_proto_path = os.path.join(models_dir, "age_deploy.prototxt")
    age_model_path = os.path.join(models_dir, "age_net.caffemodel")
    gender_proto_path = os.path.join(models_dir, "gender_deploy.prototxt")
    gender_model_path = os.path.join(models_dir, "gender_net.caffemodel")

    # Download files if they don't exist
    if not os.path.exists(age_proto_path):
        print("Downloading age_deploy.prototxt...")
        urllib.request.urlretrieve(age_proto_url, age_proto_path)
    if not os.path.exists(age_model_path):
        print("Downloading age_net.caffemodel...")
        urllib.request.urlretrieve(age_model_url, age_model_path)
    if not os.path.exists(gender_proto_path):
        print("Downloading gender_deploy.prototxt...")
        urllib.request.urlretrieve(gender_proto_url, gender_proto_path)
    if not os.path.exists(gender_model_path):
        print("Downloading gender_net.caffemodel...")
        urllib.request.urlretrieve(gender_model_url, gender_model_path)
    
    print("All models are ready.")
    return age_proto_path, age_model_path, gender_proto_path, gender_model_path

# --- Setup ---
# Automatically download the models first
age_proto, age_model, gender_proto, gender_model = download_models()

# Load the models
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)
# Initialize the FER library for emotion detection, using MTCNN for better face detection
emotion_detector = FER(mtcnn=True) 

# Define model constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# --- Video Capture ---
# Start video capture from the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# --- Main Loop ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # The 'fer' library detects both faces and emotions.
    # It returns a list of dictionaries, one for each face found.
    results = emotion_detector.detect_emotions(frame)

    # Loop over each detected face
    for result in results:
        # Get bounding box coordinates and the emotions dictionary
        x, y, w, h = result["box"]
        emotions = result["emotions"]
        
        # Determine the emotion with the highest confidence score
        dominant_emotion = max(emotions, key=emotions.get)

        # Extract the face Region of Interest (ROI) for age and gender prediction
        face_roi = frame[y:y+h, x:x+w]
        
        # Proceed only if the ROI is valid (not empty)
        if face_roi.size != 0:
            # Create a blob from the ROI to feed into the age/gender models
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            # --- Display Results on the Frame ---
            label = f"{gender}, {age}, {dominant_emotion.capitalize()}"
            
            # Draw a blue bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Create a background for the text label for better readability
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), (255, 0, 0), -1)

            # Put the final text label on the frame
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the processed frame in a window
    cv2.imshow('Real-time Analysis', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
