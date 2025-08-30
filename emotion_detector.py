

import cv2
import os
import urllib.request
from fer import FER
import threading
import time

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


# --- Video Capture and Threaded Detection ---
FRAME_WIDTH = 640  # Resize width for performance
FRAME_SKIP = 2     # Process every Nth frame for detection

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

latest_results = []
latest_frame = None
lock = threading.Lock()
stop_thread = False

def detection_thread():
    global latest_results, latest_frame, stop_thread
    frame_count = 0
    while not stop_thread:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                # Run detection on resized frame for speed
                small_frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
                results = emotion_detector.detect_emotions(small_frame)
                # Scale results back to original frame size
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                for r in results:
                    x, y, w, h = r["box"]
                    r["box"] = [int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)]
                with lock:
                    latest_results = results
        time.sleep(0.01)

# Start detection thread
thread = threading.Thread(target=detection_thread, daemon=True)
thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    # Resize for display performance
    display_frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
    with lock:
        latest_frame = frame.copy()
        results = list(latest_results)

    # Draw results
    for result in results:
        x, y, w, h = result["box"]
        emotions = result["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size != 0:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
            label = f"{gender}, {age}, {dominant_emotion.capitalize()}"
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display_frame, (x, y - text_height - 10), (x + text_width, y), (255, 0, 0), -1)
            cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Real-time Analysis', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_thread = True
thread.join()
cap.release()
cv2.destroyAllWindows()
