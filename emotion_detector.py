

import cv2
import os
import urllib.request
from fer import FER
import threading
import time
import numpy as np

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

FRAME_SKIP = 6  # Initial value; will be adapted dynamically
MIN_FRAME_SKIP = 3
MAX_FRAME_SKIP = 10

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set camera to highest supported resolution
def set_max_resolution(cap):
    # Try common high resolutions
    for width, height in [(1920,1080), (1280,720), (1024,768), (800,600)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_w >= width and actual_h >= height:
            break
set_max_resolution(cap)
def enhance_image(img):
    # Very light sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 4.5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    # Light contrast enhancement (CLAHE)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Very mild color boost
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 4)  # Very gentle saturation boost
    s = np.clip(s, 0, 255)
    hsv_boosted = cv2.merge((h, s, v))
    vibrant = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
    return vibrant


# Tracking state
latest_results = []
latest_frame = None
lock = threading.Lock()
stop_thread = False
trackers = None
tracking_boxes = []
tracking_labels = []

def detection_thread():
    global latest_results, latest_frame, stop_thread
    global trackers, tracking_boxes, tracking_labels, FRAME_SKIP
    frame_count = 0
    while not stop_thread:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            frame_count += 1
            # Only run detection every FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0 or trackers is None or len(tracking_boxes) == 0:
                # Run detection on a smaller frame for speed
                detect_width = 480
                small_frame = cv2.resize(frame, (detect_width, int(frame.shape[0] * detect_width / frame.shape[1])))
                results = emotion_detector.detect_emotions(small_frame)
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                for r in results:
                    x, y, w, h = r["box"]
                    r["box"] = [int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)]
                with lock:
                    if results:
                        latest_results = results
                        # Update trackers with new detections
                        # Always use legacy namespace if available for both MultiTracker and TrackerKCF
                        trackers = None
                        tracker_factory = None
                        # Prefer CSRT tracker for smoother tracking if available
                        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'MultiTracker_create'):
                            trackers = cv2.legacy.MultiTracker_create()
                            if hasattr(cv2.legacy, 'TrackerCSRT_create'):
                                tracker_factory = cv2.legacy.TrackerCSRT_create
                            else:
                                tracker_factory = cv2.legacy.TrackerKCF_create
                        elif hasattr(cv2, 'MultiTracker_create'):
                            trackers = cv2.MultiTracker_create()
                            if hasattr(cv2, 'TrackerCSRT_create'):
                                tracker_factory = cv2.TrackerCSRT_create
                            else:
                                tracker_factory = cv2.TrackerKCF_create
                        else:
                            raise RuntimeError("MultiTracker or TrackerCSRT/KCF is not available in your OpenCV installation.")
                        tracking_boxes = []
                        tracking_labels = []
                        # For smoothing: keep a history of last N boxes for each track
                        global box_history
                        box_history = [[] for _ in results]
                        for idx, r in enumerate(results):
                            x, y, w, h = r["box"]
                            tracker = tracker_factory()
                            trackers.add(tracker, frame, (x, y, w, h))
                            tracking_boxes.append([x, y, w, h])
                            emotions = r["emotions"]
                            dominant_emotion = max(emotions, key=emotions.get)
                            tracking_labels.append(dominant_emotion)
                            box_history[idx].append([x, y, w, h])
                        # Adaptive frame skip: more faces = lower skip, fewer faces = higher skip
                        num_faces = len(results)
                        if num_faces >= 3:
                            FRAME_SKIP = MIN_FRAME_SKIP
                        elif num_faces == 2:
                            FRAME_SKIP = max(MIN_FRAME_SKIP, int((MIN_FRAME_SKIP+MAX_FRAME_SKIP)//2))
                        elif num_faces == 1:
                            FRAME_SKIP = MAX_FRAME_SKIP
                        else:
                            FRAME_SKIP = MAX_FRAME_SKIP
            else:
                # Update tracking boxes
                if trackers is not None:
                    ok, boxes = trackers.update(frame)
                    if ok:
                        # Smoothing: moving average over last N positions
                        SMOOTH_N = 4
                        new_boxes = []
                        for i, box in enumerate(boxes):
                            box = list(map(int, box))
                            if 'box_history' in globals() and i < len(box_history):
                                box_history[i].append(box)
                                if len(box_history[i]) > SMOOTH_N:
                                    box_history[i].pop(0)
                                # Compute average
                                avg_box = [int(sum(coord[j] for coord in box_history[i]) / len(box_history[i])) for j in range(4)]
                                new_boxes.append(avg_box)
                            else:
                                new_boxes.append(box)
                        tracking_boxes = new_boxes
                    # Grace period for lost tracks: if not ok, keep last boxes for a few frames
                    else:
                        if 'lost_count' not in globals():
                            global lost_count
                            lost_count = 0
                        lost_count += 1
                        if lost_count < 5:
                            pass  # Keep last tracking_boxes
                        else:
                            tracking_boxes = []
                            if 'box_history' in globals():
                                box_history = []
                            lost_count = 0
            # Sleep a bit to reduce CPU usage
        time.sleep(0.01)

# Start detection thread
thread = threading.Thread(target=detection_thread, daemon=True)
thread.start()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    # Enhance for display (no resizing, use full native resolution)
    display_frame = enhance_image(frame)
    with lock:
        latest_frame = frame.copy()
        results = list(latest_results)
        boxes = list(tracking_boxes)
        labels = list(tracking_labels)

    # Draw tracked faces if available, else draw detection results
    if boxes:
        for i, (x, y, w, h) in enumerate(boxes):
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size != 0:
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]
                label = f"{gender}, {age}, {labels[i].capitalize() if i < len(labels) else ''}"
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display_frame, (x, y - text_height - 10), (x + text_width, y), (255, 0, 0), -1)
                cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
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

    window_name = 'Real-time Analysis'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # Track fullscreen state
    if 'fullscreen_state' not in globals():
        global fullscreen_state
        fullscreen_state = True
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen_state = True
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('m'):
        fullscreen_state = False
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

stop_thread = True
thread.join()
cap.release()
cv2.destroyAllWindows()
