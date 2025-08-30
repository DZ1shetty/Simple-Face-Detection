import cv2
import numpy as np

# --- Model and Label Setup ---

# Define paths to the model files
face_proto = "models/deploy.prototxt"
face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
age_proto = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_proto = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
emotion_model = "models/emotion-ferplus-8.onnx"

# Define the labels for each model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']

# --- Load the Networks ---

print("Loading models...")
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)
emotion_net = cv2.dnn.readNet(emotion_model)
print("Models loaded successfully.")

# --- Video Capture ---

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video stream.")
    exit()

padding = 20

while cv2.waitKey(1) < 0:
    has_frame, frame = video.read()
    if not has_frame:
        cv2.waitKey()
        break

    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # --- Face Detection ---
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x1, y1, x2, y2) = box.astype("int")
            face_roi = frame[max(0, y1 - padding):min(y2 + padding, frame_height - 1),
                             max(0, x1 - padding):min(x2 + padding, frame_width - 1)]
            if face_roi.any():
                # --- Age and Gender Prediction ---
                age_gender_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(age_gender_blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]
                age_net.setInput(age_gender_blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]
                # --- Emotion Prediction ---
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (64, 64))
                emotion_blob = cv2.dnn.blobFromImage(resized_face, 1.0, (64, 64), (0), swapRB=False)
                emotion_net.setInput(emotion_blob)
                emotion_preds = emotion_net.forward()
                emotion = EMOTION_LIST[emotion_preds[0].argmax()]
                label = f"{gender}, {age}, {emotion}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Emotion, Age, and Gender Detector", frame)

video.release()
cv2.destroyAllWindows()
