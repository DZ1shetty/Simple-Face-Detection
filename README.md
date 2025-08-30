
# Emotion, Age, and Gender Detector

This project is a real-time emotion, age, and gender detection application using Python, OpenCV, and the `fer` library. It uses your webcam to analyze faces in live video, automatically downloads required models, and displays results on the video stream.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Concepts Used](#concepts-used)
- [Troubleshooting](#troubleshooting)

---


## Features
- Real-time emotion, age, and gender detection using webcam
- Automatic download of required deep learning models (no manual steps)
- Uses robust `fer` library for emotion recognition
- Uses OpenCV DNN for age and gender prediction
- Results are displayed live on the video stream


## Requirements
- Python 3.7 or higher
- OpenCV (`opencv-python`)
- TensorFlow
- fer
- moviepy
- A working webcam


## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/DZ1shetty/Simple-Face-Detection.git
   cd Simple-Face-Detection
   ```
2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```bash
   pip install opencv-python tensorflow fer moviepy
   ```


## Usage
1. Make sure your webcam is connected and accessible.
2. Run the script:
   ```bash
   python emotion_detector.py
   ```
3. The script will automatically download the required models if not present.
4. The webcam window will open and display live emotion, age, and gender predictions.
5. Press `q` to quit the application.


## Code Overview
The main logic is in `emotion_detector.py`:
- Downloads age and gender models if not present
- Loads models for age, gender, and emotion detection
- Captures video from your default webcam
- Detects faces and predicts emotion, age, and gender in real-time
- Displays results on the video stream

### Example Code Snippet
```python
import cv2
from fer import FER
emotion_detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)
while True:
   ret, frame = cap.read()
   results = emotion_detector.detect_emotions(frame)
   for result in results:
      x, y, w, h = result["box"]
      emotions = result["emotions"]
      dominant_emotion = max(emotions, key=emotions.get)
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
      cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
   cv2.imshow('Emotion Detector', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
```


## Concepts Used
- **OpenCV:** For image processing, video capture, and DNN-based age/gender prediction
- **FER library:** For robust emotion detection using deep learning
- **TensorFlow:** Backend for emotion recognition
- **Automatic model download:** Script fetches required models if missing
- **Real-time video processing:** Using webcam frames in a loop
- **Image manipulation:** Drawing rectangles, overlaying text, displaying results


## Troubleshooting
- If you get an error about `cv2.imshow` not being implemented, ensure you are not using a headless environment and have the full `opencv-python` package installed (not `opencv-python-headless`).
- Make sure your webcam is not being used by another application.
- If you encounter permission errors, try running your terminal as administrator.
- If model downloads fail, check your internet connection and try again.

---


---

**For any issues, open an issue on the GitHub repository.**
