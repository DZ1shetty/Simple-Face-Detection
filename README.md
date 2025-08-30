# Simple Face Detector

A real-time face detection application using Python and OpenCV. This project uses your webcam to detect faces in live video, with options to blur detected faces, draw rectangles, and save snapshots.

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
- Real-time face detection using Haar Cascade classifier
- Toggle between drawing rectangles and blurring faces for privacy
- Save snapshots of the current video frame
- Performance optimizations: processes every Nth frame for efficiency
- User-friendly controls:
  - Press `b` to toggle blur effect
  - Press `s` to save a snapshot
  - Press `q` to quit the application

## Requirements
- Python 3.6 or higher
- OpenCV (`opencv-python`)
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
   pip install opencv-python
   ```

## Usage
1. Make sure your webcam is connected and accessible.
2. Run the script:
   ```bash
   python face_detector.py
   ```
3. Use the following controls during execution:
   - `b`: Toggle blur effect on detected faces
   - `s`: Save a snapshot of the current frame
   - `q`: Quit the application

## Code Overview
The main logic is in `face_detector.py`:
- Loads OpenCV's Haar Cascade for face detection
- Captures video from your default webcam
- Detects faces in real-time
- Draws rectangles or blurs faces based on user input
- Allows saving snapshots with detected faces

### Example Code Snippet
```python
import cv2
import time

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Concepts Used
- **OpenCV:** For image processing, video capture, and face detection
- **Haar Cascade Classifier:** Pre-trained model for detecting faces
- **Real-time video processing:** Using webcam frames in a loop
- **Image manipulation:** Drawing rectangles, blurring regions, saving images
- **Keyboard event handling:** For interactive controls

## Troubleshooting
- If you get an error about `cv2.imshow` not being implemented, ensure you are not using a headless environment and have the full `opencv-python` package installed (not `opencv-python-headless`).
- Make sure your webcam is not being used by another application.
- If you encounter permission errors, try running your terminal as administrator.

---

**For any issues, open an issue on the GitHub repository.**
