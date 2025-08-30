# Face Detector

This project is a simple and interactive real-time face detection application using OpenCV and Python. It utilizes your webcam to detect faces in live video, and provides options to blur detected faces, draw rectangles, and save snapshots.

## Features
- **Real-time face detection** using Haar Cascade classifier
- **Toggle between drawing rectangles and blurring faces** for privacy
- **Save snapshots** of the current video frame with detected faces
- **Performance optimizations**: processes every Nth frame for efficiency
- **User-friendly controls**:
  - Press `b` to toggle blur effect
  - Press `s` to save a snapshot
  - Press `q` to quit the application

## Requirements
- Python 3.6+
- OpenCV (`opencv-python`)

## How to Run
1. Install dependencies:
   ```bash
   pip install opencv-python
   ```
2. Run the script:
   ```bash
   python face_detector.py
   ```

## Use Cases
- Privacy masking for video streams
- Quick face detection demos
- Educational purposes for learning OpenCV

---

**Note:**
- Make sure your webcam is connected and accessible.
- The application uses the default webcam (camera index 0).
- For GUI features to work, do not use headless environments.
