# Emotion, Age, and Gender Detector

A robust, real-time application for detecting emotions, age, and gender from a webcam feed. Built with Python, OpenCV, and Deep Learning, this project features a modular architecture, stable tracking, and automatic model management.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Real-time Analysis**: Simultaneous detection of Emotion, Age, Gender, and Hand Gestures.
- **Hand Gesture Recognition**: Detects gestures like "1 Finger", "2 Fingers", "Fist", "Open Palm", etc. Includes robust logic to handle arm noise and distinguish between similar shapes.
- **Robust Face Tracking**: Uses OpenCV DNN (ResNet SSD) for detection and an IoU (Intersection over Union) tracker to maintain face identities across frames.
- **Stable Predictions**: Implements history buffering and voting/averaging mechanisms to prevent flickering results.
- **Context-Aware Accuracy**: Applies intelligent padding (20%) to face crops for better model inference context.
- **Modular Architecture**: Clean separation of concerns into Data, Logic, and UI components.
- **Automatic Setup**: Automatically downloads required Caffe and TensorFlow models on first run.
- **Performance Optimized**: Threaded detection loop and efficient visualization.

## Project Structure

The codebase has been refactored into a professional, modular structure:

- **`emotion_detector.py`**: The main entry point. Orchestrates the application, handles threading, and manages the main loop.
- **`gesture_detector.py`**: Implements hand gesture recognition using skin color segmentation and convexity defects.
- **`model_loader.py`**: Handles the automatic downloading and loading of Caffe (Age/Gender) and FER (Emotion) models.
- **`tracker.py`**: Contains the `FaceTracker` class. Implements IoU tracking logic, history buffers, and the core analysis pipeline.
- **`visualizer.py`**: Manages all UI drawing operations (bounding boxes, labels, FPS counter, confidence bars).
- **`utils.py`**: Helper functions for camera setup and image enhancement.
- **`run.bat`**: Simple batch script to launch the application in the virtual environment.

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- TensorFlow
- FER (`fer`)
- NumPy

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/DZ1shetty/Simple-Face-Detection.git
   cd Simple-Face-Detection
   ```

2. **Create a virtual environment (Recommended):**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   *If `requirements.txt` is missing, install manually:*
   ```bash
   pip install opencv-contrib-python tensorflow fer numpy
   ```

## Usage

1. **Connect your webcam.**

2. **Run the application:**

   **Using the batch script (Windows):**
   ```bash
   .\run.bat
   ```

   **Using Python directly:**
   ```bash
   python emotion_detector.py
   ```

3. **Controls:**
   - The application window will open showing the live feed.
   - Press **`q`** to quit the application.

## How It Works

1. **Initialization**: The `ModelManager` checks for model files. If missing, it downloads them from reliable sources.
2. **Detection**: A background thread runs the ResNet SSD face detector to find faces in the current frame.
3. **Tracking**: The `FaceTracker` matches new detections to existing faces using Intersection over Union (IoU). This assigns a stable ID to each face.
4. **Analysis**:
   - **Age/Gender**: Inferred using pre-trained Caffe models on the face crop.
   - **Emotion**: Inferred using the FER library on a normalized, padded face crop.
5. **Smoothing**: Results are stored in a history buffer. The displayed label is the "mode" (most frequent) of the last 10 frames, ensuring stability.
6. **Visualization**: The `Visualizer` draws the results, including a confidence bar for the dominant emotion.

## Troubleshooting

- **"Error: Could not open video stream"**: Ensure your webcam is connected and not being used by another app (like Zoom or Teams).
- **Slow Performance**: The application attempts to set the camera to a high resolution. If your PC is slow, you can modify `utils.py` to use a lower resolution (e.g., 640x480).
- **Model Download Failures**: If the app hangs at "Downloading...", check your internet connection. You can also manually place the `.prototxt` and `.caffemodel` files in the `models/` directory.

---

**For any issues, please open an issue on the GitHub repository.**
