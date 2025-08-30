# Emotion, Age, and Gender Detector

This project is an advanced real-time face analysis application using Python and OpenCV. It detects faces from your webcam feed and predicts age, gender, and emotion using pre-trained deep learning models.

## Features
- Real-time face detection using a deep learning model (SSD + Caffe)
- Age and gender prediction for each detected face
- Emotion recognition using a pre-trained ONNX model
- All predictions are displayed live on the webcam feed

## Project Structure
```
your-project-folder/
│
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   └── emotion-ferplus-8.onnx
│
└── emotion_detector.py
```

## Setup Instructions

### 1. Install Python
Make sure you have Python 3.6 or newer installed. You can download it from https://www.python.org/downloads/

### 2. Install Required Libraries
Open a terminal or command prompt in your project folder and run:

```bash
pip install opencv-python numpy
```

### 3. Download Pre-trained Models
Create a folder named `models` in your project directory. Download the following files and place them inside the `models` folder:

- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`
- `age_deploy.prototxt`
- `age_net.caffemodel`
- `emotion-ferplus-8.onnx`

You can find these files by searching their names online. They are standard models used in many OpenCV and deep learning tutorials.

### 4. Run the Application
In your terminal, run:

```bash
python emotion_detector.py
```

A window will open showing your webcam feed. Detected faces will be highlighted with a box and labeled with predicted age, gender, and emotion.

- Press `q` or close the window to exit.

## Notes
- Make sure your webcam is connected and accessible.
- The application uses the default webcam (camera index 0).
- For GUI features to work, do not use headless environments.
- All models must be present in the `models` folder for the script to run.

## Credits
- Face detection: OpenCV DNN module (SSD + Caffe)
- Age/Gender: Pre-trained Caffe models
- Emotion: FER+ ONNX model

---

Enjoy your advanced face analysis application!
