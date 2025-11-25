import os
import sys
import urllib.request
import cv2
from fer import FER

def resource_path(*relative_parts):
    """
    Resolve a path inside the bundled app when packaged with PyInstaller.
    Falls back to the local project path when running from source.
    """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base_path, *relative_parts)

def download_models():
    """
    Checks for the age, gender, and face detection models and downloads them if they are missing.
    """
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.abspath('.')
    models_dir = os.path.join(exe_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # URLs for the models
    age_proto_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/age_deploy.prototxt"
    age_model_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/age_net.caffemodel"
    gender_proto_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/gender_deploy.prototxt"
    gender_model_url = "https://github.com/pchunduri6/ComputerVision-ProjectIdeas/raw/master/Age_Gender_Detection/gender_net.caffemodel"
    
    # OpenCV Face Detector (ResNet SSD)
    face_proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    face_model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    # File paths
    bundled_age_proto = resource_path("models", "age_deploy.prototxt")
    bundled_age_model = resource_path("models", "age_net.caffemodel")
    bundled_gender_proto = resource_path("models", "gender_deploy.prototxt")
    bundled_gender_model = resource_path("models", "gender_net.caffemodel")
    
    age_proto_path = bundled_age_proto if os.path.exists(bundled_age_proto) else os.path.join(models_dir, "age_deploy.prototxt")
    age_model_path = bundled_age_model if os.path.exists(bundled_age_model) else os.path.join(models_dir, "age_net.caffemodel")
    gender_proto_path = bundled_gender_proto if os.path.exists(bundled_gender_proto) else os.path.join(models_dir, "gender_deploy.prototxt")
    gender_model_path = bundled_gender_model if os.path.exists(bundled_gender_model) else os.path.join(models_dir, "gender_net.caffemodel")
    
    face_proto_path = os.path.join(models_dir, "deploy.prototxt")
    face_model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")

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
    if not os.path.exists(face_proto_path):
        print("Downloading face detector prototxt...")
        urllib.request.urlretrieve(face_proto_url, face_proto_path)
    if not os.path.exists(face_model_path):
        print("Downloading face detector model...")
        urllib.request.urlretrieve(face_model_url, face_model_path)
    
    print("All models are ready.")
    return age_proto_path, age_model_path, gender_proto_path, gender_model_path, face_proto_path, face_model_path

class ModelManager:
    def __init__(self):
        self.age_net = None
        self.gender_net = None
        self.face_net = None
        self.emotion_detector = None
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['Male', 'Female']

    def load_models(self):
        age_proto, age_model, gender_proto, gender_model, face_proto, face_model = download_models()
        
        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        self.face_net = cv2.dnn.readNet(face_model, face_proto)
        # Initialize FER without internal detector for speed
        self.emotion_detector = FER(mtcnn=False)
