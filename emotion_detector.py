import cv2
import threading
import time
import numpy as np
import argparse
from model_loader import ModelManager
from tracker import FaceTracker
from visualizer import Visualizer
from utils import set_max_resolution, enhance_image

class EmotionDetectorApp:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_thread = False
        self.model_manager = ModelManager()
        self.face_tracker = None
        self.visualizer = Visualizer()
        self.cap = None

    def load_resources(self):
        print('Loading models...')
        self.model_manager.load_models()
        self.face_tracker = FaceTracker(self.model_manager)

    def detection_loop(self):
        while not self.stop_thread:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            
            if frame is not None:
                # Detect faces using OpenCV DNN (ResNet SSD)
                if self.model_manager.face_net is None:
                    continue
                    
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.model_manager.face_net.setInput(blob)
                detections = self.model_manager.face_net.forward()
                
                detected_boxes = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5: # Confidence threshold
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype('int')
                        
                        # Clamp to frame
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)
                        
                        if endX > startX and endY > startY:
                            detected_boxes.append([startX, startY, endX-startX, endY-startY])
                
                # Update tracker with new detections
                if self.face_tracker:
                    self.face_tracker.update(detected_boxes, frame)
                
            time.sleep(0.01) # Small sleep to prevent CPU hogging

    def run(self):
        parser = argparse.ArgumentParser(description='Emotion, Age, and Gender Detection')
        args = parser.parse_args()

        self.load_resources()

        print('Starting camera...')
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print('Error: Could not open video stream.')
            return

        set_max_resolution(self.cap)
        
        # Start detection thread
        thread = threading.Thread(target=self.detection_loop, daemon=True)
        thread.start()

        print('Press ''q'' to quit.')

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print('Error: Failed to capture frame.')
                    break
                
                # Enhance for display
                display_frame = enhance_image(frame)
                
                with self.lock:
                    self.latest_frame = frame.copy()
                    # Get results from tracker
                    results = self.face_tracker.get_results() if self.face_tracker else []

                # Draw results
                display_frame = self.visualizer.draw_results(display_frame, results)

                cv2.imshow('Emotion, Age, Gender Detector', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_thread = True
            thread.join()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    app = EmotionDetectorApp()
    app.run()
