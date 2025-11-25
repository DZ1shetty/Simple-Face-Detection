import cv2
import threading
import time
import numpy as np
import argparse
from model_loader import ModelManager
from tracker import FaceTracker
from visualizer import Visualizer
from gesture_detector import GestureDetector
from utils import set_max_resolution, enhance_image

class EmotionDetectorApp:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_thread = False
        self.model_manager = ModelManager()
        self.face_tracker = None
        self.gesture_detector = GestureDetector()
        self.latest_gesture_results = []
        self.latest_mask = None # Store mask for debugging
        self.visualizer = Visualizer()
        self.cap = None
        
        # Mode: 'Face' or 'Gesture'
        self.mode = 'Face'
        self.btn_rect = (160, 10, 200, 40) # x, y, w, h

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = self.btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                # Toggle Mode
                if self.mode == 'Face':
                    self.mode = 'Gesture'
                else:
                    self.mode = 'Face'
                print(f"Switched to {self.mode} Mode")

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
                    if confidence > 0.3: # Lowered Confidence threshold for better detection
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
                    # OPTIMIZATION: Only run full face analysis (Age/Gender/Emotion) if in Face Mode
                    run_analysis = (self.mode == 'Face')
                    self.face_tracker.update(detected_boxes, frame, run_analysis=run_analysis)
                
                # Detect Gestures
                # OPTIMIZATION: Only run gesture detection if in Gesture Mode
                mask = None
                if self.mode == 'Gesture':
                    gestures, mask = self.gesture_detector.detect_gesture(frame, faces=detected_boxes)
                else:
                    gestures = []
                    
                with self.lock:
                    self.latest_gesture_results = gestures
                    self.latest_mask = mask
                
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
        
        window_name = 'Emotion, Age, Gender Detector'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

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
                    gesture_results = self.latest_gesture_results
                    mask_debug = self.latest_mask

                # Draw Mode Button
                btn_text = f"Mode: {self.mode}"
                display_frame = self.visualizer.draw_button(display_frame, btn_text, 
                                                          (self.btn_rect[0], self.btn_rect[1]), 
                                                          (self.btn_rect[2], self.btn_rect[3]), 
                                                          active=(self.mode == 'Gesture'))

                # Draw results based on mode
                if self.mode == 'Face':
                    display_frame = self.visualizer.draw_results(display_frame, results)
                    # Close mask window if switching back
                    try: cv2.destroyWindow('Gesture Mask')
                    except: pass
                elif self.mode == 'Gesture':
                    display_frame = self.visualizer.draw_gestures(display_frame, gesture_results)
                    
                    # Show Debug Mask
                    if mask_debug is not None:
                        # Resize for corner display
                        small_mask = cv2.resize(mask_debug, (200, 150))
                        small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
                        cv2.putText(small_mask, "Debug Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Overlay on bottom right
                        h, w = display_frame.shape[:2]
                        display_frame[h-160:h-10, w-210:w-10] = small_mask

                cv2.imshow(window_name, display_frame)

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
