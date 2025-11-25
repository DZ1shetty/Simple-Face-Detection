import cv2
import time
import numpy as np

class Visualizer:
    def __init__(self):
        self.prev_time = 0
        self.fps = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'neutral': (0, 255, 0),    # Green
            'angry': (0, 0, 255),      # Red
            'disgust': (128, 0, 128),  # Purple
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 165, 0), # Orange
            'gesture': (255, 255, 0)   # Cyan/Yellowish
        }

    def _draw_rounded_rect(self, img, pt1, pt2, color, thickness=1, r=15):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top left
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def _draw_tech_corners(self, img, x, y, w, h, color, length=20, thickness=2):
        # Top Left
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        # Top Right
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        # Bottom Left
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        # Bottom Right
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def draw_results(self, frame, results):
        # Calculate FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time

        # Draw FPS with a nice background
        cv2.rectangle(frame, (10, 10), (140, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 40), self.font, 0.8, (0, 255, 0), 2)

        for res in results:
            x, y, w, h = res['box']
            label = res['label']
            confidence = res.get('confidence', 0.0)
            emotion = res.get('emotion', 'neutral')
            
            color = self.colors.get(emotion, (0, 255, 0))
            
            # Draw Tech-style Corners instead of full box for cleaner look
            self._draw_tech_corners(frame, x, y, w, h, color, length=30, thickness=2)
            
            # Draw Label with semi-transparent background
            (text_width, text_height), baseline = cv2.getTextSize(label, self.font, 0.6, 1)
            
            # Label Position (Top of box)
            label_bg_pt1 = (x, y - 25)
            label_bg_pt2 = (x + text_width + 10, y)
            
            # Draw background
            overlay = frame.copy()
            cv2.rectangle(overlay, label_bg_pt1, label_bg_pt2, color, -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw Text
            text_color = (0, 0, 0) if emotion in ['happy', 'surprise', 'neutral'] else (255, 255, 255)
            cv2.putText(frame, label, (x + 5, y - 8), self.font, 0.6, text_color, 1, cv2.LINE_AA)

            # Draw Slim Confidence Bar at bottom
            bar_bg_pt1 = (x, y + h + 5)
            bar_bg_pt2 = (x + w, y + h + 10)
            cv2.rectangle(frame, bar_bg_pt1, bar_bg_pt2, (50, 50, 50), -1)
            
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 10), color, -1)
            
        return frame
    
    def draw_gestures(self, frame, gesture_results):
        for res in gesture_results:
            x, y, w, h = res['box']
            gesture = res['gesture']
            
            # Use a distinct color for gestures
            color = self.colors['gesture']
            
            # Draw Tech-style Corners
            self._draw_tech_corners(frame, x, y, w, h, color, length=20, thickness=2)
            
            # Label
            label = f"Gesture: {gesture}"
            (text_width, text_height), baseline = cv2.getTextSize(label, self.font, 0.6, 1)
            
            # Label Position (Bottom of box to avoid conflict with face labels)
            label_bg_pt1 = (x, y + h + 15)
            label_bg_pt2 = (x + text_width + 10, y + h + 40)
            
            # Draw background
            overlay = frame.copy()
            cv2.rectangle(overlay, label_bg_pt1, label_bg_pt2, color, -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw Text
            cv2.putText(frame, label, (x + 5, y + h + 32), self.font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
        return frame

    def draw_button(self, frame, text, position, size, active=False):
        x, y = position
        w, h = size
        
        # Colors
        bg_color = (0, 255, 0) if active else (50, 50, 50)
        text_color = (0, 0, 0) if active else (255, 255, 255)
        border_color = (0, 255, 0)
        
        # Draw Button Background
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
        
        # Draw Tech Border
        self._draw_tech_corners(frame, x, y, w, h, border_color, length=10, thickness=2)
        
        # Draw Text
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, 0.7, 2)
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), self.font, 0.7, text_color, 2)
        
        return frame
