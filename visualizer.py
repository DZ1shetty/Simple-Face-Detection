import cv2
import time

class Visualizer:
    def __init__(self):
        self.prev_time = 0
        self.fps = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_results(self, frame, results):
        # Calculate FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time

        # Draw FPS
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 40), self.font, 1, (0, 255, 0), 2)

        for res in results:
            x, y, w, h = res['box']
            label = res['label']
            confidence = res.get('confidence', 0.0)
            emotion = res.get('emotion', 'neutral')
            
            # Color based on emotion (simple mapping)
            color = (0, 255, 0) # Default Green
            if emotion in ['angry', 'disgust', 'fear']: color = (0, 0, 255) # Red
            elif emotion in ['happy', 'surprise']: color = (0, 255, 255) # Yellow
            elif emotion == 'sad': color = (255, 0, 0) # Blue
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw Label Background
            (text_width, text_height), baseline = cv2.getTextSize(label, self.font, 0.8, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            
            # Draw Label Text
            text_color = (0, 0, 0) if emotion in ['happy', 'surprise'] else (255, 255, 255)
            cv2.putText(frame, label, (x, y - 5), self.font, 0.8, text_color, 2)

            # Draw Confidence Bar
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y + h + 5), (x + w, y + h + 15), (50, 50, 50), -1) # Background
            cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1) # Foreground
            
        return frame
