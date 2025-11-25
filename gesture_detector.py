import cv2
import numpy as np
from collections import deque, Counter

class GestureDetector:
    def __init__(self):
        # YCrCb range for skin color - Optimized for robustness
        # Tighter range to exclude background noise (fridge/walls)
        self.min_YCrCb = np.array([0, 133, 77], np.uint8)
        self.max_YCrCb = np.array([255, 173, 127], np.uint8)
        
        # Kernel for morphological operations
        self.kernel = np.ones((3, 3), np.uint8)
        
        # History buffer for smoothing: Stores list of detected gestures per frame
        self.history = deque(maxlen=5) 
        
        # Tracking State
        self.prev_center = None
        self.consecutive_frames = 0
        self.prev_box = None # For box smoothing

    def detect_gesture(self, frame, faces=[]):
        '''
        Detects hand gestures with high precision.
        Args:
            frame: The video frame.
            faces: List of face bounding boxes [x, y, w, h] to ignore.
        Returns:
            results: List of dictionaries containing gesture info.
            mask: The binary skin mask (for debugging).
            feedback: String message suggesting where to move hand if noise is detected.
        '''
        results = []
        feedback = None
        h_img, w_img = frame.shape[:2]
        total_pixels = h_img * w_img
        
        # 1. Skin Color Segmentation
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.min_YCrCb, self.max_YCrCb)
        
        # 2. Mask out Faces & Torso (CRITICAL for shirtless/body noise)
        for (fx, fy, fw, fh) in faces:
            # A. Mask the Face (with padding)
            pad = 20 
            cv2.rectangle(mask, (max(0, fx-pad), max(0, fy-pad)), (min(w_img, fx+fw+pad), min(h_img, fy+fh+pad)), 0, -1)
            
            # B. Mask the Torso (Everything below the chin)
            torso_y_start = fy + fh + 10 
            if torso_y_start < h_img:
                cv2.rectangle(mask, (0, torso_y_start), (w_img, h_img), 0, -1)

        # 3. Noise removal
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        current_frame_gestures = []
        best_candidate = None

        for i in range(min(2, len(contours))):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            
            # STRICT Thresholds
            # Min Area: Ignore small noise
            if area < 3000: continue
            # Max Area: Ignore huge background objects (walls, fridge) - Max 25% of screen
            if area > (total_pixels * 0.25): continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Edge Filter - Relaxed
            if x <= 2 or y <= 2 or (x+w) >= w_img-2 or (y+h) >= h_img-2:
                pass 

            # Aspect Ratio Filter
            if h == 0: continue
            aspect_ratio = float(w)/h
            if aspect_ratio < 0.2 or aspect_ratio > 4.0: 
                continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area)/hull_area
            
            # GEOMETRIC FILTER: Distinguish Fist from Box/Fridge
            # A fridge is a perfect rectangle (approx 4 corners). A fist is organic (>4 corners).
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            gesture_name = 'Unknown'
            is_object = False

            # If it looks like a box (4-6 corners) and is very solid, it's likely furniture
            if len(approx) <= 6 and solidity > 0.90:
                is_object = True
                gesture_name = 'Object'
            
            # EXTENT FILTER: Box vs Hand
            # Box fills its bounding rect almost perfectly (extent ~ 1.0)
            # Hand has wrist, fingers, gaps (extent < 0.8)
            rect_area = w * h
            extent = float(area) / rect_area
            if extent > 0.95: # Too perfect rectangle
                is_object = True
                gesture_name = 'Object'

            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            
            try:
                if not is_object:
                    defects = cv2.convexityDefects(cnt, hull_indices)
                    
                    if defects is not None:
                        count_defects = 0
                        
                        for j in range(defects.shape[0]):
                            s, e, f, d = defects[j, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            
                            # Triangle sides
                            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            
                            if b * c == 0: continue

                            cosine_angle = (b**2 + c**2 - a**2) / (2*b*c)
                            cosine_angle = max(-1.0, min(1.0, cosine_angle))
                            
                            angle = np.arccos(cosine_angle) * 57
                            
                            # Strict Angle and Depth
                            if angle <= 90 and d > 8000: 
                                count_defects += 1
                                
                        # Classification Logic
                        if count_defects == 0:
                            # 1 Finger: Tall (low aspect ratio) and not fully filling the box (low extent)
                            # Relaxed solidity/extent check to account for arm inclusion
                            if aspect_ratio < 0.6 and extent < 0.85: 
                                 gesture_name = '1 Finger'
                            elif solidity > 0.85: # Slightly relaxed for organic fists
                                gesture_name = 'Fist (0)'
                            else:
                                 gesture_name = 'Object' # Ambiguous -> Object
                        elif count_defects == 1:
                            gesture_name = '2 Fingers'
                        elif count_defects == 2:
                            gesture_name = '3 Fingers'
                        elif count_defects == 3:
                            gesture_name = '4 Fingers'
                        elif count_defects == 4:
                            gesture_name = '5 Fingers'
                        else:
                            gesture_name = 'Object' # Too many defects -> Object
                
                # Found a valid candidate (Hand or Object)
                best_candidate = {
                    'box': [x, y, w, h],
                    'gesture': gesture_name,
                    'contour': cnt,
                    'hull': hull,
                    'center': (x + w//2, y + h//2)
                }
                break # Only take the best one
            except Exception as e:
                print(f'Gesture error: {e}')

        # TRACKING & SMOOTHING LOGIC
        if best_candidate:
            # Check if it's close to previous detection
            if self.prev_center:
                dist = np.sqrt((best_candidate['center'][0] - self.prev_center[0])**2 + 
                               (best_candidate['center'][1] - self.prev_center[1])**2)
                if dist < 100: # Reasonable movement threshold
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 1 # Reset if jumped too far
            else:
                self.consecutive_frames = 1
            
            self.prev_center = best_candidate['center']
            
            # Only show if persistent for 3 frames (Removes flickering noise)
            if self.consecutive_frames >= 3:
                # Smooth Box
                if self.prev_box:
                    alpha = 0.6 # Smoothing factor
                    bx = int(alpha * self.prev_box[0] + (1-alpha) * best_candidate['box'][0])
                    by = int(alpha * self.prev_box[1] + (1-alpha) * best_candidate['box'][1])
                    bw = int(alpha * self.prev_box[2] + (1-alpha) * best_candidate['box'][2])
                    bh = int(alpha * self.prev_box[3] + (1-alpha) * best_candidate['box'][3])
                    best_candidate['box'] = [bx, by, bw, bh]
                
                self.prev_box = best_candidate['box']
                
                # Smooth Label
                self.history.append(best_candidate['gesture'])
                most_common = Counter(self.history).most_common(1)[0][0]
                best_candidate['gesture'] = most_common
                
                results.append(best_candidate)
        else:
            # No hand found, check for noise to give feedback
            # Split mask into 4 quadrants to find the cleanest area
            cx, cy = w_img // 2, h_img // 2
            
            # Define quadrants: (ROI, Name)
            quadrants = [
                (mask[0:cy, 0:cx], "Top-Left"),
                (mask[0:cy, cx:w_img], "Top-Right"),
                (mask[cy:h_img, 0:cx], "Bottom-Left"),
                (mask[cy:h_img, cx:w_img], "Bottom-Right")
            ]
            
            # Count noise (white pixels) in each quadrant
            noise_counts = []
            for q_img, name in quadrants:
                count = cv2.countNonZero(q_img)
                noise_counts.append((count, name))
            
            # Sort by noise (ascending) -> [0] is cleanest, [-1] is noisiest
            noise_counts.sort(key=lambda x: x[0])
            
            best_zone_count, best_zone_name = noise_counts[0]
            worst_zone_count, _ = noise_counts[-1]
            
            # Threshold: If worst zone has > 5% white pixels, it's noisy.
            # We only warn if there IS noise.
            quadrant_area = (w_img // 2) * (h_img // 2)
            if worst_zone_count > (quadrant_area * 0.05):
                feedback = f"Background Noisy! Try {best_zone_name}"

            # Reset Tracking when no hand is found
            self.consecutive_frames = 0
            self.prev_center = None
            self.prev_box = None
            self.history.clear()

        return results, mask, feedback
