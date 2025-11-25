import cv2
import numpy as np
import threading

class FaceTracker:
    def __init__(self, model_manager, max_missing=10):
        self.next_id = 0
        self.tracks = {}  # id -> {'box': [x,y,w,h], 'missing': 0, 'age_buffer': [], 'gender_buffer': [], 'emotion_buffer': []}
        self.max_missing = max_missing
        self.models = model_manager
        self.lock = threading.Lock()

    def update(self, detections, frame, run_analysis=True):
        with self.lock:
            # detections: list of [x, y, w, h]
            
            # Match detections to existing tracks using IoU
            matched_track_ids = set()
            matched_detection_indices = set()

            if self.tracks and detections:
                track_ids = list(self.tracks.keys())
                track_boxes = [self.tracks[tid]['box'] for tid in track_ids]
                
                # Simple distance matching (centroid)
                for idx, det_box in enumerate(detections):
                    dx, dy, dw, dh = det_box
                    
                    best_iou = 0
                    best_tid = -1
                    
                    for i, t_box in enumerate(track_boxes):
                        tid = track_ids[i]
                        tx, ty, tw, th = t_box
                        
                        # Calculate IoU
                        xA = max(dx, tx)
                        yA = max(dy, ty)
                        xB = min(dx+dw, tx+tw)
                        yB = min(dy+dh, ty+th)
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        boxAArea = dw * dh
                        boxBArea = tw * th
                        iou = interArea / float(boxAArea + boxBArea - interArea)
                        
                        if iou > 0.3 and iou > best_iou: # Threshold for matching
                            best_iou = iou
                            best_tid = tid
                    
                    if best_tid != -1:
                        matched_track_ids.add(best_tid)
                        matched_detection_indices.add(idx)
                        # Update track
                        self.tracks[best_tid]['box'] = det_box
                        self.tracks[best_tid]['missing'] = 0
                        
                        # Run analysis on this face
                        if run_analysis:
                            self.analyze_face(best_tid, frame, det_box)

            # Create new tracks for unmatched detections
            for idx, det_box in enumerate(detections):
                if idx not in matched_detection_indices:
                    self.tracks[self.next_id] = {
                        'box': det_box,
                        'missing': 0,
                        'age_buffer': [],
                        'gender_buffer': [],
                        'emotion_buffer': []
                    }
                    if run_analysis:
                        self.analyze_face(self.next_id, frame, det_box)
                    self.next_id += 1

            # Remove missing tracks
            for tid in list(self.tracks.keys()):
                if tid not in matched_track_ids:
                    self.tracks[tid]['missing'] += 1
                    if self.tracks[tid]['missing'] > self.max_missing:
                        del self.tracks[tid]

    def analyze_face(self, tid, frame, box):
        x, y, w, h = box
        
        # Add padding for better model accuracy (context is key)
        h_img, w_img = frame.shape[:2]
        padding_ratio = 0.2 # 20% padding
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        
        px = max(0, x - pad_w)
        py = max(0, y - pad_h)
        pw = min(w_img - px, w + 2*pad_w)
        ph = min(h_img - py, h + 2*pad_h)
        
        face_roi = frame[py:py+ph, px:px+pw]
        if face_roi.size == 0: return

        try:
            # Age/Gender (Caffe models expect 227x227)
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), self.models.MODEL_MEAN_VALUES, swapRB=False)
            self.models.gender_net.setInput(blob)
            gender_preds = self.models.gender_net.forward()
            gender = self.models.GENDER_LIST[gender_preds[0].argmax()]
            
            self.models.age_net.setInput(blob)
            age_preds = self.models.age_net.forward()
            age_idx = age_preds[0].argmax()
            
            # Emotion (FER library model expects 64x64 grayscale)
            # We bypass the internal detector for speed and accuracy (using our robust SSD crop)
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (64, 64))
            gray_face = gray_face.astype('float32') / 255.0 # Normalize
            gray_face = np.expand_dims(gray_face, axis=0) # (1, 64, 64)
            gray_face = np.expand_dims(gray_face, axis=-1) # (1, 64, 64, 1)
            
            emotion_scores = self.models.emotion_detector._classify_emotions(gray_face)
            # emotion_scores is [[p0, p1, ...]]
            
            # Map scores to labels
            labels = self.models.emotion_detector._get_labels()
            # labels is {0: 'angry', ...}
            
            # Find max score
            max_idx = np.argmax(emotion_scores[0])
            top_emotion = labels[max_idx]
            confidence = emotion_scores[0][max_idx]

            # Update buffers
            self.tracks[tid]['gender_buffer'].append(gender)
            self.tracks[tid]['age_buffer'].append(age_idx) # Store index for averaging
            self.tracks[tid]['emotion_buffer'].append((top_emotion, confidence))
            
            # Keep buffers small
            MAX_BUF = 10
            if len(self.tracks[tid]['gender_buffer']) > MAX_BUF: self.tracks[tid]['gender_buffer'].pop(0)
            if len(self.tracks[tid]['age_buffer']) > MAX_BUF: self.tracks[tid]['age_buffer'].pop(0)
            if len(self.tracks[tid]['emotion_buffer']) > MAX_BUF: self.tracks[tid]['emotion_buffer'].pop(0)
            
        except Exception as e:
            print(f"Analysis error: {e}")

    def get_results(self):
        with self.lock:
            results = []
            for tid, data in self.tracks.items():
                # Compute stable results
                if not data['age_buffer']: continue
                
                # Gender: Mode
                g_counts = {}
                for g in data['gender_buffer']: g_counts[g] = g_counts.get(g, 0) + 1
                if not g_counts: continue
                stable_gender = max(g_counts, key=lambda k: g_counts[k])
                
                # Age: Average index -> lookup
                avg_age_idx = int(sum(data['age_buffer']) / len(data['age_buffer']))
                stable_age = self.models.AGE_LIST[avg_age_idx]
                
                # Emotion: Mode of labels, but also get avg confidence
                e_counts = {}
                total_conf = 0
                for e, conf in data['emotion_buffer']: 
                    e_counts[e] = e_counts.get(e, 0) + 1
                    total_conf += conf
                if not e_counts: continue
                stable_emotion = max(e_counts, key=lambda k: e_counts[k])
                avg_conf = total_conf / len(data['emotion_buffer'])
                
                results.append({
                    'box': data['box'],
                    'label': f"{stable_gender}, {stable_age}, {stable_emotion.capitalize()}",
                    'confidence': avg_conf,
                    'emotion': stable_emotion
                })
            return results
