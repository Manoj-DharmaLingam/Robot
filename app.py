import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import deque
import time

# ================= CONFIGURATION FOR 1000+ PEOPLE =================
# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Model Settings
MODEL_NAME = 'buffalo_l'
CTX_ID = 0

# Detection Settings - EXTREME CROWD MODE
DET_SIZE = (640, 640)
DET_THRESH = 0.58

# Recognition Settings
RECOGNITION_THRESHOLD = 0.36
CONFIDENCE_DECAY = 0.95  # Confidence decays each frame without verification

# Performance Settings - CONTINUOUS TARGET VALIDATION
MAX_FACES_TO_DETECT = 200
MAX_FACES_TO_PROCESS = 20  # Increased for better target tracking
ENABLE_CONTINUOUS_VALIDATION = True  # Always re-verify target

# Tracking strategy
TRACKING_MODE = "ADAPTIVE"  # "ADAPTIVE", "CONTINUOUS", or "HYBRID"
VALIDATION_INTERVAL = 2  # Re-verify target every N frames
LOCAL_SEARCH_RADIUS = 300  # Search radius around target
EXPAND_RADIUS_WHEN_MOVING = True  # Expand search for moving targets

# Zone scanning for initial detection
GRID_ZONES = 6
ZONE_SWITCH_INTERVAL = 2
USE_GRID_UNTIL_FOUND = True  # Switch to local search after finding target

# Movement detection
DETECT_MOVEMENT = True
MOVEMENT_THRESHOLD = 20  # pixels per frame
VELOCITY_SMOOTHING = 0.7  # Smooth velocity estimation

# Quality thresholds
MIN_FACE_SIZE = 50
MIN_DETECTION_SCORE = 0.65
MIN_MATCH_CONFIDENCE = 0.38
# =================================================================

dll_path = r"C:\Users\D.S.Manoj\Desktop\robot 1\venv\Lib\site-packages\onnxruntime\capi"
os.environ["PATH"] += os.pathsep + dll_path
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(dll_path)


class AdaptiveTargetTracker:
    """Advanced tracker with continuous target validation"""
    def __init__(self):
        self.target_bbox = None
        self.target_center = None
        self.velocity = np.array([0.0, 0.0])
        self.confidence = 0.0
        self.frames_tracked = 0
        self.last_verified_frame = 0
        self.is_moving = False
        self.movement_history = deque(maxlen=5)
        
    def init(self, bbox, confidence=1.0):
        self.target_bbox = bbox.copy()
        self.target_center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        self.confidence = confidence
        self.frames_tracked = 0
        self.velocity = np.array([0.0, 0.0])
        self.movement_history.clear()
        
    def update_with_detection(self, new_bbox, similarity, current_frame):
        """Update with verified detection"""
        if new_bbox is None:
            return False
        
        # Calculate new center
        new_center = np.array([(new_bbox[0] + new_bbox[2])/2, (new_bbox[1] + new_bbox[3])/2])
        
        # Update velocity
        if self.target_center is not None:
            displacement = new_center - self.target_center
            self.velocity = displacement * (1 - VELOCITY_SMOOTHING) + self.velocity * VELOCITY_SMOOTHING
            
            # Track movement
            movement_magnitude = np.linalg.norm(displacement)
            self.movement_history.append(movement_magnitude)
            self.is_moving = movement_magnitude > MOVEMENT_THRESHOLD
        
        # Update state
        self.target_bbox = new_bbox
        self.target_center = new_center
        self.confidence = similarity
        self.last_verified_frame = current_frame
        self.frames_tracked += 1
        
        return True
    
    def predict_position(self):
        """Predict next position based on velocity"""
        if self.target_bbox is None:
            return None
        
        # Predict center
        predicted_center = self.target_center + self.velocity
        
        # Predict bbox (maintaining size)
        w = self.target_bbox[2] - self.target_bbox[0]
        h = self.target_bbox[3] - self.target_bbox[1]
        
        predicted_bbox = np.array([
            predicted_center[0] - w/2,
            predicted_center[1] - h/2,
            predicted_center[0] + w/2,
            predicted_center[1] + h/2
        ])
        
        return predicted_bbox
    
    def get_search_region(self, frame_shape):
        """Get adaptive search region based on movement"""
        if self.target_bbox is None:
            return None
        
        # Use predicted position if moving
        if self.is_moving and EXPAND_RADIUS_WHEN_MOVING:
            center = self.target_center + self.velocity
            radius = LOCAL_SEARCH_RADIUS * 1.5  # Expand for moving targets
        else:
            center = self.target_center
            radius = LOCAL_SEARCH_RADIUS
        
        # Calculate region
        h, w = frame_shape[:2]
        x1 = int(max(0, center[0] - radius))
        y1 = int(max(0, center[1] - radius))
        x2 = int(min(w, center[0] + radius))
        y2 = int(min(h, center[1] + radius))
        
        return (x1, y1, x2, y2)
    
    def decay_confidence(self):
        """Decay confidence when not verified"""
        self.confidence *= CONFIDENCE_DECAY
        return self.confidence
    
    def should_revalidate(self, current_frame):
        """Check if target needs revalidation"""
        frames_since_verification = current_frame - self.last_verified_frame
        return frames_since_verification >= VALIDATION_INTERVAL
    
    def get_average_movement(self):
        """Get average movement speed"""
        if len(self.movement_history) == 0:
            return 0
        return np.mean(self.movement_history)


class ContinuousTrackingSystem:
    def __init__(self):
        print("🤖 Initializing Continuous Target Tracking System")
        print("🎯 Designed for moving targets in 1000+ crowds")
        
        # Initialize FaceAnalysis
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
        
        # Target storage
        self.target_embeddings = []
        self.avg_target_embedding = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Tracking system
        self.tracker = AdaptiveTargetTracker()
        self.target_found = False
        self.target_id = None
        
        # Grid scanning for initial detection
        self.current_zone = 0
        self.zone_frame_count = 0
        self.using_grid_scan = True
        
        # Stats
        self.total_detections = 0
        self.validations_performed = 0
        self.false_positives_rejected = 0
        
        # Performance metrics
        self.detection_times = deque(maxlen=20)
        self.validation_times = deque(maxlen=20)
        
        print("✅ Continuous Tracking System Ready!")
    
    def load_target(self, image_path):
        """Load target with optimized embeddings"""
        if not os.path.exists(image_path):
            print("⚠️  Target image not found.")
            return False
        
        print(f"🎯 Loading target from {image_path}...")
        img_target = cv2.imread(image_path)
        
        if img_target is None:
            print("❌ Could not read target image.")
            return False
        
        embeddings_list = []
        
        # Core variations
        faces = self.app.get(img_target)
        if len(faces) > 0:
            embeddings_list.append(faces[0].embedding)
        
        img_flipped = cv2.flip(img_target, 1)
        faces_flipped = self.app.get(img_flipped)
        if len(faces_flipped) > 0:
            embeddings_list.append(faces_flipped[0].embedding)
        
        for gamma in [0.7, 0.9, 1.1, 1.3]:
            adjusted = self.adjust_gamma(img_target, gamma)
            faces_adj = self.app.get(adjusted)
            if len(faces_adj) > 0:
                embeddings_list.append(faces_adj[0].embedding)
        
        for angle in [-15, -8, 8, 15]:
            rotated = self.rotate_image(img_target, angle)
            faces_rot = self.app.get(rotated)
            if len(faces_rot) > 0:
                embeddings_list.append(faces_rot[0].embedding)
        
        if len(embeddings_list) == 0:
            print("❌ No face found in target image.")
            return False
        
        self.target_embeddings = embeddings_list
        self.avg_target_embedding = np.mean(embeddings_list, axis=0)
        self.avg_target_embedding /= np.linalg.norm(self.avg_target_embedding)
        
        print(f"✅ Target locked! {len(embeddings_list)} embeddings | Continuous validation enabled")
        return True
    
    @staticmethod
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def compute_similarity(feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    def is_target_match(self, embedding):
        """Match against target embeddings"""
        if not self.target_embeddings:
            return False, 0.0
        
        # Quick check with average
        avg_sim = self.compute_similarity(self.avg_target_embedding, embedding)
        if avg_sim < 0.20:
            return False, 0.0
        
        # Detailed check
        similarities = [self.compute_similarity(ref_emb, embedding) 
                       for ref_emb in self.target_embeddings]
        max_sim = max(similarities)
        
        return max_sim > RECOGNITION_THRESHOLD, max_sim
    
    def get_zone_bounds(self, frame_shape):
        """Get current zone bounds"""
        h, w = frame_shape[:2]
        
        rows, cols = 2, 3  # 2x3 grid
        zone_h, zone_w = h // rows, w // cols
        
        row = self.current_zone // cols
        col = self.current_zone % cols
        
        x1 = col * zone_w
        y1 = row * zone_h
        x2 = x1 + zone_w
        y2 = y1 + zone_h
        
        # Overlap
        overlap = 60
        x1 = max(0, x1 - overlap)
        y1 = max(0, y1 - overlap)
        x2 = min(w, x2 + overlap)
        y2 = min(h, y2 + overlap)
        
        return (x1, y1, x2, y2)
    
    def detect_faces_adaptive(self, frame):
        """Adaptive detection: local search if tracking, zone scan if searching"""
        start_time = time.time()
        
        # If target is being tracked, search locally
        if self.target_found and self.tracker.target_bbox is not None:
            search_region = self.tracker.get_search_region(frame.shape)
            
            if search_region is not None:
                x1, y1, x2, y2 = search_region
                
                # Crop to region
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    cropped = frame[y1:y2, x1:x2]
                    faces = self.app.get(cropped)
                    
                    # Adjust coordinates
                    for face in faces:
                        face.bbox += np.array([x1, y1, x1, y1])
                    
                    self.detection_times.append(time.time() - start_time)
                    return faces, "LOCAL"
        
        # Otherwise use zone scanning
        x1, y1, x2, y2 = self.get_zone_bounds(frame.shape)
        zone_frame = frame[y1:y2, x1:x2]
        faces = self.app.get(zone_frame)
        
        # Adjust coordinates
        for face in faces:
            face.bbox += np.array([x1, y1, x1, y1])
        
        self.detection_times.append(time.time() - start_time)
        return faces, "ZONE"
    
    def filter_faces(self, faces, frame_shape):
        """Filter and prioritize faces"""
        if len(faces) == 0:
            return []
        
        # Size filter
        filtered = []
        for face in faces:
            box = face.bbox
            w, h = box[2] - box[0], box[3] - box[1]
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE and face.det_score >= MIN_DETECTION_SCORE:
                filtered.append(face)
        
        if len(filtered) == 0:
            return []
        
        # If tracking target, prioritize faces near predicted position
        if self.target_found and self.tracker.target_bbox is not None:
            predicted = self.tracker.predict_position()
            if predicted is not None:
                pred_center = np.array([(predicted[0] + predicted[2])/2, 
                                       (predicted[1] + predicted[3])/2])
                
                scored = []
                for face in filtered:
                    box = face.bbox
                    face_center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
                    distance = np.linalg.norm(face_center - pred_center)
                    
                    # Score: closer to prediction = higher priority
                    score = face.det_score * 100 / (distance + 1)
                    scored.append((score, face))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                return [face for _, face in scored[:MAX_FACES_TO_PROCESS]]
        
        # Otherwise prioritize by size and confidence
        h, w = frame_shape[:2]
        center_x, center_y = w / 2, h / 2
        
        scored = []
        for face in filtered:
            box = face.bbox
            size = (box[2] - box[0]) * (box[3] - box[1])
            cx, cy = (box[0] + box[2])/2, (box[1] + box[3])/2
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            score = size * face.det_score / (dist + 1)
            scored.append((score, face))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [face for _, face in scored[:MAX_FACES_TO_PROCESS]]
    
    def validate_target(self, faces):
        """Continuously validate tracked target"""
        start_time = time.time()
        self.validations_performed += 1
        
        if len(faces) == 0:
            self.validation_times.append(time.time() - start_time)
            return None, 0.0
        
        # Look for target in detected faces
        best_match = None
        best_similarity = 0.0
        
        for face in faces:
            is_match, similarity = self.is_target_match(face.embedding)
            
            if is_match and similarity > best_similarity:
                best_match = face
                best_similarity = similarity
        
        self.validation_times.append(time.time() - start_time)
        
        if best_match is not None:
            return best_match, best_similarity
        else:
            self.false_positives_rejected += len(faces)
            return None, 0.0
    
    def process_frame(self, frame):
        """Main processing with continuous validation"""
        self.frame_count += 1
        
        # Update zone scanning
        if self.using_grid_scan:
            self.zone_frame_count += 1
            if self.zone_frame_count >= ZONE_SWITCH_INTERVAL:
                self.current_zone = (self.current_zone + 1) % GRID_ZONES
                self.zone_frame_count = 0
        
        # Detect faces
        faces, detection_mode = self.detect_faces_adaptive(frame)
        self.total_detections = len(faces)
        
        # Filter faces
        faces = self.filter_faces(faces, frame.shape)
        
        # Validate/search for target
        target_face, similarity = self.validate_target(faces)
        
        if target_face is not None:
            # Target found/validated
            if not self.target_found:
                print(f"🎯 TARGET ACQUIRED! (Similarity: {similarity:.3f})")
                self.target_found = True
                self.using_grid_scan = False  # Switch to local search
            
            # Update tracker
            self.tracker.update_with_detection(target_face.bbox, similarity, self.frame_count)
            
            return target_face, similarity, detection_mode
        
        else:
            # Target not found in this frame
            if self.target_found:
                # Decay confidence
                confidence = self.tracker.decay_confidence()
                
                # If confidence too low, return to grid scanning
                if confidence < 0.3:
                    print(f"⚠️  Target lost (confidence: {confidence:.2f}). Resuming search...")
                    self.target_found = False
                    self.using_grid_scan = True
                    self.tracker.target_bbox = None
            
            return None, 0.0, detection_mode
    
    def draw_results(self, frame, target_face, similarity, detection_mode):
        """Draw tracking results"""
        
        # Draw search region visualization
        if self.target_found and self.tracker.target_bbox is not None:
            search_region = self.tracker.get_search_region(frame.shape)
            if search_region is not None:
                x1, y1, x2, y2 = search_region
                # Draw search area
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.putText(frame, "SEARCH AREA", (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Draw zone boundary if scanning
        if self.using_grid_scan:
            x1, y1, x2, y2 = self.get_zone_bounds(frame.shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, f"ZONE {self.current_zone + 1}/{GRID_ZONES}", 
                       (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Draw target
        if target_face is not None:
            box = target_face.bbox.astype(int)
            
            # Animated pulsing
            pulse = int(12 * (1 + np.sin(time.time() * 10)))
            cv2.rectangle(frame, 
                        (box[0]-pulse, box[1]-pulse), 
                        (box[2]+pulse, box[3]+pulse), 
                        (0, 255, 255), 2)
            
            # Main box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            
            # Movement indicator
            if self.tracker.is_moving:
                # Draw velocity vector
                if self.tracker.velocity is not None:
                    center = self.tracker.target_center.astype(int)
                    velocity_end = (center + self.tracker.velocity * 5).astype(int)
                    cv2.arrowedLine(frame, tuple(center), tuple(velocity_end), 
                                  (0, 255, 255), 2, tipLength=0.3)
            
            # Label
            status = "MOVING" if self.tracker.is_moving else "STATIONARY"
            label = f"TARGET {similarity:.2f} | {status}"
            
            label_bg = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                        (box[0], box[1] - label_bg[1] - 15), 
                        (box[0] + label_bg[0], box[1]), 
                        (0, 255, 0), -1)
            cv2.putText(frame, label, (box[0], box[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Landmarks
            if hasattr(target_face, 'kps') and target_face.kps is not None:
                for point in target_face.kps.astype(int):
                    cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
        
        # Status panel
        panel_h = 260
        cv2.rectangle(frame, (10, 90), (450, 90 + panel_h), (0, 0, 0), -1)
        
        if self.target_found:
            panel_color = (0, 255, 0)
            status_text = "🎯 TARGET LOCKED - CONTINUOUS TRACKING"
        else:
            panel_color = (0, 165, 255)
            status_text = "🔍 SEARCHING FOR TARGET..."
        
        cv2.rectangle(frame, (10, 90), (450, 90 + panel_h), panel_color, 2)
        
        y = 115
        cv2.putText(frame, status_text, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, panel_color, 2)
        
        y += 25
        cv2.putText(frame, f"Mode: {detection_mode} | Confidence: {self.tracker.confidence:.2f}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        cv2.putText(frame, f"Faces Detected: {self.total_detections}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        cv2.putText(frame, f"Validations: {self.validations_performed}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        avg_movement = self.tracker.get_average_movement()
        cv2.putText(frame, f"Avg Movement: {avg_movement:.1f} px/frame", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        velocity_mag = np.linalg.norm(self.tracker.velocity)
        cv2.putText(frame, f"Velocity: {velocity_mag:.1f} px/frame", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        avg_det = np.mean(self.detection_times) * 1000 if len(self.detection_times) > 0 else 0
        cv2.putText(frame, f"Detection: {avg_det:.0f}ms", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        avg_val = np.mean(self.validation_times) * 1000 if len(self.validation_times) > 0 else 0
        cv2.putText(frame, f"Validation: {avg_val:.0f}ms", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        frames_tracked = self.tracker.frames_tracked
        cv2.putText(frame, f"Tracked Frames: {frames_tracked}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def draw_ui(self, frame):
        fps = self.update_fps()
        
        fps_color = (0, 255, 0) if fps > 12 else (0, 255, 255) if fps > 8 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, fps_color, 2)
        
        mode_text = "CONTINUOUS TRACKING MODE"
        cv2.putText(frame, mode_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        image_name = input("📂 Target image: ").strip()

        if image_name:
            if not self.load_target(image_name):
                return
        else:
            print("⚠️ No target. Exiting.")
            return
        
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Could not open webcam.")
            return
        
        cv2.namedWindow("Continuous Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Continuous Tracking", 1280, 720)
        
        print("🎥 CONTINUOUS TRACKING MODE Active!")
        print("✅ Target will be re-validated every frame while moving")
        print("📊 Adaptive search: Zone scan → Local search → Continuous validation")
        print("Controls: q=quit | s=save | r=reload | t=reset")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process with continuous validation
            target_face, similarity, detection_mode = self.process_frame(frame)
            
            # Draw results
            frame_display = self.draw_results(frame.copy(), target_face, similarity, detection_mode)
            frame_display = self.draw_ui(frame_display)
            
            cv2.imshow('Continuous Tracking', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
            elif key == ord('r') and image_name:
                print("🔄 Reloading...")
                self.tracker = AdaptiveTargetTracker()
                self.target_found = False
                self.using_grid_scan = True
                self.load_target(image_name)
            elif key == ord('t'):
                print("🔄 Reset tracking")
                self.tracker = AdaptiveTargetTracker()
                self.target_found = False
                self.using_grid_scan = True
        
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        print(f"👋 Stopped. Avg FPS: {avg_fps:.1f}")
        print(f"📊 Stats: Validations: {self.validations_performed} | Tracked Frames: {self.tracker.frames_tracked}")


if __name__ == "__main__":
    tracker = ContinuousTrackingSystem()
    tracker.run()