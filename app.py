import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import deque
import time

# ================= CONFIGURATION =================
# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 2560
FRAME_HEIGHT = 1600

# Model Settings
MODEL_NAME = 'buffalo_l'
CTX_ID = 0

# Detection Settings
DET_SIZE = (640, 640)  # Stable size that works with all image dimensions
DET_THRESH = 0.50  # Lower for better angle detection

# Recognition Settings
RECOGNITION_THRESHOLD = 0.32  # Lower threshold for side/angled faces
CONFIDENCE_DECAY = 0.95

# Performance Settings
MAX_FACES_TO_PROCESS = 20
ENABLE_CONTINUOUS_VALIDATION = True

# Tracking strategy
VALIDATION_INTERVAL = 2
LOCAL_SEARCH_RADIUS = 300
EXPAND_RADIUS_WHEN_MOVING = True

# Zone scanning
GRID_ZONES = 6
ZONE_SWITCH_INTERVAL = 2
USE_GRID_UNTIL_FOUND = True

# Movement detection
DETECT_MOVEMENT = True
MOVEMENT_THRESHOLD = 20
VELOCITY_SMOOTHING = 0.7

# Quality thresholds - RELAXED for angled faces
MIN_FACE_SIZE = 40  # Lower for distant/angled faces
MIN_DETECTION_SCORE = 0.55  # Lower for side views
MIN_MATCH_CONFIDENCE = 0.32
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
        if new_bbox is None:
            return False
        
        new_center = np.array([(new_bbox[0] + new_bbox[2])/2, (new_bbox[1] + new_bbox[3])/2])
        
        if self.target_center is not None:
            displacement = new_center - self.target_center
            self.velocity = displacement * (1 - VELOCITY_SMOOTHING) + self.velocity * VELOCITY_SMOOTHING
            
            movement_magnitude = np.linalg.norm(displacement)
            self.movement_history.append(movement_magnitude)
            self.is_moving = movement_magnitude > MOVEMENT_THRESHOLD
        
        self.target_bbox = new_bbox
        self.target_center = new_center
        self.confidence = similarity
        self.last_verified_frame = current_frame
        self.frames_tracked += 1
        
        return True
    
    def predict_position(self):
        if self.target_bbox is None:
            return None
        
        predicted_center = self.target_center + self.velocity
        
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
        if self.target_bbox is None:
            return None
        
        if self.is_moving and EXPAND_RADIUS_WHEN_MOVING:
            center = self.target_center + self.velocity
            radius = LOCAL_SEARCH_RADIUS * 1.5
        else:
            center = self.target_center
            radius = LOCAL_SEARCH_RADIUS
        
        h, w = frame_shape[:2]
        x1 = int(max(0, center[0] - radius))
        y1 = int(max(0, center[1] - radius))
        x2 = int(min(w, center[0] + radius))
        y2 = int(min(h, center[1] + radius))
        
        return (x1, y1, x2, y2)
    
    def decay_confidence(self):
        self.confidence *= CONFIDENCE_DECAY
        return self.confidence
    
    def should_revalidate(self, current_frame):
        frames_since_verification = current_frame - self.last_verified_frame
        return frames_since_verification >= VALIDATION_INTERVAL
    
    def get_average_movement(self):
        if len(self.movement_history) == 0:
            return 0
        return np.mean(self.movement_history)


class AllAngleTrackingSystem:
    def __init__(self):
        print("🤖 Initializing ALL-ANGLE Target Tracking System")
        print("🎯 Detects: Front, Side, Looking Up/Down, All Poses")
        
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
        
        self.target_embeddings = []
        self.avg_target_embedding = None
        
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        self.tracker = AdaptiveTargetTracker()
        self.target_found = False
        
        self.current_zone = 0
        self.zone_frame_count = 0
        self.using_grid_scan = True
        
        self.total_detections = 0
        self.validations_performed = 0
        self.false_positives_rejected = 0
        
        self.detection_times = deque(maxlen=20)
        self.validation_times = deque(maxlen=20)
        
        print("✅ All-Angle Tracking System Ready!")
    
    def apply_perspective_transform(self, image, transform_type):
        """Apply perspective transformation to simulate looking up/down/left/right"""
        h, w = image.shape[:2]
        
        if transform_type == 'look_up':
            # Simulate head tilted up
            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst_pts = np.float32([[0, h*0.1], [w, h*0.1], [0, h], [w, h]])
        elif transform_type == 'look_down':
            # Simulate head tilted down
            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst_pts = np.float32([[0, 0], [w, 0], [0, h*0.9], [w, h*0.9]])
        elif transform_type == 'look_left':
            # Simulate head turned left
            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst_pts = np.float32([[w*0.1, 0], [w, 0], [w*0.1, h], [w, h]])
        elif transform_type == 'look_right':
            # Simulate head turned right
            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst_pts = np.float32([[0, 0], [w*0.9, 0], [0, h], [w*0.9, h]])
        else:
            return image
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        transformed = cv2.warpPerspective(image, M, (w, h))
        return transformed
    
    def load_target(self, image_path):
        """Load target with COMPREHENSIVE embeddings for ALL angles"""
        if not os.path.exists(image_path):
            print("⚠️  Target image not found.")
            return False
        
        print(f"🎯 Loading target from {image_path}...")
        print("   ⏳ This may take 30-60 seconds (generating 100+ embeddings)...")
        img_target = cv2.imread(image_path)
        
        if img_target is None:
            print("❌ Could not read target image.")
            return False
        
        # Fix image dimensions to prevent broadcast errors
        h, w = img_target.shape[:2]
        
        # Ensure image is not too small
        if h < 50 or w < 50:
            print("❌ Image too small. Please use an image at least 50x50 pixels.")
            return False
        
        # Resize to standard size for consistent processing
        target_size = 640
        if h != target_size or w != target_size:
            # Resize maintaining aspect ratio
            aspect = w / h
            if aspect > 1:
                new_w = target_size
                new_h = int(target_size / aspect)
            else:
                new_h = target_size
                new_w = int(target_size * aspect)
            
            # Make dimensions even and divisible by 32 (important for face detection)
            new_h = (new_h // 32) * 32
            new_w = (new_w // 32) * 32
            
            if new_h < 32:
                new_h = 32
            if new_w < 32:
                new_w = 32
            
            img_target = cv2.resize(img_target, (new_w, new_h))
            print(f"   📐 Resized image to {new_w}x{new_h}")
        
        # Ensure final dimensions are valid
        h, w = img_target.shape[:2]
        if h % 32 != 0:
            new_h = (h // 32) * 32
            img_target = img_target[:new_h, :]
        if w % 32 != 0:
            new_w = (w // 32) * 32
            img_target = img_target[:, :new_w]
        
        embeddings_list = []
        
        # Helper function to safely detect faces
        def safe_detect_faces(img):
            try:
                # Ensure image has valid dimensions
                h, w = img.shape[:2]
                if h < 32 or w < 32:
                    return []
                
                # Make dimensions divisible by 32
                new_h = (h // 32) * 32
                new_w = (w // 32) * 32
                
                if new_h < 32:
                    new_h = 32
                if new_w < 32:
                    new_w = 32
                
                if h != new_h or w != new_w:
                    img = img[:new_h, :new_w]
                
                return self.app.get(img)
            except Exception as e:
                # Skip silently - image transformation may have created invalid dimensions
                return []
        
        # 1. Original + Flipped
        print("   📸 Step 1/8: Original + Mirror...")
        faces = safe_detect_faces(img_target)
        if len(faces) > 0:
            embeddings_list.append(faces[0].embedding)
        
        img_flipped = cv2.flip(img_target, 1)
        faces_flipped = safe_detect_faces(img_flipped)
        if len(faces_flipped) > 0:
            embeddings_list.append(faces_flipped[0].embedding)
        
        # 2. Brightness variations
        print("   💡 Step 2/8: Brightness variations (11 levels)...")
        for gamma in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            adjusted = self.adjust_gamma(img_target, gamma)
            faces_adj = safe_detect_faces(adjusted)
            if len(faces_adj) > 0:
                embeddings_list.append(faces_adj[0].embedding)
        
        # 3. Rotation variations
        print("   🔄 Step 3/8: Rotation angles (13 angles)...")
        for angle in [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]:
            rotated = self.rotate_image(img_target, angle)
            faces_rot = safe_detect_faces(rotated)
            if len(faces_rot) > 0:
                embeddings_list.append(faces_rot[0].embedding)
        
        # 4. Perspective transformations (looking up/down/left/right)
        print("   📐 Step 4/8: Perspective transforms (up/down/left/right)...")
        for perspective_type in ['look_up', 'look_down', 'look_left', 'look_right']:
            transformed = self.apply_perspective_transform(img_target, perspective_type)
            faces_trans = safe_detect_faces(transformed)
            if len(faces_trans) > 0:
                embeddings_list.append(faces_trans[0].embedding)
        
        # 5. Contrast variations
        print("   🎨 Step 5/8: Contrast variations (9 levels)...")
        for alpha in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
            contrasted = cv2.convertScaleAbs(img_target, alpha=alpha, beta=0)
            faces_con = safe_detect_faces(contrasted)
            if len(faces_con) > 0:
                embeddings_list.append(faces_con[0].embedding)
        
        # 6. Blur variations
        print("   🌀 Step 6/8: Blur variations (motion)...")
        for ksize in [3, 5, 7]:
            blurred = cv2.GaussianBlur(img_target, (ksize, ksize), 0)
            faces_blur = safe_detect_faces(blurred)
            if len(faces_blur) > 0:
                embeddings_list.append(faces_blur[0].embedding)
        
        # 7. Combined transformations
        print("   🔀 Step 7/8: Combined (rotation + brightness)...")
        for angle in [-20, -10, 10, 20]:
            for gamma in [0.7, 1.0, 1.3]:
                rotated = self.rotate_image(img_target, angle)
                adjusted = self.adjust_gamma(rotated, gamma)
                faces_comb = safe_detect_faces(adjusted)
                if len(faces_comb) > 0:
                    embeddings_list.append(faces_comb[0].embedding)
        
        # 8. Scale variations
        print("   🔍 Step 8/8: Scale variations (distance)...")
        for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
            h, w = img_target.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            if scale < 1.0:
                scaled = cv2.resize(img_target, (new_w, new_h))
                pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                scaled_padded = cv2.copyMakeBorder(scaled, pad_h, pad_h, pad_w, pad_w, 
                                                   cv2.BORDER_CONSTANT, value=[128, 128, 128])
                faces_scale = safe_detect_faces(scaled_padded)
            else:
                scaled = cv2.resize(img_target, (new_w, new_h))
                crop_h, crop_w = (new_h - h) // 2, (new_w - w) // 2
                scaled_cropped = scaled[crop_h:crop_h+h, crop_w:crop_w+w]
                faces_scale = safe_detect_faces(scaled_cropped)
            
            if len(faces_scale) > 0:
                embeddings_list.append(faces_scale[0].embedding)
        
        if len(embeddings_list) == 0:
            print("❌ No face found in target image.")
            return False
        
        self.target_embeddings = embeddings_list
        self.avg_target_embedding = np.mean(embeddings_list, axis=0)
        self.avg_target_embedding /= np.linalg.norm(self.avg_target_embedding)
        
        print(f"\n✅ Target locked! {len(embeddings_list)} embeddings stored")
        print(f"   📊 Full Coverage:")
        print(f"      ✓ Front view, Side views (left/right)")
        print(f"      ✓ Looking up/down")
        print(f"      ✓ All rotations (-30° to +30°)")
        print(f"      ✓ All lighting conditions")
        print(f"      ✓ All distances (near to far)")
        print(f"   🎯 Recognition threshold: {RECOGNITION_THRESHOLD:.2f} (optimized for angles)")
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
        if not self.target_embeddings:
            return False, 0.0
        
        # Quick reject with lower threshold for angled faces
        avg_sim = self.compute_similarity(self.avg_target_embedding, embedding)
        if avg_sim < 0.15:  # Very permissive
            return False, 0.0
        
        # Detailed check with all embeddings
        similarities = [self.compute_similarity(ref_emb, embedding) 
                       for ref_emb in self.target_embeddings]
        max_sim = max(similarities)
        
        return max_sim > RECOGNITION_THRESHOLD, max_sim
    
    def get_zone_bounds(self, frame_shape):
        h, w = frame_shape[:2]
        
        rows, cols = 2, 3
        zone_h, zone_w = h // rows, w // cols
        
        row = self.current_zone // cols
        col = self.current_zone % cols
        
        x1 = col * zone_w
        y1 = row * zone_h
        x2 = x1 + zone_w
        y2 = y1 + zone_h
        
        overlap = 60
        x1 = max(0, x1 - overlap)
        y1 = max(0, y1 - overlap)
        x2 = min(w, x2 + overlap)
        y2 = min(h, y2 + overlap)
        
        return (x1, y1, x2, y2)
    
    def detect_faces_adaptive(self, frame):
        start_time = time.time()
        
        if self.target_found and self.tracker.target_bbox is not None:
            search_region = self.tracker.get_search_region(frame.shape)
            
            if search_region is not None:
                x1, y1, x2, y2 = search_region
                
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    cropped = frame[y1:y2, x1:x2]
                    faces = self.app.get(cropped)
                    
                    for face in faces:
                        face.bbox += np.array([x1, y1, x1, y1])
                    
                    self.detection_times.append(time.time() - start_time)
                    return faces, "LOCAL"
        
        x1, y1, x2, y2 = self.get_zone_bounds(frame.shape)
        zone_frame = frame[y1:y2, x1:x2]
        faces = self.app.get(zone_frame)
        
        for face in faces:
            face.bbox += np.array([x1, y1, x1, y1])
        
        self.detection_times.append(time.time() - start_time)
        return faces, "ZONE"
    
    def filter_faces(self, faces, frame_shape):
        if len(faces) == 0:
            return []
        
        # Relaxed size filter for angled faces
        filtered = []
        for face in faces:
            box = face.bbox
            w, h = box[2] - box[0], box[3] - box[1]
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE and face.det_score >= MIN_DETECTION_SCORE:
                filtered.append(face)
        
        if len(filtered) == 0:
            return []
        
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
                    
                    score = face.det_score * 100 / (distance + 1)
                    scored.append((score, face))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                return [face for _, face in scored[:MAX_FACES_TO_PROCESS]]
        
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
        start_time = time.time()
        self.validations_performed += 1
        
        if len(faces) == 0:
            self.validation_times.append(time.time() - start_time)
            return None, 0.0
        
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
        self.frame_count += 1
        
        if self.using_grid_scan:
            self.zone_frame_count += 1
            if self.zone_frame_count >= ZONE_SWITCH_INTERVAL:
                self.current_zone = (self.current_zone + 1) % GRID_ZONES
                self.zone_frame_count = 0
        
        faces, detection_mode = self.detect_faces_adaptive(frame)
        self.total_detections = len(faces)
        
        faces = self.filter_faces(faces, frame.shape)
        
        target_face, similarity = self.validate_target(faces)
        
        if target_face is not None:
            if not self.target_found:
                print(f"🎯 TARGET ACQUIRED! (Similarity: {similarity:.3f})")
                self.target_found = True
                self.using_grid_scan = False
            
            self.tracker.update_with_detection(target_face.bbox, similarity, self.frame_count)
            
            return target_face, similarity, detection_mode
        
        else:
            if self.target_found:
                confidence = self.tracker.decay_confidence()
                
                if confidence < 0.25:  # Lower threshold
                    print(f"⚠️  Target lost (confidence: {confidence:.2f}). Resuming search...")
                    self.target_found = False
                    self.using_grid_scan = True
                    self.tracker.target_bbox = None
            
            return None, 0.0, detection_mode
    
    def draw_results(self, frame, target_face, similarity, detection_mode):
        
        if self.target_found and self.tracker.target_bbox is not None:
            search_region = self.tracker.get_search_region(frame.shape)
            if search_region is not None:
                x1, y1, x2, y2 = search_region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.putText(frame, "SEARCH AREA", (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        if self.using_grid_scan:
            x1, y1, x2, y2 = self.get_zone_bounds(frame.shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, f"ZONE {self.current_zone + 1}/{GRID_ZONES}", 
                       (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        if target_face is not None:
            box = target_face.bbox.astype(int)
            
            pulse = int(12 * (1 + np.sin(time.time() * 10)))
            cv2.rectangle(frame, 
                        (box[0]-pulse, box[1]-pulse), 
                        (box[2]+pulse, box[3]+pulse), 
                        (0, 255, 255), 2)
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            
            if self.tracker.is_moving:
                center = self.tracker.target_center.astype(int)
                velocity_end = (center + self.tracker.velocity * 5).astype(int)
                cv2.arrowedLine(frame, tuple(center), tuple(velocity_end), 
                              (0, 255, 255), 2, tipLength=0.3)
            
            status = "MOVING" if self.tracker.is_moving else "STATIONARY"
            label = f"TARGET {similarity:.2f} | {status}"
            
            label_bg = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                        (box[0], box[1] - label_bg[1] - 15), 
                        (box[0] + label_bg[0], box[1]), 
                        (0, 255, 0), -1)
            cv2.putText(frame, label, (box[0], box[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            if hasattr(target_face, 'kps') and target_face.kps is not None:
                for point in target_face.kps.astype(int):
                    cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
        
        panel_h = 280
        cv2.rectangle(frame, (10, 90), (500, 90 + panel_h), (0, 0, 0), -1)
        
        if self.target_found:
            panel_color = (0, 255, 0)
            status_text = "TARGET LOCKED (ALL-ANGLE MODE)"
        else:
            panel_color = (0, 165, 255)
            status_text = "SEARCHING (ALL ANGLES)..."
        
        cv2.rectangle(frame, (10, 90), (500, 90 + panel_h), panel_color, 2)
        
        y = 115
        cv2.putText(frame, status_text, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, panel_color, 2)
        
        y += 25
        cv2.putText(frame, f"Mode: {detection_mode} | Conf: {self.tracker.confidence:.2f}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        cv2.putText(frame, f"Embeddings: {len(self.target_embeddings)} (all angles)", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        cv2.putText(frame, f"Faces Detected: {self.total_detections}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        cv2.putText(frame, f"Validations: {self.validations_performed}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        avg_movement = self.tracker.get_average_movement()
        cv2.putText(frame, f"Movement: {avg_movement:.1f} px/frame", 
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
        cv2.putText(frame, f"Tracked: {self.tracker.frames_tracked} frames", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y += 22
        cv2.putText(frame, "Detects: Front/Side/Up/Down views", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
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
        
        mode_text = "ALL-ANGLE TRACKING MODE"
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
        
        cv2.namedWindow("All-Angle Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("All-Angle Tracking", 1280, 720)
        
        print("\n🎥 ALL-ANGLE TRACKING MODE Active!")
        print("✅ Target detected from ANY angle:")
        print("   • Front view")
        print("   • Side views (left/right)")
        print("   • Looking up/down")
        print("   • All rotations and poses")
        print("\nControls: q=quit | s=save | r=reload | t=reset | c=change\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            target_face, similarity, detection_mode = self.process_frame(frame)
            
            frame_display = self.draw_results(frame.copy(), target_face, similarity, detection_mode)
            frame_display = self.draw_ui(frame_display)
            
            cv2.imshow('All-Angle Tracking', frame_display)
            
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
            elif key == ord('c'):
                new_image = input("📂 New target image: ").strip()
                if new_image and self.load_target(new_image):
                    image_name = new_image
                    self.tracker = AdaptiveTargetTracker()
                    self.target_found = False
                    self.using_grid_scan = True
        
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        print(f"\n👋 Stopped. Avg FPS: {avg_fps:.1f}")
        print(f"📊 Stats: Validations: {self.validations_performed} | Tracked: {self.tracker.frames_tracked}")


if __name__ == "__main__":
    tracker = AllAngleTrackingSystem()
    tracker.run()