import cv2
import mediapipe as mp
import time
import numpy as np
import csv
from collections import deque
 
# EAR FUNCTION
 
def calculate_EAR(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)
 
 
def get_eye_points(landmarks, indices, w, h):
    return np.array([
        [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
        for idx in indices
    ])
 
 
def get_gaze_ratio(eye, iris):
    eye_center = np.mean(eye, axis=0)
    iris_center = np.mean(iris, axis=0)
    eye_width = abs(eye[3][0] - eye[0][0])
    if eye_width == 0:
        return 0
    return (iris_center[0] - eye_center[0]) / eye_width
 
 
def get_brow_height(landmarks, brow_idx, eye_idx, w, h, norm_factor):
    brow = [int(landmarks[i].y * h) for i in brow_idx]
    eye = [int(landmarks[i].y * h) for i in eye_idx]
    raw = (sum(eye) / len(eye)) - (sum(brow) / len(brow))
    if norm_factor == 0:
        return raw
    return raw / norm_factor
 
# FOREHEAD WRINKLE
 
FOREHEAD_POINTS = [10, 67, 103, 109, 338, 297, 332]
ROI_MARGIN_RATIO = 0.12  # shrink ROI inward so crop edges aren't detected as fake edges
  
def get_forehead_roi(frame, landmarks, indices, w, h):
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
    x, y, w_box, h_box = cv2.boundingRect(pts)
 
    mx = int(w_box * ROI_MARGIN_RATIO)
    my = int(h_box * ROI_MARGIN_RATIO)
    x, y = x + mx, y + my
    w_box, h_box = max(1, w_box - 2 * mx), max(1, h_box - 2 * my)
 
    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x + w_box)
    y2 = min(h, y + h_box)
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]
 
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
 
def wrinkle_intensity(roi):
    if roi is None or roi.size == 0:
        return None  # Return None instead of 0 to indicate a failed detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)             
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges) / 255.0)
 
# MOUTH / LIP / FACE-HEIGHT FUNCTIONS
 
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)
 
 
def get_point(landmarks, idx, w, h):
    return np.array([
        int(landmarks[idx].x * w),
        int(landmarks[idx].y * h)
    ])
 
 
def get_mouth_open_ratio(landmarks, w, h):
    upper_lip = get_point(landmarks, UPPER_LIP_INNER[0], w, h)
    lower_lip = get_point(landmarks, LOWER_LIP_INNER[0], w, h)
    left_mouth = get_point(landmarks, LEFT_MOUTH[0], w, h)
    right_mouth = get_point(landmarks, RIGHT_MOUTH[0], w, h)
 
    mouth_open = distance(upper_lip, lower_lip)
    mouth_width = distance(left_mouth, right_mouth)
    return mouth_open / mouth_width if mouth_width != 0 else 0
 
 
def get_lip_thickness(landmarks, w, h, norm_factor):
    
    upper_outer = get_point(landmarks, UPPER_LIP_OUTER[0], w, h)
    upper_inner = get_point(landmarks, UPPER_LIP_INNER[0], w, h)
    lower_outer = get_point(landmarks, LOWER_LIP_OUTER[0], w, h)
    lower_inner = get_point(landmarks, LOWER_LIP_INNER[0], w, h)
 
    thickness = distance(upper_outer, upper_inner) + distance(lower_outer, lower_inner)
    return thickness / norm_factor if norm_factor != 0 else thickness
 
 
def get_face_height(landmarks, w, h, norm_factor):
    # Experimental, low-confidence proxy — excluded from score.
    top = get_point(landmarks, JAW_BOTTOM[0], w, h)
    chin = get_point(landmarks, JAW_TOP[0], w, h)
    raw = distance(top, chin)
    return raw / norm_factor if norm_factor != 0 else raw
 
def get_head_pose(landmarks, w, h):
    # Key Landmarks
    nose = get_point(landmarks, 4, w, h)
    left_eye_outer = get_point(landmarks, 33, w, h)
    right_eye_outer = get_point(landmarks, 263, w, h)
    left_temple = get_point(landmarks, 127, w, h)
    right_temple = get_point(landmarks, 356, w, h)
    chin = get_point(landmarks, 152, w, h)
    forehead = get_point(landmarks, 10, w, h)

    # 1. Roll: Eye-to-eye angle
    dy = right_eye_outer[1] - left_eye_outer[1]
    dx = right_eye_outer[0] - left_eye_outer[0]
    roll = np.degrees(np.arctan2(dy, dx))

    # 2. Yaw: Horizontal symmetry (nose relative to temples)
    dist_left = distance(nose, left_temple)
    dist_right = distance(nose, right_temple)
    total_w = dist_left + dist_right
    yaw = ((dist_right - dist_left) / total_w) * 90 if total_w != 0 else 0

    # 3. Pitch: Vertical symmetry (nose relative to forehead and chin)
    dist_up = distance(nose, forehead)
    dist_down = distance(nose, chin)
    total_h = dist_up + dist_down
    pitch = ((dist_down - dist_up) / total_h) * 90 if total_h != 0 else 0

    return pitch, yaw, roll

# INIT
 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
mp_drawing = mp.solutions.drawing_utils
 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
 
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
 
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
 
LEFT_EYE_TOP = [159]
RIGHT_EYE_TOP = [386]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
 
UPPER_LIP_INNER = [13]
LOWER_LIP_INNER = [14]
UPPER_LIP_OUTER = [0]
LOWER_LIP_OUTER = [17]
LEFT_MOUTH = [61]
RIGHT_MOUTH = [291]
 
JAW_TOP = [152]
JAW_BOTTOM = [10]

#INITALIZE CAMER
cap = cv2.VideoCapture(0)

# 1. CAMERA WARMUP (Let auto-exposure settle)
print("Initializing camera and auto-exposure (2 seconds)...")
warmup_start = time.time()
while time.time() - warmup_start < 2.0:
    success, frame = cap.read()
    if not success:
        break
 
# CALIBRATION
 
CALIBRATION_SECONDS = 10
print(f"Calibrating... look at the camera normally for {CALIBRATION_SECONDS} seconds.")
 
calib_start = time.time()
calib_ears = []
calib_brows = []
calib_wrinkles = []
calib_mouth_open = []
calib_lip_thickness = []
calib_face_height = []
calib_interocular = []
calib_pitches = []
calib_yaws = []
calib_rolls = []
 
while time.time() - calib_start < CALIBRATION_SECONDS:
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
 
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        left = get_eye_points(lm, LEFT_EYE, w, h)
        right = get_eye_points(lm, RIGHT_EYE, w, h)
        ear = (calculate_EAR(left) + calculate_EAR(right)) / 2.0
        if ear > 0:
            calib_ears.append(ear)
 
        interocular = abs(lm[LEFT_EYE_OUTER].x * w - lm[RIGHT_EYE_OUTER].x * w)
        lb = get_brow_height(lm, LEFT_BROW, LEFT_EYE_TOP, w, h, interocular)
        rb = get_brow_height(lm, RIGHT_BROW, RIGHT_EYE_TOP, w, h, interocular)
        calib_brows.append((lb + rb) / 2.0)
 
        roi = get_forehead_roi(frame, lm, FOREHEAD_POINTS, w, h)
        wrinkle_val = wrinkle_intensity(roi)
        if wrinkle_val is not None:
            calib_wrinkles.append(wrinkle_val)
 
        calib_mouth_open.append(get_mouth_open_ratio(lm, w, h))
        calib_lip_thickness.append(get_lip_thickness(lm, w, h, interocular))
        calib_face_height.append(get_face_height(lm, w, h, interocular))

        # Head Pose Calibration
        p, y, r = get_head_pose(lm, w, h)
        calib_pitches.append(p)
        calib_yaws.append(y)
        calib_rolls.append(r)

    cv2.putText(frame, "Calibrating... look at camera normally(SIT STILL AND RELAXED)", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
if len(calib_ears) == 0:
    print("Calibration failed — no face detected. Using default threshold.")
    baseline_ear = 0.30
else:
    baseline_ear = float(np.mean(calib_ears))
 
baseline_brow = float(np.mean(calib_brows)) if calib_brows else 0.15
brow_std = float(np.std(calib_brows)) if calib_brows else 0.02
raw_wrinkle_mean = float(np.mean(calib_wrinkles)) if calib_wrinkles else 0.05
baseline_wrinkle = raw_wrinkle_mean if raw_wrinkle_mean > 0.001 else 0.05
baseline_mouth_open = float(np.mean(calib_mouth_open)) if calib_mouth_open else 0.1
baseline_lip_thickness = float(np.mean(calib_lip_thickness)) if calib_lip_thickness else 0.3
baseline_face_height = float(np.mean(calib_face_height)) if calib_face_height else 1.5
baseline_interocular = float(np.mean(calib_interocular)) if calib_interocular else 100.0
baseline_pitch = float(np.mean(calib_pitches)) if calib_pitches else 0.0
baseline_yaw = float(np.mean(calib_yaws)) if calib_yaws else 0.0
baseline_roll = float(np.mean(calib_rolls)) if calib_rolls else 0.0
 
EAR_THRESHOLD = baseline_ear * 0.75
 
print("Baseline EAR:", round(baseline_ear, 3))
print("Baseline Brow:", round(baseline_brow, 3))
print("Baseline Wrinkle:", round(baseline_wrinkle, 3))
print("Baseline Mouth Openness:", round(baseline_mouth_open, 3))
print("Baseline Lip Thickness:", round(baseline_lip_thickness, 3))
print("Baseline Face Height (experimental):", round(baseline_face_height, 3))
print("Baseline Head Pose (P/Y/R):", round(baseline_pitch, 1), round(baseline_yaw, 1), round(baseline_roll, 1))
 
# MAIN LOOP
 
DURATION = 30
EYE_CLOSED_FRAMES = 2
MIN_BLINK_INTERVAL = 0.3
SACCADE_THRESHOLD = 0.05
MIN_SACCADE_INTERVAL = 0.15
FIXATION_THRESHOLD = 0.02
 
blink_count = 0
closed_frames = 0
blink_start_time = None
last_blink_time = 0
 
ear_list = []
gaze_list = []
brow_list = []
wrinkle_list = []
mouth_open_list = []
lip_thickness_list = []
face_height_list = []
pitch_list = []
yaw_list = []
roll_list = []
lean_list = []

# Sliding window (30 frames ~1 sec) to track micro-movements (stability)
pose_sliding_window = deque(maxlen=30)
stability_scores = []

blink_durations = []
fixation_durations = []
movement_speeds = []
 
saccade_count = 0
last_saccade_time = 0
 
# raised-eyebrow event tracking (duration + count), same pattern as blinks
raised_brow_start = None
raised_brow_durations = []
raised_brow_count = 0
 
ear_smoothing_window = deque(maxlen=3)
wrinkle_smoothing_window = deque(maxlen=5)

smoothed_wrinkle_list = []
last_frame_time = time.time()
fixation_start = None
 
smoothed_wrinkle = baseline_wrinkle
 
csv_file = open("facial_features_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp", "ear", "smoothed_ear", "gaze", "brow", "brow_state",
    "wrinkle", "smoothed_wrinkle","mouth_open_ratio", "lip_thickness", "face_height",
    "pitch", "yaw", "roll", "forward_lean_ratio", "head_stability_std",
    "is_blink_frame", "is_saccade", "is_fixation"
])
 
start_global = time.time()
 
while time.time() - start_global < DURATION:
    success, frame = cap.read()
    if not success:
        break
 
    now_frame_time = time.time()
    dt = now_frame_time - last_frame_time
    last_frame_time = now_frame_time
 
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
 
    is_blink_frame = 0
    is_saccade = 0
    is_fixation = 0
    gaze = 0
    gaze_text = "N/A"
    brow_state = "N/A"
    wrinkle = 0
    stability_val = 0.0
 
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION
        )
        lm = face_landmarks.landmark
 
        left = get_eye_points(lm, LEFT_EYE, w, h)
        right = get_eye_points(lm, RIGHT_EYE, w, h)
        l_iris = get_eye_points(lm, LEFT_IRIS, w, h)
        r_iris = get_eye_points(lm, RIGHT_IRIS, w, h)
 
        ear = (calculate_EAR(left) + calculate_EAR(right)) / 2.0
        ear_list.append(ear)
 
        ear_smoothing_window.append(ear)
        smoothed_ear = sum(ear_smoothing_window) / len(ear_smoothing_window)
 
        if ear < EAR_THRESHOLD:
            if blink_start_time is None:
                blink_start_time = time.time()
            closed_frames += 1
            is_blink_frame = 1
        else:
            if closed_frames >= EYE_CLOSED_FRAMES:
                now = time.time()
                if now - last_blink_time > MIN_BLINK_INTERVAL:
                    blink_count += 1
                    last_blink_time = now
                    if blink_start_time:
                        blink_durations.append(now - blink_start_time)
            closed_frames = 0
            blink_start_time = None
 
        gaze = (get_gaze_ratio(left, l_iris) + get_gaze_ratio(right, r_iris)) / 2.0
        gaze_list.append(gaze)
 
        if gaze < -0.2:
            gaze_text = "LEFT"
        elif gaze > 0.2:
            gaze_text = "RIGHT"
        else:
            gaze_text = "CENTER"
 
        if len(gaze_list) > 1 and dt > 0:
            diff = abs(gaze - gaze_list[-2])
            speed = diff / dt
            movement_speeds.append(speed)
 
            if diff > SACCADE_THRESHOLD:
                now = time.time()
                is_saccade = 1
                if now - last_saccade_time > MIN_SACCADE_INTERVAL:
                    saccade_count += 1
                    last_saccade_time = now
 
            if diff < FIXATION_THRESHOLD:
                is_fixation = 1
                if fixation_start is None:
                    fixation_start = time.time()
            else:
                if fixation_start:
                    fixation_durations.append(time.time() - fixation_start)
                    fixation_start = None
 
        interocular = abs(lm[LEFT_EYE_OUTER].x * w - lm[RIGHT_EYE_OUTER].x * w)
 
        lb = get_brow_height(lm, LEFT_BROW, LEFT_EYE_TOP, w, h, interocular)
        rb = get_brow_height(lm, RIGHT_BROW, RIGHT_EYE_TOP, w, h, interocular)
 
        brow = (lb + rb) / 2.0
        brow_list.append(brow)
 
        now = time.time()
 
        # dynamic threshold based on your real calibration noise
        threshold = max(brow_std * 1.5, 0.01)
 
        if brow < baseline_brow - threshold:
            brow_state = "FURROW"
 
            if raised_brow_start:
                raised_brow_durations.append(now - raised_brow_start)
                raised_brow_start = None
 
        elif brow > baseline_brow + threshold:
            brow_state = "RAISED"
 
            if raised_brow_start is None:
                raised_brow_start = now
                raised_brow_count += 1
 
        else:
            brow_state = "NORMAL"
 
            if raised_brow_start:
                raised_brow_durations.append(now - raised_brow_start)
                raised_brow_start = None
 
        roi = get_forehead_roi(frame, lm, FOREHEAD_POINTS, w, h)
        wrinkle_raw = wrinkle_intensity(roi)
        if wrinkle_raw is not None:
            wrinkle = wrinkle_raw
            wrinkle_list.append(wrinkle)
            wrinkle_smoothing_window.append(wrinkle)
            smoothed_wrinkle = sum(wrinkle_smoothing_window) / len(wrinkle_smoothing_window)
        else:
            # If face is lost/no ROI, reuse baseline or last available smoothed frame
            wrinkle = baseline_wrinkle
            if smoothed_wrinkle_list:
                smoothed_wrinkle = smoothed_wrinkle_list[-1]
                
        smoothed_wrinkle_list.append(smoothed_wrinkle)
 
        # MOUTH openness (gap between lips)
        mouth_open_ratio = get_mouth_open_ratio(lm, w, h)
        mouth_open_list.append(mouth_open_ratio)
 
        # LIP THICKNESS — real AU24 compression proxy, independent of openness
        lip_thickness = get_lip_thickness(lm, w, h, interocular)
        lip_thickness_list.append(lip_thickness)
 
        # FACE HEIGHT (experimental, not scored)
        face_height = get_face_height(lm, w, h, interocular)
        face_height_list.append(face_height)
 
        # HEAD POSE
        pitch, yaw, roll = get_head_pose(lm, w, h)
        # Ratio > 1.05 means user is leaning forward (closer to camera)
        lean_ratio = interocular / baseline_interocular

        pitch_list.append(pitch)
        yaw_list.append(yaw)
        roll_list.append(roll)
        lean_list.append(lean_ratio)
 
        # Head Stability (using sliding window)
        pose_sliding_window.append((pitch, yaw, roll))
        if len(pose_sliding_window) >= 15:
            arr = np.array(pose_sliding_window)
            # Sum of standard deviations across all 3 axes
            stability_val = float(np.sum(np.std(arr, axis=0)))
            stability_scores.append(stability_val)
        else:
            stability_val = 0.0
 
        cv2.putText(frame, f"EAR:{ear:.2f}", (30, 30), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks:{blink_count}", (30, 52), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze:{gaze_text}", (30, 74), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Brow:{brow_state}", (30, 96), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Wrinkle:{smoothed_wrinkle:.2f}", (30, 118), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"MouthOpen:{mouth_open_ratio:.2f}", (30, 140), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"LipThick:{lip_thickness:.2f}", (30, 162), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch-baseline_pitch:.1f} Yaw: {yaw-baseline_yaw:.1f}", (30, 30), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Lean Ratio: {lean_ratio:.2f}", (30, 52), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Micro-movement (Stability): {stability_val:.2f}", (30, 74), 0, 0.55, (0, 255, 0), 2)
 
        csv_writer.writerow([
            time.time(), round(ear, 4), round(smoothed_ear, 4), round(gaze, 4),
            round(brow, 4), brow_state,
            round(wrinkle, 4), round(smoothed_wrinkle, 4),
            round(mouth_open_ratio, 4), round(lip_thickness, 4), round(face_height, 4),
            round(pitch, 2), round(yaw, 2), round(roll, 2),
            round(lean_ratio, 3), round(stability_val, 3),
            is_blink_frame, is_saccade, is_fixation
        ])
 
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
# close out any open raised-brow event at end of session
if raised_brow_start:
    raised_brow_durations.append(time.time() - raised_brow_start)
 
cap.release()
cv2.destroyAllWindows()
csv_file.close()
 
# POST ANALYSIS
 
total_time = time.time() - start_global
avg_ear = float(np.mean(ear_list)) if ear_list else 0
blink_rate = (blink_count / DURATION) * 60
gaze_var = float(np.var(gaze_list)) if gaze_list else 0
avg_fixation = float(np.mean(fixation_durations)) if fixation_durations else 0
avg_blink_duration = float(np.mean(blink_durations)) if blink_durations else 0
avg_speed = float(np.mean(movement_speeds)) if movement_speeds else 0
avg_brow = float(np.mean(brow_list)) if brow_list else 0
avg_wrinkle = float(np.mean(wrinkle_list)) if wrinkle_list else 0
avg_smoothed_wrinkle = float(np.mean(smoothed_wrinkle_list)) if smoothed_wrinkle_list else 0
avg_mouth_open = float(np.mean(mouth_open_list)) if mouth_open_list else 0
avg_lip_thickness = float(np.mean(lip_thickness_list)) if lip_thickness_list else 0
avg_face_height = float(np.mean(face_height_list)) if face_height_list else 0
avg_raised_brow_duration = float(np.mean(raised_brow_durations)) if raised_brow_durations else 0

avg_pitch = float(np.mean(pitch_list)) if pitch_list else 0.0
avg_yaw = float(np.mean(yaw_list)) if yaw_list else 0.0
avg_lean = float(np.mean(lean_list)) if lean_list else 1.0
avg_stability = float(np.mean(stability_scores)) if stability_scores else 0.0
 
eye_strain = 1 if avg_ear < baseline_ear * 0.85 else (0.5 if avg_ear < baseline_ear * 0.95 else 0)
stress_signal = 1 if blink_rate > 30 else (0.7 if blink_rate > 20 else (0.4 if blink_rate > 12 else 0))
brow_signal = 1 if avg_brow < baseline_brow * 0.85 else (0.5 if avg_brow < baseline_brow * 0.95 else 0)
gaze_signal = 1 if gaze_var > 0.05 else (0.5 if gaze_var > 0.03 else 0)
 
# lip compression: real thickness-based proxy now, not mouth-openness reuse
lip_signal = 1 if avg_lip_thickness < baseline_lip_thickness * 0.8 else (0.5 if avg_lip_thickness < baseline_lip_thickness * 0.9 else 0)
 
wrinkle_signal = 1 if avg_smoothed_wrinkle > baseline_wrinkle * 1.5 else (0.5 if avg_smoothed_wrinkle > baseline_wrinkle * 1.2 else 0)

lean_signal = 0.5 if avg_lean > 1.05 else 0.0
stillness_signal = 0.5 if avg_stability < 1.2 else 0.0

cognitive_score = (
    eye_strain + stress_signal + brow_signal + gaze_signal + lip_signal + 
    (wrinkle_signal * 0.5) + lean_signal + stillness_signal
) / 6.5 * 100
 
print("\n------ RESULT (placeholder scoring, not validated) ------")
print(f"Baseline EAR: {baseline_ear:.3f}")
print(f"Avg EAR (session): {avg_ear:.3f}")
print(f"Blink Rate (per min): {blink_rate:.2f}")
print(f"Avg Blink Duration (sec): {avg_blink_duration:.3f}")
print(f"Gaze Variance: {gaze_var:.4f}")
print(f"Saccades: {saccade_count}")
print(f"Avg Fixation Duration (sec): {avg_fixation:.3f}")
print(f"Avg Eye Movement Speed: {avg_speed:.3f}")
print(f"Avg Brow (normalized): {avg_brow:.3f}")
print(f"Raised-Brow Events: {raised_brow_count}, Avg Duration (sec): {avg_raised_brow_duration:.3f} [logged only]")
print(f"Avg Wrinkle (reworked, vs baseline {baseline_wrinkle:.3f}): {avg_wrinkle:.3f} [half-weight in score]")
print(f"Avg Mouth Openness: {avg_mouth_open:.3f} (baseline {baseline_mouth_open:.3f})")
print(f"Avg Lip Thickness (AU24 proxy): {avg_lip_thickness:.3f} (baseline {baseline_lip_thickness:.3f})")
print(f"Avg Face Height [EXPERIMENTAL, not scored]: {avg_face_height:.3f} (baseline {baseline_face_height:.3f})")
print(f"Avg Head Pitch Deviation: {avg_pitch - baseline_pitch:.2f} degrees")
print(f"Avg Head Yaw Deviation: {avg_yaw - baseline_yaw:.2f} degrees")
print(f"Avg Forward Lean Ratio: {avg_lean:.3f} (Values > 1.0 mean leaning closer to monitor)")
print(f"Head Micro-Movement Index (lower = more still): {avg_stability:.3f}")
print(f"Cognitive Load Score: {cognitive_score:.0f}/100")
print("Per-frame data saved to facial_features_log.csv")