# main.py
"""
Real-Time Static Hand Gesture Recognition (rules-based, MediaPipe + OpenCV)
Gesture vocabulary:
 - open_palm, fist, peace (v), thumbs_up, ok, pointing, crossed_fingers
Author: Your Name
Run: python main.py
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time

# ---------- Configuration ----------
CAMERA_ID = 0
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Smoothing: require same gesture for N frames before confirming
SMOOTHING_BUFFER = 6

# Minimum normalized distance threshold multiplier for "close" checks (OK, crossed)
DISTANCE_THRESHOLD_MULT = 0.18  # fraction of hand bbox diagonal

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# ---------- helper functions ----------
def landmark_to_np(landmark_list, image_w, image_h):
    """Convert MediaPipe landmark list to Nx2 numpy array in pixel coords."""
    points = []
    for lm in landmark_list.landmark:
        points.append((lm.x * image_w, lm.y * image_h))
    return np.array(points)  # shape (21, 2)

def normalized_landmarks(landmark_list):
    """Return Nx2 array with normalized (x,y) in [0,1] from MediaPipe landmark_list"""
    return np.array([[lm.x, lm.y] for lm in landmark_list.landmark])

def hand_bbox_size(landmarks):
    """Estimate hand size from landmarks (normalized coords). We'll use diagonal of bbox."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    diag = np.sqrt(w * w + h * h)
    return diag if diag > 1e-6 else 1e-6

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# finger landmark indices (MediaPipe)
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0

def finger_is_extended(landmarks, finger_tip_idx, finger_pip_idx):
    """
    For frontal hand, we check if fingertip y is above PIP y.
    Normalized coords: y increases downward, so smaller y means higher/up.
    This assumes a roughly upright camera orientation (frontal).
    """
    tip_y = landmarks[finger_tip_idx][1]
    pip_y = landmarks[finger_pip_idx][1]
    return tip_y < pip_y  # True if finger is extended (tip higher than pip)

def thumb_is_extended(landmarks):
    """
    Thumb extension test: check relative x position of thumb tip compared to IP/MCP.
    For frontal single-hand, check if thumb_tip is far from palm center and thumb_tip.x 
    is to one side compared to thumb_mcp.x. We'll also check vertical relation to detect thumbs-up.
    """
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    mcp = landmarks[THUMB_MCP]
    wrist = landmarks[WRIST]
    # distance from thumb tip to wrist (normalized)
    dist_to_wrist = euclidean(tip, wrist)
    # basic extension: tip farther from wrist than MCP is.
    extended = dist_to_wrist > euclidean(mcp, wrist) * 1.05
    return extended

def thumbs_up_detected(landmarks, hand_diag):
    """
    Thumbs up heuristic:
     - thumb extended
     - other fingers folded
     - thumb tip is higher (smaller y) than thumb IP (i.e., pointing up)
    Uses normalized coordinates.
    """
    thumb_ext = thumb_is_extended(landmarks)
    others_folded = not finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP) \
                    and not finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP) \
                    and not finger_is_extended(landmarks, RING_TIP, RING_PIP) \
                    and not finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP)
    thumb_pointing_up = landmarks[THUMB_TIP][1] < landmarks[THUMB_IP][1]  # y smaller = up
    # ensure thumb visible and reasonably separated from palm
    return thumb_ext and others_folded and thumb_pointing_up

def open_palm_detected(landmarks):
    return all([
        finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP),
        finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP),
        finger_is_extended(landmarks, RING_TIP, RING_PIP),
        finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP),
        thumb_is_extended(landmarks)
    ])

def fist_detected(landmarks):
    # all fingers folded -> tips lower than pip (y greater)
    return all([
        not finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP),
        not finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP),
        not finger_is_extended(landmarks, RING_TIP, RING_PIP),
        not finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP),
        not thumb_is_extended(landmarks)
    ])

def peace_detected(landmarks):
    # index & middle extended, ring & pinky folded; thumb can be either
    return (finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP)
            and finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
            and not finger_is_extended(landmarks, RING_TIP, RING_PIP)
            and not finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP))

def pointing_detected(landmarks):
    # index extended, others folded (thumb folded or relaxed)
    return (finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP)
            and not finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
            and not finger_is_extended(landmarks, RING_TIP, RING_PIP)
            and not finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP))

def ok_detected(landmarks, hand_diag):
    # OK: thumb tip near index tip + other fingers extended or relaxed (we'll allow both)
    tip_thumb = landmarks[THUMB_TIP]
    tip_index = landmarks[INDEX_TIP]
    dist = euclidean(tip_thumb, tip_index)
    thresh = DISTANCE_THRESHOLD_MULT * hand_diag
    return dist < thresh

def crossed_fingers_detected(landmarks, hand_diag):
    # Crossed fingers: index and middle tips very close (overlapping) while both being somewhat extended
    tip_index = landmarks[INDEX_TIP]
    tip_middle = landmarks[MIDDLE_TIP]
    dist = euclidean(tip_index, tip_middle)
    thresh = (DISTANCE_THRESHOLD_MULT * 1.1) * hand_diag  # slightly looser than OK threshold
    index_ext = finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle_ext = finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    # Additionally ensure ring and pinky are folded (likely when crossing)
    ring_folded = not finger_is_extended(landmarks, RING_TIP, RING_PIP)
    pinky_folded = not finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP)
    return (index_ext and middle_ext and ring_folded and pinky_folded and dist < thresh)

def classify_hand_gesture(landmarks):
    """
    landmarks: Nx2 normalized coordinates (0..1)
    returns: gesture_name (str) or 'Unknown'
    """
    hand_diag = hand_bbox_size(landmarks)  # in normalized units
    # Check gestures in order of specificity
    if fist_detected(landmarks):
        return "Fist"
    if open_palm_detected(landmarks):
        return "Open Palm"
    if thumbs_up_detected(landmarks, hand_diag):
        return "Thumbs Up"
    if peace_detected(landmarks):
        return "Peace"
    if ok_detected(landmarks, hand_diag):
        return "OK"
    if crossed_fingers_detected(landmarks, hand_diag):
        return "Crossed Fingers"
    if pointing_detected(landmarks):
        return "Pointing"
    return "Unknown"

# ---------- Main loop ----------
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # smoothing buffer
    recent_gestures = deque(maxlen=SMOOTHING_BUFFER)
    gesture_to_display = "No Hand"

    prev_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_h, image_w = frame.shape[:2]

            results = hands.process(frame_rgb)

            gesture = "No Hand"
            annotated_frame = frame.copy()

            if results.multi_hand_landmarks:
                # we only use the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # normalized landmarks array (N,2)
                landmarks = normalized_landmarks(hand_landmarks)

                # classify
                gesture = classify_hand_gesture(landmarks)

                # draw bounding rectangle around hand for visual feedback
                xs = landmarks[:, 0] * image_w
                ys = landmarks[:, 1] * image_h
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                # pad
                pad = 10
                cv2.rectangle(annotated_frame, (x_min - pad, y_min - pad), (x_max + pad, y_max + pad),
                              (0, 255, 0), 2)

            # smoothing: push gesture and decide majority in buffer
            recent_gestures.append(gesture)
            if len(recent_gestures) == recent_gestures.maxlen:
                most_common, count = Counter(recent_gestures).most_common(1)[0]
                # require majority (more than half)
                if count > recent_gestures.maxlen // 2:
                    gesture_to_display = most_common

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else fps
            prev_time = curr_time

            # overlay text
            cv2.putText(annotated_frame, f"Gesture: {gesture_to_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Raw: {gesture}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, image_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Hand Gesture Recognition", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()