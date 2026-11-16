import cv2
import mediapipe as mp
import numpy as np

SEQUENCE_LENGTH = 20      
SAMPLE_FPS = 15           
KEYPOINTS = 21 * 3        
NO_SIGN = "NOSIGN"

FRAME_W, FRAME_H = 640, 480
CROP_X1, CROP_X2 = 0, 300
CROP_Y1, CROP_Y2 = 40, 400
CROP_W, CROP_H = CROP_X2 - CROP_X1, CROP_Y2 - CROP_Y1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def crop_frame(frame):
    """Resize a camera frame to 640x480 and return (full_frame, crop_view)."""
    full = cv2.resize(frame, (FRAME_W, FRAME_H))
    return full, full[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]


def hand_found(results):
    return bool(results.multi_hand_landmarks)


def keypoints_from_results(results):
    """63 raw numbers for the first hand (same as your original extract_keypoints)."""
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    return np.zeros(KEYPOINTS)


def draw_hand(image, results):
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style())


def draw_text(img, text, org, scale=0.6, color=(255, 255, 255), thickness=2):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)
