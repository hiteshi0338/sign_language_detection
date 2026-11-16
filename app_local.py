import time
from collections import deque

import cv2
import numpy as np

from asl_common import (SEQUENCE_LENGTH, SAMPLE_FPS, NO_SIGN, mp_hands, crop_frame,
                        hand_found, keypoints_from_results, draw_hand, draw_text,
                        CROP_X1, CROP_X2, CROP_Y1, CROP_Y2)
from asl_model import ACTIONS, predict_probs, display_name

SMOOTH_SAMPLES = 4         
STABLE_SAMPLES = 6          
THRESHOLD = 0.80        
NO_HAND_RESET = 5           
MIN_HAND_SHARE = 0.70       
MOTION_LETTERS = {"J", "Z"}
MOTION_THRESHOLD = 0.90     
MOTION_STABLE_SAMPLES = 2   
START_POSE_FIX = {"J": {"I"}, "Z": {"D", "G", "L"}}
FIX_WINDOW_SECONDS = 1.6

buffer = deque(maxlen=SEQUENCE_LENGTH)       
present = deque(maxlen=SEQUENCE_LENGTH)     
history = deque(maxlen=SMOOTH_SAMPLES)       
recent = deque(maxlen=STABLE_SAMPLES)        
motion_recent = deque(maxlen=MOTION_STABLE_SAMPLES)  

text = ""
last_letter = None
last_typed_at = 0.0
no_hand = 0
top = []
last_results = None


def reset_tracking():
    global last_letter
    buffer.clear()
    present.clear()
    history.clear()
    recent.clear()
    motion_recent.clear()
    last_letter = None


def type_letter(letter):
    """Add a letter, replacing a just-typed starting handshape for J / Z."""
    global text, last_letter, last_typed_at
    now = time.perf_counter()
    if (text and letter in START_POSE_FIX and text[-1] in START_POSE_FIX[letter]
            and now - last_typed_at < FIX_WINDOW_SECONDS):
        text = text[:-1]
    text += letter
    last_letter = letter
    last_typed_at = now


def step(results):
    """Run once per sample (SAMPLE_FPS times a second)."""
    global no_hand, top
    found = hand_found(results)
    buffer.append(keypoints_from_results(results))
    present.append(found)

    if not found:
        no_hand += 1
        if no_hand >= NO_HAND_RESET:
            reset_tracking()
            top = []
        return
    no_hand = 0

    if len(buffer) < SEQUENCE_LENGTH or sum(present) < MIN_HAND_SHARE * SEQUENCE_LENGTH:
        return

    raw = predict_probs(np.array(buffer))
    history.append(raw)
    probs = np.mean(history, axis=0)             
    best = int(np.argmax(probs))
    order = np.argsort(probs)[::-1][:3]
    top = [(display_name(ACTIONS[i]), float(probs[i])) for i in order]

    
    raw_best = int(np.argmax(raw))
    if ACTIONS[raw_best] in MOTION_LETTERS and raw[raw_best] >= MOTION_THRESHOLD:
        motion_recent.append(raw_best)
    else:
        motion_recent.clear()
    if (len(motion_recent) == MOTION_STABLE_SAMPLES and len(set(motion_recent)) == 1
            and ACTIONS[raw_best] != last_letter):
        type_letter(ACTIONS[raw_best])
        motion_recent.clear()
        recent.clear()
        return

    
    if (ACTIONS[best] == NO_SIGN or ACTIONS[best] in MOTION_LETTERS
            or probs[best] < THRESHOLD):
        recent.clear()
        return
    recent.append(best)
    if (len(recent) == STABLE_SAMPLES and len(set(recent)) == 1
            and ACTIONS[best] != last_letter):
        type_letter(ACTIONS[best])


def draw_ui(full):
    cv2.rectangle(full, (CROP_X1, CROP_Y1), (CROP_X2, CROP_Y2), (255, 255, 255), 2)
    draw_text(full, "HAND HERE", (CROP_X1 + 85, CROP_Y1 - 10), 0.55)
    if no_hand >= 2:
        draw_text(full, "Move your hand", (CROP_X1 + 40, CROP_Y1 + 170))
        draw_text(full, "into this box", (CROP_X1 + 50, CROP_Y1 + 200))
    elif len(buffer) < SEQUENCE_LENGTH:
        draw_text(full, "reading...", (CROP_X1 + 90, CROP_Y2 - 15), 0.55, (0, 200, 255))

    cv2.rectangle(full, (0, 0), (full.shape[1], 36), (245, 117, 16), -1)
    draw_text(full, "Text: " + text[-40:], (8, 26), 0.8)
    for i, (name, p) in enumerate(top):
        y = 60 + i * 36
        cv2.rectangle(full, (320, y), (320 + int(p * 200), y + 26), (245, 117, 16), -1)
        draw_text(full, f"{name} {p * 100:.0f}%", (325, y + 20), 0.6)
    draw_text(full, "SPACE space | BKSP delete | C clear | Q quit",
              (8, full.shape[0] - 12), 0.5)


def main():
    global text, last_letter, last_results
    cap = cv2.VideoCapture(0)
    interval = 1.0 / SAMPLE_FPS
    next_sample = time.perf_counter()

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break
            full, crop = crop_frame(frame)

            now = time.perf_counter()
            if now >= next_sample:
                next_sample = max(next_sample + interval, now - interval)
                last_results = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                step(last_results)
            if last_results is not None:
                draw_hand(crop, last_results)
            draw_ui(full)
            cv2.imshow("ASL A-Z", full)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                text = ""
                reset_tracking()
            elif key == 32:
                text += " "
                last_letter = None
            elif key == 8:
                text = text[:-1]
                last_letter = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
