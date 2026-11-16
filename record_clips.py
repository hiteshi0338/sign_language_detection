"""
Record short motion clips for J, Z and the extra "no sign" class.
Runs on your laptop. Needs: mediapipe==0.10.14, opencv-python, numpy.

How to use
  1. Run:  python record_clips.py
  2. Keep your hand inside the white box.
  3. Press a key, then do the sign during the red RECORDING bar:
        j  -> J : make the I handshape, then draw a J in the air with your little finger
        z  -> Z : point your index finger and draw a Z in the air
        n  -> no sign : hand visible but NOT a letter (moving into the box, waving,
              relaxed hand, fist turning, shaking...). Teaches the model to stay quiet.
  4. Each clip is 20 frames sampled at 15 per second (about 1.3 seconds).
  5. ESC quits. Running it again carries on where you stopped.

Aim for 60+ clips of J and of Z, and 60+ of 'no sign'. Vary speed, size, angle,
and starting position. Keep the whole movement inside the box.
"""
import os
import sys
import time

import cv2
import numpy as np

from asl_common import (SEQUENCE_LENGTH, SAMPLE_FPS, NO_SIGN, mp_hands, crop_frame,
                        hand_found, keypoints_from_results, draw_hand, draw_text,
                        CROP_X1, CROP_X2, CROP_Y1, CROP_Y2)

DATA_PATH = "MP_Data"
TARGET_CLIPS = 60
COUNTDOWN_SECONDS = 2.0

CLASSES = {
    ord('j'): ("J", 0.80, "J: make the I shape, draw a J with your little finger"),
    ord('z'): ("Z", 0.80, "Z: point your index finger, draw a Z in the air"),
    ord('n'): (NO_SIGN, 0.50, "NO SIGN: move/wave/relax your hand - do NOT sign a letter"),
}


def clip_dirs(label):
    path = os.path.join(DATA_PATH, label)
    return [d for d in os.listdir(path) if d.isdigit()] if os.path.isdir(path) else []


def check_old_data():
   
    for label in ("J", "Z"):
        for d in clip_dirs(label):
            n = len(os.listdir(os.path.join(DATA_PATH, label, d)))
            if n != SEQUENCE_LENGTH:
                sys.exit(f"MP_Data/{label} contains old still-image data "
                         f"({n} frames per sequence, expected {SEQUENCE_LENGTH}).\n"
                         f"Delete the folder MP_Data/{label} and run this again.")


def show(window, image, top, bottom=None, color=(245, 117, 16)):
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), color, -1)
    draw_text(image, top, (8, 26), 0.65)
    if bottom:
        draw_text(image, bottom, (8, image.shape[0] - 12), 0.55)
    cv2.imshow(window, image)


def record_clip(cap, hands, label, min_share, instruction):
    interval = 1.0 / SAMPLE_FPS
    start = time.perf_counter()
    countdown_end = start + COUNTDOWN_SECONDS
    frames, seen = [], 0
    next_t = countdown_end
    t_first = None

    while len(frames) < SEQUENCE_LENGTH:
        ok, frame = cap.read()
        if not ok:
            return "camera read failed"
        full, crop = crop_frame(frame)
        now = time.perf_counter()

        if now < countdown_end:
            left = countdown_end - now
            cv2.rectangle(full, (CROP_X1, CROP_Y1), (CROP_X2, CROP_Y2), (255, 255, 255), 2)
            draw_text(full, str(int(left) + 1), (CROP_X1 + 120, CROP_Y1 + 190), 1.5, (0, 165, 255), 3)
            show("Record clips", full, f"GET READY - {label}", instruction, (0, 140, 255))
            cv2.waitKey(1)
            continue

        if now >= next_t:                      
            next_t += interval
            if t_first is None:
                t_first = now
            results = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            found = hand_found(results)
            frames.append(keypoints_from_results(results))
            seen += found
            draw_hand(crop, results)

        cv2.rectangle(full, (CROP_X1, CROP_Y1), (CROP_X2, CROP_Y2), (255, 255, 255), 2)
        show("Record clips", full, f"RECORDING {label}  {len(frames)}/{SEQUENCE_LENGTH}",
             instruction, (0, 0, 200))
        cv2.waitKey(1)

    duration = time.perf_counter() - t_first
    expected = SEQUENCE_LENGTH / SAMPLE_FPS
    if seen / SEQUENCE_LENGTH < min_share:
        return f"discarded: hand found in only {seen}/{SEQUENCE_LENGTH} frames"
    if duration > expected * 1.5:
        return (f"discarded: camera too slow ({SEQUENCE_LENGTH / duration:.0f} fps, "
                f"need about {SAMPLE_FPS}). Close other apps using the camera.")

    ids = [int(d) for d in clip_dirs(label)]
    clip_id = max(ids) + 1 if ids else 0
    folder = os.path.join(DATA_PATH, label, str(clip_id))
    os.makedirs(folder, exist_ok=True)
    for n, kp in enumerate(frames):
        np.save(os.path.join(folder, str(n)), kp)
    return f"saved ({len(clip_dirs(label))}/{TARGET_CLIPS})"


def main():
    check_old_data()
    cap = cv2.VideoCapture(0)
    message = "Press j, z or n to record a clip. ESC to quit."

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break
            full, crop = crop_frame(frame)
            draw_hand(crop, hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            cv2.rectangle(full, (CROP_X1, CROP_Y1), (CROP_X2, CROP_Y2), (255, 255, 255), 2)

            counts = "   ".join(f"{v[0]}: {len(clip_dirs(v[0]))}/{TARGET_CLIPS}"
                                for v in CLASSES.values())
            show("Record clips", full, message, counts)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                break
            if key in CLASSES:
                label, min_share, instruction = CLASSES[key]
                result = record_clip(cap, hands, label, min_share, instruction)
                message = f"{label}: {result}"
                print(message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
