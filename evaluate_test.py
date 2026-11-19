
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix
# )

# from asl_common import (
#     SEQUENCE_LENGTH,
#     SAMPLE_FPS,
#     NO_SIGN,
#     mp_hands,
#     crop_frame,
#     hand_found,
#     keypoints_from_results
# )

# from asl_model import ACTIONS, predict_probs


# # --------------------------------------------------
# # CONFIGURATION
# # --------------------------------------------------

# TEST_PATH = "test"
# RESULTS_PATH = "evaluation_results"

# STATIC_CLASSES = [
#     action for action in ACTIONS
#     if action not in ["J", "Z", NO_SIGN]
# ]

# VIDEO_CLASSES = ["J", "Z", NO_SIGN]

# os.makedirs(RESULTS_PATH, exist_ok=True)

# y_true = []
# y_pred = []
# misclassified = []
# skipped = []


# # --------------------------------------------------
# # STATIC IMAGE PROCESSING
# # Same approach as original training data:
# # one image -> landmarks -> repeated 20 frames
# # --------------------------------------------------

# def process_static_image(image_path, hands):

#     image = cv2.imread(image_path)

#     if image is None:
#         return None

#     # Original static-image training used the image
#     # directly, without the video ROI crop.
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     results = hands.process(image)

#     keypoints = keypoints_from_results(results)

#     sequence = np.repeat(
#         keypoints[np.newaxis, :],
#         SEQUENCE_LENGTH,
#         axis=0
#     )

#     return sequence


# # --------------------------------------------------
# # VIDEO PROCESSING
# # 20 frames sampled at 15 FPS
# # --------------------------------------------------

# def process_video(video_path, hands):

#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         return None

#     source_fps = cap.get(cv2.CAP_PROP_FPS)

#     if not source_fps or source_fps <= 0:
#         cap.release()
#         return None

#     frame_interval = 1.0 / SAMPLE_FPS
#     next_sample_time = 0.0

#     frames = []
#     detected = 0
#     frame_index = 0

#     while len(frames) < SEQUENCE_LENGTH:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         current_time = frame_index / source_fps
#         frame_index += 1

#         if current_time + 1e-9 < next_sample_time:
#             continue

#         next_sample_time += frame_interval

#         _, crop = crop_frame(frame)

#         rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb)

#         if hand_found(results):
#             detected += 1

#         frames.append(keypoints_from_results(results))

#     cap.release()

#     if len(frames) != SEQUENCE_LENGTH:
#         return None

#     # Match the original recorder's minimum hand
#     # detection requirements.
#     hand_share = detected / SEQUENCE_LENGTH

#     if hand_share < 0.70:
#         return None

#     return np.array(frames, dtype=np.float32)


# # --------------------------------------------------
# # EVALUATION
# # --------------------------------------------------

# def evaluate_sample(path, true_label, sequence):

#     if sequence is None:
#         skipped.append(path)
#         print(f"Skipped: {path}")
#         return

#     probabilities = predict_probs(sequence)

#     predicted_index = int(np.argmax(probabilities))
#     predicted_label = ACTIONS[predicted_index]

#     y_true.append(true_label)
#     y_pred.append(predicted_label)

#     if predicted_label != true_label:
#         misclassified.append({
#             "file": path,
#             "actual": true_label,
#             "predicted": predicted_label,
#             "confidence": float(probabilities[predicted_index])
#         })

#     print(
#         f"Actual: {true_label:6} | "
#         f"Predicted: {predicted_label:6} | "
#         f"{os.path.basename(path)}"
#     )


# # --------------------------------------------------
# # MAIN
# # --------------------------------------------------

# with mp_hands.Hands(
#     static_image_mode=False,
#     model_complexity=0,
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# ) as hands:

#     for label in ACTIONS:

#         folder = os.path.join(TEST_PATH, label)

#         if not os.path.isdir(folder):
#             print(f"Missing folder: {folder}")
#             continue

#         files = sorted(os.listdir(folder))

#         for filename in files:

#             path = os.path.join(folder, filename)
#             extension = os.path.splitext(filename)[1].lower()

#             if label in STATIC_CLASSES:

#                 if extension not in [".jpg", ".jpeg", ".png"]:
#                     continue

#                 sequence = process_static_image(path, hands)

#             elif label in VIDEO_CLASSES:

#                 if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
#                     continue

#                 sequence = process_video(path, hands)

#             else:
#                 continue

#             evaluate_sample(path, label, sequence)


# # --------------------------------------------------
# # RESULTS
# # --------------------------------------------------

# print("\n========== EVALUATION RESULTS ==========")

# if not y_true:
#     print("No valid test samples were evaluated.")
#     raise SystemExit

# accuracy = accuracy_score(y_true, y_pred)

# print(f"\nValid samples: {len(y_true)}")
# print(f"Skipped samples: {len(skipped)}")
# print(f"Misclassified samples: {len(misclassified)}")
# print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# print("\nClassification Report:\n")

# print(
#     classification_report(
#         y_true,
#         y_pred,
#         labels=ACTIONS,
#         target_names=ACTIONS,
#         zero_division=0
#     )
# )


# # --------------------------------------------------
# # CONFUSION MATRIX
# # --------------------------------------------------

# cm = confusion_matrix(
#     y_true,
#     y_pred,
#     labels=ACTIONS
# )

# plt.figure(figsize=(18, 15))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=ACTIONS,
#     yticklabels=ACTIONS
# )

# plt.title("Independent Test Dataset - Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("Actual Label")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()

# plt.savefig(
#     os.path.join(RESULTS_PATH, "confusion_matrix.png"),
#     dpi=300
# )

# plt.show()


# # --------------------------------------------------
# # SAVE MISCLASSIFIED SAMPLES
# # --------------------------------------------------

# with open(
#     os.path.join(RESULTS_PATH, "misclassified_samples.txt"),
#     "w"
# ) as file:

#     for item in misclassified:
#         file.write(
#             f"File: {item['file']}\n"
#             f"Actual: {item['actual']}\n"
#             f"Predicted: {item['predicted']}\n"
#             f"Confidence: {item['confidence']:.4f}\n"
#             f"{'-' * 40}\n"
#         )


# # --------------------------------------------------
# # SAVE SKIPPED SAMPLES
# # --------------------------------------------------

# with open(
#     os.path.join(RESULTS_PATH, "skipped_samples.txt"),
#     "w"
# ) as file:

#     for path in skipped:
#         file.write(path + "\n")


# print(f"\nResults saved in: {RESULTS_PATH}")




import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import mediapipe as mp

from asl_common import (
    SEQUENCE_LENGTH,
    SAMPLE_FPS,
    NO_SIGN,
    crop_frame,
    hand_found,
    keypoints_from_results
)

from asl_model import ACTIONS, predict_probs


# ==========================================
# PATHS AND SETTINGS
# ==========================================

TEST_PATH = "test"
RESULTS_PATH = "evaluation_results"

VIDEO_CLASSES = {"J", "Z", NO_SIGN}
STATIC_CLASSES = set(ACTIONS) - VIDEO_CLASSES

os.makedirs(RESULTS_PATH, exist_ok=True)

y_true = []
y_pred = []
misclassified = []
skipped = []


# ==========================================
# STATIC IMAGE PREPROCESSING
# ==========================================

def process_static_image(path, hands):

    image = cv2.imread(path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    keypoints = keypoints_from_results(results)

    # Original training method:
    # Repeat the same image landmarks 20 times.
    sequence = np.repeat(
        keypoints[np.newaxis, :],
        SEQUENCE_LENGTH,
        axis=0
    )

    return sequence.astype(np.float32)


# ==========================================
# VIDEO PREPROCESSING
# ==========================================

def process_video(path, hands):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps <= 0:
        cap.release()
        return None

    interval = 1.0 / SAMPLE_FPS
    next_sample = 0.0

    frames = []
    detected = 0
    frame_index = 0

    while len(frames) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_index / source_fps
        frame_index += 1

        if timestamp + 1e-9 < next_sample:
            continue

        next_sample += interval

        _, crop = crop_frame(frame)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if hand_found(results):
            detected += 1

        frames.append(keypoints_from_results(results))

    cap.release()

    if len(frames) != SEQUENCE_LENGTH:
        return None

    # Same minimum hand-detection share as recorder.
    if detected / SEQUENCE_LENGTH < 0.70:
        return None

    return np.asarray(frames, dtype=np.float32)


def load_saved_sequence(sequence_path):

    # Sort numerically: 0, 1, 2, ... 19
    files = [
        f for f in os.listdir(sequence_path)
        if f.endswith(".npy")
    ]

    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    if len(files) != SEQUENCE_LENGTH:
        print(
            f"Invalid frame count: {sequence_path} "
            f"({len(files)} frames)"
        )
        return None

    frames = []

    for filename in files:
        frame = np.load(
            os.path.join(sequence_path, filename)
        )

        if frame.shape != (63,):
            print("Invalid frame shape:", filename, frame.shape)
            return None

        frames.append(frame)

    sequence = np.asarray(frames, dtype=np.float32)

    if sequence.shape != (SEQUENCE_LENGTH, 63):
        return None

    return sequence

# ==========================================
# PREDICTION
# ==========================================

def evaluate_sample(path, actual, sequence):

    if sequence is None:
        skipped.append(path)
        print("SKIPPED:", path)
        return

    probabilities = predict_probs(sequence)

    predicted = ACTIONS[int(np.argmax(probabilities))]

    y_true.append(actual)
    y_pred.append(predicted)

    if predicted != actual:
        misclassified.append({
            "file": path,
            "actual": actual,
            "predicted": predicted,
            "confidence": float(np.max(probabilities))
        })

    print(
        f"Actual: {actual:6} | "
        f"Predicted: {predicted:6} | "
        f"{os.path.basename(path)}"
    )


# ==========================================
# MAIN EVALUATION
# ==========================================

# image_extensions = {".jpg", ".jpeg", ".png"}
# video_extensions = {".mp4", ".avi", ".mov", ".mkv"}


# ==========================================
# MAIN EVALUATION
# ==========================================

image_extensions = {".jpg", ".jpeg", ".png"}

with mp.solutions.hands.Hands(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.3
) as image_hands:

    for label in ACTIONS:

        folder = os.path.join(TEST_PATH, label)

        if not os.path.isdir(folder):
            print("Missing folder:", folder)
            continue

        # ----------------------------------
        # STATIC LETTERS
        # ----------------------------------

        if label not in VIDEO_CLASSES:

            for filename in sorted(os.listdir(folder)):

                path = os.path.join(folder, filename)

                if not os.path.isfile(path):
                    continue

                extension = os.path.splitext(filename)[1].lower()

                if extension not in image_extensions:
                    continue

                sequence = process_static_image(
                    path, image_hands
                )

                evaluate_sample(path, label, sequence)

        # ----------------------------------
        # DYNAMIC LETTERS + NO SIGN
        # Load already extracted .npy data
        # ----------------------------------

        else:

            sequence_folders = sorted(
                [
                    name for name in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, name))
                ],
                key=lambda x: int(x) if x.isdigit() else x
            )

            for sequence_name in sequence_folders:

                sequence_path = os.path.join(
                    folder, sequence_name
                )

                sequence = load_saved_sequence(
                    sequence_path
                )

                evaluate_sample(
                    sequence_path,
                    label,
                    sequence
                )



# with mp.solutions.hands.Hands(
#     static_image_mode=False,
#     model_complexity=0,
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# ) as video_hands:

#     with mp.solutions.hands.Hands(
#         static_image_mode=True,
#         model_complexity=0,
#         min_detection_confidence=0.3
#     ) as image_hands:

#         for label in ACTIONS:

#             folder = os.path.join(TEST_PATH, label)

#             if not os.path.isdir(folder):
#                 print("Missing folder:", folder)
#                 continue

#             for filename in sorted(os.listdir(folder)):

#                 path = os.path.join(folder, filename)

#                 if not os.path.isfile(path):
#                     continue

#                 extension = os.path.splitext(filename)[1].lower()

#                 if label in STATIC_CLASSES:

#                     if extension not in image_extensions:
#                         continue

#                     sequence = process_static_image(
#                         path, image_hands
#                     )

#                 else:

#                     if extension not in video_extensions:
#                         continue

#                     sequence = process_video(
#                         path, video_hands
#                     )

#                 evaluate_sample(path, label, sequence)


# ==========================================
# METRICS
# ==========================================

print("\n========== EVALUATION RESULTS ==========")

if not y_true:
    raise RuntimeError("No valid test samples were evaluated.")

print("Evaluated samples:", len(y_true))
print("Skipped samples:", len(skipped))
print("Misclassified:", len(misclassified))

accuracy = accuracy_score(y_true, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(ACTIONS),
        target_names=list(ACTIONS),
        zero_division=0
    )
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=ACTIONS)

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=ACTIONS,
    yticklabels=ACTIONS
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("ASL Recognition - Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(ACTIONS)
)

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ACTIONS,
    yticklabels=ACTIONS
)

plt.title("Independent Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_PATH, "confusion_matrix.png"),
    dpi=300
)

plt.show()


# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    os.path.join(RESULTS_PATH, "misclassified_samples.txt"),
    "w"
) as f:

    for item in misclassified:
        f.write(
            f"File: {item['file']}\n"
            f"Actual: {item['actual']}\n"
            f"Predicted: {item['predicted']}\n"
            f"Confidence: {item['confidence']:.4f}\n"
            f"{'-' * 40}\n"
        )


with open(
    os.path.join(RESULTS_PATH, "skipped_samples.txt"),
    "w"
) as f:

    for path in skipped:
        f.write(path + "\n")


print("\nResults saved in:", RESULTS_PATH)
