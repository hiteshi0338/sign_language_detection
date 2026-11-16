# ASL Alphabet Recognition (A-Z) with MediaPipe and an LSTM

Recognises the American Sign Language alphabet from a webcam. Hand landmarks are
extracted with MediaPipe Hands, and an LSTM classifies them. J and Z are learned
as real movements. A "no sign" class keeps the model quiet when the hand is visible
but not making a letter.

## Results

| | |
|---|---|
| Classes | 27 (A-Z + NOSIGN) |
| Test accuracy | 96.0% (193 / 201 held-out sequences) |
| Split | 70% train / 15% validation / 15% test (augmentation on training data only) |

**Caveat:** the test data comes from the same recording session as the training data
(same person, lighting and background), so accuracy on new people or settings is
expected to be lower. Per-class scores are based on only 6-9 test clips each.
Weakest letters in testing: O, S, N, C, T and E.

## How it works

```
Webcam -> crop box -> MediaPipe (21 hand landmarks = 63 numbers)
       -> last 20 frames sampled at 15 fps -> LSTM -> letter
```

- Still letters were recorded as images and converted to keypoints.
- J, Z and NOSIGN were recorded as 20-frame clips with `record_clips.py`.
- Training runs on Kaggle (GPU). The trained weights are exported to
  `model_weights.npz`, and the apps run them with plain NumPy, so TensorFlow is not
  needed to use the model.

## Files

| File | Purpose |
|---|---|
| `collectdata.py` | Capture still images of letters (press the letter key) |
| `function.py`, `data.py` | Convert still images to keypoint sequences (`MP_Data/`) |
| `record_clips.py` | Record motion clips for J, Z and NOSIGN |
| `ASL_Motion_Kaggle.ipynb` | Train the LSTM and export the weights |
| `asl_common.py` | Shared settings (sequence length, crop box, helpers) |
| `asl_model.py` | NumPy implementation of the trained LSTM |
| `app_local.py` | Live OpenCV app, supports J and Z |
| `app.py` | Gradio web app (still letters only, plus photo upload) |
| `model_weights.npz`, `actions.json` | The trained model and its class order |

The image and keypoint datasets are not stored in this repository.

## Setup

Python 3.11 is required (MediaPipe 0.10.14 does not support newer versions).

```
python -m venv .venv311
.venv311\Scripts\activate
pip install mediapipe==0.10.14 "numpy<2" opencv-python==4.10.0.84 gradio
```

## Run

Live camera app (J and Z included):

```
python app_local.py
```

Keys: `Space` adds a space, `Backspace` deletes, `C` clears, `Q` quits.
Keep your hand inside the white box.

Web app:

```
python app.py
```

Then open http://127.0.0.1:7860. A web page delivers too few frames per second for
motion, so J and Z are only available in `app_local.py`.

## Retrain with your own data

1. `python collectdata.py` - capture still images for the letters.
2. `python data.py` - convert them to keypoints (skips J and Z).
3. `python record_clips.py` - record J, Z and NOSIGN clips.
4. Zip `MP_Data`, upload it as a Kaggle dataset, run `ASL_Motion_Kaggle.ipynb`.
5. Download `model_weights.npz` and `actions.json` into the project folder.

## Limitations and ideas

- Single hand, ASL only, fixed crop box on the left of the frame.
- Words and phrases (hello, mom, papa, goodbye) are not implemented. They would need
  two-hand and body tracking plus a separate sequence model.
- Planned: hosted web version, more data for confusable letters, evaluation on
  people not in the training set.