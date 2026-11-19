# ASL Alphabet Recognition (A-Z) with MediaPipe and an LSTM

Recognises the American Sign Language alphabet from a webcam. Hand landmarks are
extracted with MediaPipe Hands and classified by an LSTM. J and Z are learned as real
movements, and an extra "no sign" class keeps the model quiet when a hand is visible
but not making a letter.

**Live demo:** https://hiteshi0338.github.io/sign_language_detection/  (runs entirely in the browser; no video is uploaded)

## Results

**Accuracy on unseen data: 76.7%** (586/764 samples across 27 classes)

| Metric | Score |
|---|---:|
| Accuracy | 76.7% |
| Macro F1 | 0.72 |
| Weighted F1 | 0.74 |

The independent test set was collected separately and was never used for training or tuning, providing a realistic estimate of model performance.

### Per-Class Performance

| Tier | Classes |
|---|---|
| **Strong (F1 ≥ 0.90)** | B, F, I, K, M, P, Q, V, W, X, Y |
| **Medium (0.60–0.89)** | A, C, D, E, H, J, L, S, U |
| **Weak (F1 < 0.60)** | R, G, N, NOSIGN, O, T, Z |

### Key Limitations
Each still letter has 30 test samples, J and Z have 15 each, and NOSIGN has 14, so a single
class score is only a rough guide. The clearest failures are:

- **T, N, O:** Difficulty distinguishing similar hand shapes.
- **NOSIGN:** Frequently misclassified as letters.

The full classification report is in `evaluation_results.txt`.


## How it works

```
Webcam -> crop box -> MediaPipe (21 hand landmarks = 63 numbers)
       -> last 20 frames sampled at 15 fps -> LSTM -> letter
```

- Still letters were recorded as images and converted to keypoints.
- J, Z and NOSIGN were recorded as 20-frame clips with `record_clips.py`.
- Training runs on Kaggle (GPU). The weights are exported to `model_weights.npz`, and the
  apps run them with plain NumPy or JavaScript, so TensorFlow is not needed to use the model.
- Keypoints are raw positions inside a fixed crop box, so hand position and distance
  affect the result.

## Files

| File | Purpose |
|---|---|
| `collectdata.py` | Capture still images of letters |
| `function.py`, `data.py` | Convert still images to keypoint sequences (`MP_Data/`) |
| `record_clips.py` | Record motion clips for J, Z and NOSIGN |
| `ASL_Motion_Kaggle.ipynb` | Train the LSTM and export the weights |
| `evaluate_test.py` | Evaluate the trained model on a separate test set |
| `evaluation_results.txt` | Output of the evaluation on unseen data |
| `asl_common.py` | Shared settings (sequence length, crop box, helpers) |
| `asl_model.py` | NumPy implementation of the trained LSTM |
| `app_local.py` | Live OpenCV app, supports J and Z |
| `app.py` | Gradio web app (still letters only, plus photo upload) |
| `docs/` | Browser version (GitHub Pages): `index.html`, `app.js`, `asl_logic.js`, `model.json`, self-hosted MediaPipe files |
| `export_web_model.py` | Converts `model_weights.npz` into `docs/model.json` |
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

Web app (still letters only):

```
python app.py
```

Then open http://127.0.0.1:7860.

Browser version, locally:

```
python export_web_model.py
python -m http.server 8000 --directory docs
```

Then open http://localhost:8000 in Chrome or Edge.

## Evaluate

```
python evaluate_test.py
```

The script expects static gesture images to be stored directly inside `test/<CLASS>/` (e.g., `test/A/image1.jpg`), while dynamic gestures and No Sign must be stored as separate sequence folders containing 20 numerically named `.npy` files each (e.g., `test/J/60/0.npy` to `19.npy`).

## Retrain with your own data

1. `python collectdata.py` - capture still images for the letters.
2. `python data.py` - convert them to keypoints (skips J and Z).
3. `python record_clips.py` - record J, Z and NOSIGN clips.
4. Zip `MP_Data`, upload it as a Kaggle dataset, run `ASL_Motion_Kaggle.ipynb`.
5. Download `model_weights.npz` and `actions.json` into the project folder.
6. Run `python export_web_model.py` to refresh the browser version.



## Limitations

- Overall accuracy on unseen data is 76.7%. Several classes are unreliable: T, Z, O, N, G,
  R and the "no sign" class.
- Training data came from a single recording setup, which limits how well the model
  handles new conditions.
- Single hand, ASL only, fixed crop box on the left of the frame.
- The browser version uses the same model, but its accuracy was not measured separately,
  and J and Z need a fast enough device (about 12-15 tracking samples per second).
- Words and phrases (hello, mom, papa, goodbye) are not implemented.

## Possible improvements

- Train on more people, lighting and backgrounds, especially for the weak classes.
- Normalise keypoints relative to the wrist and add scale and shift augmentation, so the
  model depends less on where the hand is in the box.
- Re-test on a fresh set of unseen people and settings.