import threading
import cv2
import gradio as gr
import numpy as np

from asl_common import (SEQUENCE_LENGTH, NO_SIGN, mp_hands, crop_frame, hand_found,
                        keypoints_from_results, draw_hand, draw_text,
                        FRAME_W, FRAME_H, CROP_X1, CROP_X2, CROP_Y1, CROP_Y2,
                        CROP_W, CROP_H)
from asl_model import ACTIONS, predict_probs, display_name

SMOOTH_FRAMES = 3              
STABLE_FRAMES = 3              
THRESHOLD = 0.80               
NO_HAND_RESET = 4              


hands = mp_hands.Hands(static_image_mode=True, model_complexity=0,
                       max_num_hands=1, min_detection_confidence=0.5)
hands_lock = threading.Lock()


def analyze(crop):
 
    with hands_lock:
        results = hands.process(crop)
    if not hand_found(results):
        return None
    keypoints = keypoints_from_results(results)
    probs = predict_probs(np.tile(keypoints, (SEQUENCE_LENGTH, 1)))
    draw_hand(crop, results)
    return probs


def top3(probs):
    order = np.argsort(probs)[::-1][:3]
    return {display_name(ACTIONS[i]): float(probs[i]) for i in order}


def new_state():
    return {"history": [], "recent": [], "text": "", "last": None, "no_hand": 0}


def process(image, state):
    if state is None:
        state = new_state()
    if image is None:
        return None, {}, state["text"], state

    frame, crop = crop_frame(image)
    crop = np.ascontiguousarray(crop)
    probs = analyze(crop)

    label = {}
    if probs is not None:
        state["no_hand"] = 0
        state["history"] = (state["history"] + [probs])[-SMOOTH_FRAMES:]
        smooth = np.mean(state["history"], axis=0)      
        best = int(np.argmax(smooth))
        label = top3(smooth)

        if ACTIONS[best] == NO_SIGN or smooth[best] < THRESHOLD:
            state["recent"] = []
        else:
            state["recent"] = (state["recent"] + [best])[-STABLE_FRAMES:]
            if (len(state["recent"]) == STABLE_FRAMES
                    and len(set(state["recent"])) == 1
                    and ACTIONS[best] != state["last"]):
                state["text"] += ACTIONS[best]
                state["last"] = ACTIONS[best]
        frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2] = crop     
    else:
        state["no_hand"] += 1
        state["recent"] = []
        state["history"] = []
        if state["no_hand"] >= NO_HAND_RESET:
            state["last"] = None               
        draw_text(frame, "Move your hand", (CROP_X1 + 40, CROP_Y1 + 170))
        draw_text(frame, "into this box", (CROP_X1 + 50, CROP_Y1 + 200))

    cv2.rectangle(frame, (CROP_X1, CROP_Y1), (CROP_X2, CROP_Y2), (255, 255, 255), 2)
    draw_text(frame, "HAND HERE", (CROP_X1 + 85, CROP_Y1 - 10), 0.55)
    return frame, label, state["text"], state


def clear_text(state):
    state = state or new_state()
    state.update(text="", last=None, recent=[], history=[])
    return "", state


def add_space(state):
    state = state or new_state()
    state["text"] += " "
    state["last"] = None
    return state["text"], state


def backspace(state):
    state = state or new_state()
    state["text"] = state["text"][:-1]
    state["last"] = None
    return state["text"], state


def center_crop_to_ratio(image, ratio):

    h, w = image.shape[:2]
    if w / h > ratio:                       
        new_w = int(h * ratio)
        x0 = (w - new_w) // 2
        return image[:, x0:x0 + new_w]
    new_h = int(w / ratio)                 
    y0 = (h - new_h) // 2
    return image[y0:y0 + new_h, :]


def predict_upload(image):
    if image is None:
        return None, {}, "Upload a photo first."
    crop = center_crop_to_ratio(image, CROP_W / CROP_H)
    crop = np.ascontiguousarray(cv2.resize(crop, (CROP_W, CROP_H)))
    probs = analyze(crop)
    if probs is None:
        return crop, {}, "No hand found. Use a clear photo with the whole hand visible."
    best = int(np.argmax(probs))
    return crop, top3(probs), f"Prediction: {display_name(ACTIONS[best])}  ({probs[best] * 100:.0f}% sure)"


SPEAK_JS = """
(t) => {
  if (!t || !t.trim()) { return; }
  if (!('speechSynthesis' in window)) { alert('Speech is not supported in this browser.'); return; }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(t));
}
"""

with gr.Blocks(title="ASL Letter Recognition") as demo:
    gr.Markdown("# ASL letter recognition")
    state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Live camera"):
            gr.Markdown(
                "Allow camera access, then hold your hand **inside the white box** "
                "on the left side of the picture and keep the sign steady. "
                "Use a plain background and good lighting.  \n"
                "*J and Z are drawn with movement, so this web version cannot read them. "
                "Use the local version (app_local.py) for those.*"
            )
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, label="Your camera",
                               mirror_webcam=False, type="numpy",
                               webcam_constraints={"video": {"width": FRAME_W, "height": FRAME_H}})
                out_img = gr.Image(label="What the model sees", interactive=False)
            with gr.Row():
                label = gr.Label(num_top_classes=3, label="Top guesses")
                text = gr.Textbox(label="Text so far", interactive=False, lines=2)
            with gr.Row():
                speak_btn = gr.Button("Speak", variant="primary")
                space_btn = gr.Button("Space")
                back_btn = gr.Button("Backspace")
                clear_btn = gr.Button("Clear")

        with gr.Tab("Upload photo"):
            gr.Markdown(
                "Test a single photo. Use a portrait-shaped photo of just your hand, "
                "like your training images (the middle of the picture is used)."
            )
            with gr.Row():
                up_in = gr.Image(sources=["upload"], type="numpy", label="Photo")
                up_out = gr.Image(label="What the model sees", interactive=False)
            up_label = gr.Label(num_top_classes=3, label="Top guesses")
            up_msg = gr.Textbox(label="Result", interactive=False)
            up_btn = gr.Button("Predict", variant="primary")

    cam.stream(process, [cam, state], [out_img, label, text, state],
               stream_every=0.2, time_limit=600, concurrency_limit=4)
    clear_btn.click(clear_text, state, [text, state])
    space_btn.click(add_space, state, [text, state])
    back_btn.click(backspace, state, [text, state])
    speak_btn.click(None, inputs=text, outputs=None, js=SPEAK_JS)
    up_btn.click(predict_upload, up_in, [up_out, up_label, up_msg])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)


