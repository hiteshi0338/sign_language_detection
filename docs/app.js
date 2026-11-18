
import { Recognizer, decodeModel, CONFIG } from "./asl_logic.js";

const FRAME_W = 640, FRAME_H = 480;
const CROP = { x: 0, y: 40, w: 300, h: 360 };

const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

const $ = (id) => document.getElementById(id);
const view = $("view");
const vctx = view.getContext("2d");
const cropCanvas = document.createElement("canvas");
cropCanvas.width = CROP.w;
cropCanvas.height = CROP.h;
const cctx = cropCanvas.getContext("2d", { willReadFrequently: true });
const video = document.createElement("video");
video.playsInline = true;
video.muted = true;

let recognizer = null;
let hands = null;
let lastLandmarks = null;     
let running = false;
let busy = false;
let nextSample = 0;
const sampleTimes = [];

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function outlinedText(text, x, y, size = 18) {
  vctx.font = `600 ${size}px system-ui, sans-serif`;
  vctx.lineWidth = 4;
  vctx.strokeStyle = "rgba(0,0,0,0.85)";
  vctx.strokeText(text, x, y);
  vctx.fillStyle = "#fff";
  vctx.fillText(text, x, y);
}

function drawOverlay() {
  vctx.lineWidth = 2;
  vctx.strokeStyle = "#ffffff";
  vctx.strokeRect(CROP.x, CROP.y, CROP.w, CROP.h);
  outlinedText("HAND HERE", CROP.x + 88, CROP.y - 10, 16);

  if (lastLandmarks) {
    const px = (p) => [CROP.x + p.x * CROP.w, CROP.y + p.y * CROP.h];
    vctx.lineWidth = 3;
    vctx.strokeStyle = "#35d07f";
    vctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      const [x1, y1] = px(lastLandmarks[a]);
      const [x2, y2] = px(lastLandmarks[b]);
      vctx.moveTo(x1, y1);
      vctx.lineTo(x2, y2);
    }
    vctx.stroke();
    vctx.fillStyle = "#ff4d4d";
    for (const p of lastLandmarks) {
      const [x, y] = px(p);
      vctx.beginPath();
      vctx.arc(x, y, 3.5, 0, Math.PI * 2);
      vctx.fill();
    }
  }
  if (recognizer.noHand >= 2) {
    outlinedText("Move your hand", CROP.x + 62, CROP.y + 175);
    outlinedText("into this box", CROP.x + 72, CROP.y + 203);
  } else if (recognizer.bufferLength < CONFIG.SEQUENCE_LENGTH && lastLandmarks) {
    outlinedText("reading...", CROP.x + 100, CROP.y + CROP.h - 14, 16);
  }
}

function renderPanel() {
  $("text").textContent = recognizer.text;
  const bars = $("bars");
  bars.replaceChildren();
  for (const [name, p] of recognizer.top) {
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.round(p * 100)}%`;
    const label = document.createElement("span");
    label.textContent = `${name}  ${Math.round(p * 100)}%`;
    bar.append(fill, label);
    bars.append(bar);
  }
  if (sampleTimes.length > 3) {
    const span = (sampleTimes[sampleTimes.length - 1] - sampleTimes[0]) / 1000;
    const fps = (sampleTimes.length - 1) / span;
    $("rate").textContent = `Tracking speed: ${fps.toFixed(0)} samples per second`
      + (fps < 12 ? " (slow: J and Z may not be recognised; close other tabs or apps)" : "");
  }
}

function landmarksToKeypoints(list) {
  const kp = new Array(CONFIG.KEYPOINTS);
  list.forEach((p, i) => { kp[3 * i] = p.x; kp[3 * i + 1] = p.y; kp[3 * i + 2] = p.z; });
  return kp;
}

let latestResults = null;

async function takeSample(now) {
  cctx.drawImage(view, CROP.x, CROP.y, CROP.w, CROP.h, 0, 0, CROP.w, CROP.h);
  latestResults = null;
  await hands.send({ image: cropCanvas });
  const found = latestResults && latestResults.multiHandLandmarks
    && latestResults.multiHandLandmarks.length > 0;
  if (found) {
    lastLandmarks = latestResults.multiHandLandmarks[0];
    recognizer.step(landmarksToKeypoints(lastLandmarks), now);
  } else {
    lastLandmarks = null;
    recognizer.step(null, now);
  }
  sampleTimes.push(now);
  if (sampleTimes.length > 30) sampleTimes.shift();
  renderPanel();
}

async function loop(now) {
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;

  vctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);          
  const interval = 1000 / CONFIG.SAMPLE_FPS;
  if (!busy && now >= nextSample) {
    nextSample = Math.max(nextSample + interval, now - interval);
    busy = true;
    try { await takeSample(now); } catch (e) { console.error(e); setStatus("Hand tracking error: " + e.message, true); }
    busy = false;
  }
  drawOverlay();
}

async function loadModel() {
  const res = await fetch("model.json", { cache: "no-cache" });
  if (!res.ok) throw new Error("model.json not found (run export_web_model.py and upload docs/model.json)");
  return decodeModel(await res.json());
}

async function init() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("This browser cannot use the camera here. Open the page over https in a current browser.", true);
    return;
  }
  try {
    setStatus("Loading the model…");
    recognizer = new Recognizer(await loadModel());
    window.asl = { recognizer };                           
    renderPanel();
    setStatus("Loading hand tracking… (about 10 MB the first time)");
    hands = new window.Hands({ locateFile: (file) => `mediapipe/hands/${file}` });
    hands.setOptions({ maxNumHands: 1, modelComplexity: 0, selfieMode: false,
                       minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    hands.onResults((r) => { latestResults = r; });
    await hands.initialize();
    setStatus("Ready. Click “Start camera” and allow access.");
    $("start").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("Could not start: " + e.message, true);
  }
}

async function startCamera() {
  $("start").disabled = true;
  try {
    setStatus("Waiting for camera permission…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: FRAME_W }, height: { ideal: FRAME_H }, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    running = true;
    $("start").textContent = "Camera on";
    nextSample = performance.now();
    setStatus("Running. Put your hand inside the white box.");
    requestAnimationFrame(loop);
  } catch (e) {
    $("start").disabled = false;
    setStatus(e.name === "NotAllowedError"
      ? "Camera access was blocked. Allow the camera in your browser's address bar and try again."
      : "Could not open the camera: " + e.message, true);
  }
}

$("start").addEventListener("click", startCamera);
$("space").addEventListener("click", () => { recognizer && recognizer.addSpace(); recognizer && renderPanel(); });
$("back").addEventListener("click", () => { recognizer && recognizer.backspace(); recognizer && renderPanel(); });
$("clear").addEventListener("click", () => { recognizer && recognizer.clearAll(); recognizer && renderPanel(); });
$("speak").addEventListener("click", () => {
  const text = recognizer ? recognizer.text.trim() : "";
  if (!text) return;
  if (!("speechSynthesis" in window)) { setStatus("Speech is not supported in this browser.", true); return; }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
});

init();
