// Recognition logic for the browser version. No DOM, no camera: it can be tested in Node.
// It is a line-by-line port of asl_model.py (the LSTM) and app_local.py (the typing rules).

export const CONFIG = {
  SEQUENCE_LENGTH: 20,        // frames the LSTM looks at (must match training)
  SAMPLE_FPS: 15,             // samples per second (must match training clips)
  KEYPOINTS: 63,              // 21 landmarks x (x, y, z)
  NO_SIGN: "NOSIGN",
  SMOOTH_SAMPLES: 4,          // average this many recent predictions (cuts flicker)
  STABLE_SAMPLES: 6,          // same still letter this many samples in a row (~0.4 s)
  THRESHOLD: 0.80,            // minimum averaged confidence for a still letter
  NO_HAND_RESET: 5,           // samples without a hand before everything resets
  MIN_HAND_SHARE: 0.70,       // share of the window that must contain a hand
  MOTION_LETTERS: ["J", "Z"],
  MOTION_THRESHOLD: 0.90,     // J / Z need higher confidence...
  MOTION_STABLE_SAMPLES: 2,   // ...but only for this many samples in a row
  // If J / Z is recognised right after its starting handshape was typed, replace it.
  START_POSE_FIX: { J: ["I"], Z: ["D", "G", "L"] },
  FIX_WINDOW_MS: 1600,
};

// ---------------------------------------------------------------- model --

function decodeFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** model.json (made by export_web_model.py) -> { actions, w } */
export function decodeModel(json) {
  const w = {};
  for (const [name, entry] of Object.entries(json.weights)) {
    const data = decodeFloat32(entry.b64);
    const size = entry.shape.reduce((a, b) => a * b, 1);
    if (data.length !== size) throw new Error(`Weight ${name} has the wrong size`);
    w[name] = { data, shape: entry.shape };
  }
  return { actions: json.actions, w };
}

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/** Keras LSTM forward pass (gate order: input, forget, cell, output). */
function lstm(x, T, inDim, K, R, b, returnSequences) {
  const units = R.shape[0];
  const u4 = 4 * units;
  const Kd = K.data, Rd = R.data;
  // z[t] = x[t] @ K + b
  const z = new Float32Array(T * u4);
  for (let t = 0; t < T; t++) {
    const zt = t * u4;
    for (let j = 0; j < u4; j++) z[zt + j] = b.data[j];
    for (let i = 0; i < inDim; i++) {
      const xi = x[t * inDim + i];
      const row = i * u4;
      for (let j = 0; j < u4; j++) z[zt + j] += xi * Kd[row + j];
    }
  }
  const h = new Float32Array(units);
  const c = new Float32Array(units);
  const g = new Float32Array(u4);
  const out = returnSequences ? new Float32Array(T * units) : null;
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < u4; j++) g[j] = z[t * u4 + j];
    for (let i = 0; i < units; i++) {
      const hi = h[i];
      const row = i * u4;
      for (let j = 0; j < u4; j++) g[j] += hi * Rd[row + j];
    }
    for (let k = 0; k < units; k++) {
      const gi = sigmoid(g[k]);
      const gf = sigmoid(g[units + k]);
      const gc = Math.tanh(g[2 * units + k]);
      const go = sigmoid(g[3 * units + k]);
      c[k] = gf * c[k] + gi * gc;
      h[k] = go * Math.tanh(c[k]);
    }
    if (returnSequences) out.set(h, t * units);
  }
  return returnSequences ? out : h;
}

function dense(x, W, b, relu) {
  const [inDim, outDim] = W.shape;
  const y = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) y[j] = b.data[j];
  for (let i = 0; i < inDim; i++) {
    const xi = x[i];
    const row = i * outDim;
    for (let j = 0; j < outDim; j++) y[j] += xi * W.data[row + j];
  }
  if (relu) for (let j = 0; j < outDim; j++) if (y[j] < 0) y[j] = 0;
  return y;
}

/** sequence: Float32Array of length T*63 -> Float32Array of class probabilities. */
export function predictProbs(model, sequence, T = CONFIG.SEQUENCE_LENGTH) {
  const w = model.w;
  let x = lstm(sequence, T, CONFIG.KEYPOINTS, w.lstm0_kernel, w.lstm0_recurrent, w.lstm0_bias, true);
  x = lstm(x, T, w.lstm0_recurrent.shape[0], w.lstm1_kernel, w.lstm1_recurrent, w.lstm1_bias, true);
  x = lstm(x, T, w.lstm1_recurrent.shape[0], w.lstm2_kernel, w.lstm2_recurrent, w.lstm2_bias, false);
  x = dense(x, w.dense0_kernel, w.dense0_bias, true);
  x = dense(x, w.dense1_kernel, w.dense1_bias, true);
  const logits = dense(x, w.dense2_kernel, w.dense2_bias, false);
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] - max); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

export const displayName = (label) => (label === "NOSIGN" ? "(no sign)" : label);

function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

// ----------------------------------------------------------- recognizer --

export class Recognizer {
  constructor(model, config = CONFIG) {
    this.model = model;
    this.cfg = config;
    this.actions = model.actions;
    this.text = "";
    this.resetTracking();
    this.top = [];
    this.noHand = 0;
    this.lastTypedAt = 0;
  }

  resetTracking() {
    this.buffer = [];        // last SEQUENCE_LENGTH keypoint frames
    this.present = [];       // was a hand found in each of them
    this.history = [];       // recent probability vectors
    this.recent = [];        // recent winning still letters
    this.motionRecent = [];  // recent J / Z hits
    this.lastLetter = null;
  }

  get bufferLength() { return this.buffer.length; }

  typeLetter(letter, nowMs) {
    const fix = this.cfg.START_POSE_FIX[letter];
    if (this.text && fix && fix.includes(this.text[this.text.length - 1])
        && nowMs - this.lastTypedAt < this.cfg.FIX_WINDOW_MS) {
      this.text = this.text.slice(0, -1);
    }
    this.text += letter;
    this.lastLetter = letter;
    this.lastTypedAt = nowMs;
  }

  /** Run once per sample. keypoints: array of 63 numbers, or null when no hand was found. */
  step(keypoints, nowMs) {
    const c = this.cfg;
    const found = keypoints !== null;
    this.buffer.push(found ? keypoints : new Array(c.KEYPOINTS).fill(0));
    this.present.push(found);
    if (this.buffer.length > c.SEQUENCE_LENGTH) { this.buffer.shift(); this.present.shift(); }

    if (!found) {
      this.noHand += 1;
      if (this.noHand >= c.NO_HAND_RESET) { this.resetTracking(); this.top = []; }
      return;
    }
    this.noHand = 0;

    const handFrames = this.present.filter(Boolean).length;
    if (this.buffer.length < c.SEQUENCE_LENGTH || handFrames < c.MIN_HAND_SHARE * c.SEQUENCE_LENGTH) return;

    const flat = new Float32Array(c.SEQUENCE_LENGTH * c.KEYPOINTS);
    this.buffer.forEach((frame, t) => flat.set(frame, t * c.KEYPOINTS));
    const raw = predictProbs(this.model, flat, c.SEQUENCE_LENGTH);

    this.history.push(raw);
    if (this.history.length > c.SMOOTH_SAMPLES) this.history.shift();
    const probs = new Float32Array(raw.length);
    for (const h of this.history) for (let i = 0; i < probs.length; i++) probs[i] += h[i] / this.history.length;

    const best = argmax(probs);
    this.top = Array.from(probs.keys())
      .sort((a, b) => probs[b] - probs[a]).slice(0, 3)
      .map((i) => [displayName(this.actions[i]), probs[i]]);

    // Motion letters (J, Z) only match for a few samples while the movement fills the
    // window, so they are judged on the raw prediction, not the smoothed one.
    const rawBest = argmax(raw);
    if (c.MOTION_LETTERS.includes(this.actions[rawBest]) && raw[rawBest] >= c.MOTION_THRESHOLD) {
      this.motionRecent.push(rawBest);
      if (this.motionRecent.length > c.MOTION_STABLE_SAMPLES) this.motionRecent.shift();
    } else {
      this.motionRecent = [];
    }
    if (this.motionRecent.length === c.MOTION_STABLE_SAMPLES
        && new Set(this.motionRecent).size === 1
        && this.actions[rawBest] !== this.lastLetter) {
      this.typeLetter(this.actions[rawBest], nowMs);
      this.motionRecent = [];
      this.recent = [];
      return;
    }

    // Still letters: smoothed and held steady.
    if (this.actions[best] === c.NO_SIGN || c.MOTION_LETTERS.includes(this.actions[best])
        || probs[best] < c.THRESHOLD) {
      this.recent = [];
      return;
    }
    this.recent.push(best);
    if (this.recent.length > c.STABLE_SAMPLES) this.recent.shift();
    if (this.recent.length === c.STABLE_SAMPLES && new Set(this.recent).size === 1
        && this.actions[best] !== this.lastLetter) {
      this.typeLetter(this.actions[best], nowMs);
    }
  }

  addSpace() { this.text += " "; this.lastLetter = null; }
  backspace() { this.text = this.text.slice(0, -1); this.lastLetter = null; }
  clearAll() { this.text = ""; this.resetTracking(); this.noHand = 0; }
}
