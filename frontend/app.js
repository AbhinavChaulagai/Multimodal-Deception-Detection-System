/**
 * Deception Detection — frontend logic
 *
 * Flow:
 *   1. User clicks "Enable Camera & Mic"  → getUserMedia
 *   2. User clicks "Record"               → MediaRecorder starts
 *   3. User clicks "Stop & Analyse"       → POST video blob to /predict
 *   4. Display score, label, transcript
 *
 * Set API_BASE to your Render backend URL before deploying.
 */

// ─── Configuration ────────────────────────────────────────────────────
const API_BASE = "https://YOUR-APP-NAME.onrender.com";  // ← update after Render deploy
const MAX_RECORD_SECONDS = 60;

// ─── DOM refs ─────────────────────────────────────────────────────────
const preview       = document.getElementById("preview");
const overlay       = document.getElementById("overlay");
const overlayText   = document.getElementById("overlay-text");
const timerEl       = document.getElementById("timer");

const btnStartCam   = document.getElementById("btn-start-cam");
const btnRecord     = document.getElementById("btn-record");
const btnStop       = document.getElementById("btn-stop");
const btnRetry      = document.getElementById("btn-retry");
const btnRetryErr   = document.getElementById("btn-retry-err");

const cameraCard    = document.getElementById("camera-card");
const loadingCard   = document.getElementById("loading-card");
const resultCard    = document.getElementById("result-card");
const errorCard     = document.getElementById("error-card");
const demoBanner    = document.getElementById("demo-banner");

const resultLabel   = document.getElementById("result-label");
const scorePct      = document.getElementById("score-pct");
const gaugeFill     = document.getElementById("gauge-fill");
const confidenceEl  = document.getElementById("confidence");
const transcriptEl  = document.getElementById("transcript-text");
const errorMsg      = document.getElementById("error-msg");

// ─── State ────────────────────────────────────────────────────────────
let stream      = null;
let recorder    = null;
let chunks      = [];
let timerTick   = null;
let elapsed     = 0;
let demoMode    = false;

// ─── Health check on load ─────────────────────────────────────────────
(async () => {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.demo_mode) {
      demoMode = true;
      demoBanner.classList.remove("hidden");
    }
  } catch (_) {
    // backend unreachable — let the predict call surface the error
  }
})();

// ─── Helpers ──────────────────────────────────────────────────────────

function showOnly(card) {
  [cameraCard, loadingCard, resultCard, errorCard].forEach(c => c.classList.add("hidden"));
  card.classList.remove("hidden");
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function setGauge(score) {
  // score 0→1  maps to rotation -90deg (left) → +90deg (right)
  const deg = (score * 180) - 90;
  gaugeFill.style.setProperty("--angle", `${deg}deg`);
  scorePct.textContent = `${Math.round(score * 100)}%`;
}

// Pick best supported MIME type for MediaRecorder
function bestMime() {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4",
  ];
  return candidates.find(t => MediaRecorder.isTypeSupported(t)) || "";
}

// ─── Camera / mic ──────────────────────────────────────────────────────

btnStartCam.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    preview.srcObject = stream;
    overlay.classList.add("hidden");

    btnStartCam.hidden = true;
    btnRecord.hidden   = false;
  } catch (err) {
    alert(`Camera / microphone access denied.\n${err.message}`);
  }
});

// ─── Record ────────────────────────────────────────────────────────────

btnRecord.addEventListener("click", () => {
  chunks  = [];
  elapsed = 0;

  const mime = bestMime();
  recorder   = new MediaRecorder(stream, mime ? { mimeType: mime } : {});

  recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  recorder.onstop          = handleStop;
  recorder.start(200);   // collect in 200 ms chunks

  btnRecord.hidden = true;
  btnStop.hidden   = false;
  timerEl.hidden   = false;

  timerTick = setInterval(() => {
    elapsed++;
    timerEl.textContent = fmtTime(elapsed);
    if (elapsed >= MAX_RECORD_SECONDS) btnStop.click();
  }, 1000);
});

// ─── Stop & submit ─────────────────────────────────────────────────────

btnStop.addEventListener("click", () => {
  clearInterval(timerTick);
  timerEl.hidden = true;
  btnStop.hidden = true;
  if (recorder && recorder.state !== "inactive") recorder.stop();
});

async function handleStop() {
  const mime = recorder.mimeType || "video/webm";
  const blob = new Blob(chunks, { type: mime });
  chunks     = [];

  if (demoMode) {
    errorMsg.textContent = "Live inference is disabled in the web demo — the model exceeds free-tier memory limits. Clone the repo and run the backend locally for full analysis.";
    showOnly(errorCard);
    return;
  }

  showOnly(loadingCard);

  try {
    const form = new FormData();
    const ext  = mime.includes("mp4") ? "mp4" : "webm";
    form.append("file", blob, `recording.${ext}`);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body:   form,
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    displayResult(data);

  } catch (err) {
    errorMsg.textContent = `Error: ${err.message}`;
    showOnly(errorCard);
  }
}

// ─── Display result ────────────────────────────────────────────────────

function displayResult(data) {
  const score    = data.score;          // 0–1
  const isDecep  = data.label === "Deceptive";

  resultLabel.textContent = data.label;
  resultLabel.className   = "result-label " + (isDecep ? "deceptive" : "truthful");

  setGauge(score);
  confidenceEl.textContent  = data.confidence;
  transcriptEl.textContent  = data.transcript || "(no speech detected)";

  showOnly(resultCard);
}

// ─── Retry ─────────────────────────────────────────────────────────────

function resetUI() {
  showOnly(cameraCard);
  btnRecord.hidden = stream ? false : true;
  btnStop.hidden   = true;
  timerEl.hidden   = true;
  overlayText.textContent = stream ? "" : "Camera off";
  if (!stream) overlay.classList.remove("hidden");
}

btnRetry.addEventListener("click", resetUI);
btnRetryErr.addEventListener("click", resetUI);
