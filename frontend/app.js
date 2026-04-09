/* ════════════════════════════════════════════════════════════
   DoshaNet v2 — Frontend Application Logic
   ════════════════════════════════════════════════════════════ */

const API = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? "http://localhost:8000"
  : window.location.origin;

// Supabase Init
const SUPABASE_URL = "https://bfqvsfzglvjyscivwavt.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJmcXZzZnpnbHZqeXNjaXZ3YXZ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUxODEyNzMsImV4cCI6MjA5MDc1NzI3M30.Wt48G-7jGJxLCQsfrq3bedbBqvhwdjNczAh--pfhFl8";
const supabaseClient = (window.supabase && window.supabase.createClient)
  ? window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
  : null;

const DOSHA = {
  Vata: {
    emoji: "🌬️",
    color: "#8b5cf6",
    desc: "Air & Space — Creative, quick, light. Tends toward dryness, variability, and anxiety.",
    diet: "Warm, cooked foods, healthy fats (ghee), sweet/sour/salty tastes.",
    herbs: "Ashwagandha (stress), Triphala (digestion), Ginger.",
    lifestyle: "Regular routine, warm oil massage (Abhyanga), grounding yoga."
  },
  Pitta: {
    emoji: "🔥",
    color: "#f97316",
    desc: "Fire & Water — Focused, intelligent, intense. Tends toward heat, acidity, and irritability.",
    diet: "Cooling foods, fresh vegetables, sweet/bitter/astringent tastes.",
    herbs: "Shatavari (cooling), Brahmi (mental focus), Aloe Vera.",
    lifestyle: "Avoid midday sun, practice moderation, cooling breathwork (Sitali)."
  },
  Kapha: {
    emoji: "🌱",
    color: "#10b981",
    desc: "Earth & Water — Steady, calm, strong. Tends toward lethargy, congestion, and attachment.",
    diet: "Light, spicy, warm foods, avoiding heavy sweets and dairy.",
    herbs: "Trikatu (spices for metabolism), Turmeric, Tulsi (Holy Basil).",
    lifestyle: "Early wake times, vigorous exercise, dry skin brushing (Garshana)."
  }
};

// ── State ────────────────────────────────────────────────────
let state = {
  imageFile:    null,
  imageBytes:   null,
  faceMeasured: false,
  faceRatio:    null,
  quizState:    null,
  currentQ:     null,
  prediction:   null,
  confidence:   null,
  quizAnswers:  {},    // question_idx → answer
  confChart:    null,
  mediapipe:    null,
  camera:       null,
  webcamActive: false,
  capturedCanvas: null,
  imageUrl: null,
};

// ── DOM helpers ──────────────────────────────────────────────
const $ = id => document.getElementById(id);

function showToast(msg, type="warn") {
  const t = document.createElement("div");
  t.className = "toast";
  
  const icon = document.createElement("span");
  icon.textContent = type === "ok" ? "✅" : "⚠️";
  
  const text = document.createElement("span");
  text.textContent = msg;

  t.appendChild(icon);
  t.appendChild(text);

  $("toast-container").appendChild(t);
  setTimeout(()=>t.remove(), 5000);
}

// ── Hero orb canvas animation ────────────────────────────────
(function animateOrbs() {
  const cvs = $("dosha-orb-canvas");
  const ctx = cvs.getContext("2d");
  const W = cvs.width, H = cvs.height;
  const orbs = [
    { x:W*0.5, y:H*0.28, r:70, color:"#8b5cf6", phase:0 },
    { x:W*0.68,y:H*0.65, r:65, color:"#f97316", phase:2.1 },
    { x:W*0.32,y:H*0.65, r:65, color:"#14b8a6", phase:4.2 },
  ];
  const labels = ["V","P","K"];
  let t = 0;
  function draw() {
    ctx.clearRect(0,0,W,H);
    orbs.forEach((o,i) => {
      const ox = o.x + Math.sin(t*0.6 + o.phase)*14;
      const oy = o.y + Math.cos(t*0.4 + o.phase)*10;
      const g = ctx.createRadialGradient(ox, oy, 10, ox, oy, o.r);
      g.addColorStop(0, o.color+"BB");
      g.addColorStop(1, o.color+"11");
      ctx.beginPath();
      ctx.arc(ox, oy, o.r, 0, Math.PI*2);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.strokeStyle = o.color+"55";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = o.color;
      ctx.font = "bold 28px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(labels[i], ox, oy);
    });
    t += 0.025;
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── API Health ───────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const dot = $("api-status"), lbl = $("api-label");
    if (r.ok) {
      const d = await r.json();
      dot.className = "status-dot online";
      lbl.textContent = `v${d.version} online`;
    } else {
      dot.className = "status-dot offline"; lbl.textContent = "offline";
    }
  } catch { $("api-status").className = "status-dot offline"; $("api-label").textContent = "offline"; }
}
checkHealth();
setInterval(checkHealth, 15000);

// ── Step management ──────────────────────────────────────────
function goStep(n) {
  ["face","quiz","results"].forEach((p,i) => {
    const el = $(`panel-${p}`);
    el.classList.toggle("hidden", i+1 !== n);
    const ind = $(`step-ind-${i+1}`);
    if (ind) {
      ind.classList.remove("active","complete");
      if (i+1 === n) ind.classList.add("active");
      if (i+1 < n)   ind.classList.add("complete");
    }
  });
}

// ── Mode switching (upload / webcam) ─────────────────────────
window.switchMode = function switchMode(mode) {
  $("upload-mode").classList.toggle("hidden", mode !== "upload");
  $("webcam-mode").classList.toggle("hidden", mode !== "webcam");
  $("tab-upload").classList.toggle("active", mode === "upload");
  $("tab-webcam").classList.toggle("active", mode === "webcam");
  if (mode === "webcam") initWebcam();
  else stopWebcam();
};

// ── Upload & drop zone ───────────────────────────────────────
const dropZone  = $("drop-zone");
const imgInput  = $("image-input");
const preview   = $("preview-img");
const dropCont  = $("drop-content");

dropZone.addEventListener("click", ()=>imgInput.click());
dropZone.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key===" ") imgInput.click(); });
imgInput.addEventListener("change", ()=>handleFile(imgInput.files[0]));
dropZone.addEventListener("dragover", e=>{ e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", ()=>dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e=>{ e.preventDefault(); dropZone.classList.remove("dragover"); handleFile(e.dataTransfer.files[0]); });

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) { showToast("Please select an image file."); return; }
  state.imageFile = file;
  const reader = new FileReader();
  reader.onload = async e => {
    state.imageBytes = e.target.result;
    preview.src = e.target.result;
    preview.classList.remove("hidden");
    dropCont.classList.add("hidden");
    analyzeFaceFromImage(e.target.result);
  };
  reader.readAsDataURL(file);
}

// ── MediaPipe face analysis ───────────────────────────────────
function analyzeFaceFromImage(dataUrl) {
  $("metrics-note").textContent = "Analyzing face with MediaPipe FaceMesh…";

  const img = new Image();
  img.onload = () => {
    try {
      const faceMesh = new FaceMesh({ locateFile: f =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
      faceMesh.setOptions({
        maxNumFaces: 1, refineLandmarks: false,
        minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      faceMesh.onResults(results => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const lm = results.multiFaceLandmarks[0];
          extractFaceMetrics(lm, img);
        } else {
          $("metrics-note").textContent = "No face detected — metrics estimated from image colors.";
          estimateSkinTone(img);
          enableQuizStart();
        }
      });
      // Send the image to FaceMesh
      const cvs = document.createElement("canvas");
      cvs.width = img.naturalWidth; cvs.height = img.naturalHeight;
      cvs.getContext("2d").drawImage(img, 0, 0);
      faceMesh.send({ image: cvs }).catch(err => {
        console.warn("MediaPipe send error:", err);
        estimateSkinTone(img);
        enableQuizStart();
      });
    } catch(e) {
      console.warn("MediaPipe init error:", e);
      estimateSkinTone(img);
      enableQuizStart();
    }
  };
  img.src = dataUrl;
}

function extractFaceMetrics(landmarks, img) {
  // Key landmark indices for face geometry
  // Left cheek: 234, Right cheek: 454, Forehead: 10, Chin: 152
  const lm = landmarks;
  const leftCheek  = lm[234];
  const rightCheek = lm[454];
  const forehead   = lm[10];
  const chin       = lm[152];

  const faceW = Math.abs(rightCheek.x - leftCheek.x);
  const faceH = Math.abs(chin.y - forehead.y);
  const ratio = faceH > 0.01 ? Math.min(faceW / faceH, 1.0) : 0.5;
  const normRatio = Math.max(0, Math.min(1, (ratio - 0.4) / 0.65));

  // Symmetry: average absolute difference between mirrored landmarks
  const pairIndices = [[33,263],[133,362],[70,300],[105,334],[4,4]];
  let symSum = 0;
  pairIndices.forEach(([l,r]) => {
    symSum += Math.abs(Math.abs(lm[l].x - 0.5) - Math.abs(lm[r].x - 0.5));
  });
  const symScore = Math.max(0, 1 - symSum * 8);

  state.faceRatio = normRatio;
  state.faceMeasured = true;

  // Update UI
  $("bar-ratio").style.width = (normRatio * 100) + "%";
  $("val-ratio").textContent = normRatio.toFixed(2);
  $("bar-sym").style.width = (symScore * 100) + "%";
  $("val-sym").textContent = symScore.toFixed(2);
  $("metrics-note").textContent = `✅ FaceMesh: ${landmarks.length} landmarks detected. Face ratio auto-fills questionnaire.`;

  estimateSkinTone(img);
  enableQuizStart();
}

function estimateSkinTone(img) {
  try {
    const cvs = document.createElement("canvas");
    cvs.width = 40; cvs.height = 40;
    const ctx = cvs.getContext("2d");
    ctx.drawImage(img, img.naturalWidth*0.3, img.naturalHeight*0.2,
                       img.naturalWidth*0.4, img.naturalHeight*0.4, 0, 0, 40, 40);
    const data = ctx.getImageData(0,0,40,40).data;
    let r=0,g=0,b=0,n=0;
    for(let i=0;i<data.length;i+=4){r+=data[i];g+=data[i+1];b+=data[i+2];n++;}
    r=Math.round(r/n); g=Math.round(g/n); b=Math.round(b/n);
    $("skin-swatch").style.background = `rgb(${r},${g},${b})`;
    $("val-skin").textContent = `rgb(${r},${g},${b})`;
  } catch(e) { $("val-skin").textContent = "—"; }
}

function enableQuizStart() {
  $("start-quiz-btn").disabled = false;
}

// ── Webcam ───────────────────────────────────────────────────
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width:320, height:320 } });
    const video = $("webcam-video");
    video.srcObject = stream;
    state.webcamActive = true;
    $("webcam-status").textContent = "Camera active — face in frame";
    $("capture-btn").disabled = false;
    state.webcamStream = stream;

    // Start MediaPipe on webcam
    const faceMesh = new FaceMesh({ locateFile: f =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
    faceMesh.setOptions({ maxNumFaces:1, minDetectionConfidence:0.5 });
    faceMesh.onResults(results => {
      const cvs = $("webcam-canvas");
      const octx = cvs.getContext("2d");
      cvs.width = video.videoWidth; cvs.height = video.videoHeight;
      octx.clearRect(0,0,cvs.width,cvs.height);
      if (results.multiFaceLandmarks) {
        for (const lm of results.multiFaceLandmarks) {
          drawConnectors(octx, lm, FACEMESH_TESSELATION, {color:"#8b5cf644", lineWidth:1});
          drawLandmarks(octx, lm, {color:"#8b5cf6", radius:1});
        }
      }
    });
    state.mediapipe = faceMesh;
    if (window.Camera) {
      state.camera = new Camera(video, {
        onFrame: async () => { await faceMesh.send({ image: video }); },
        width:320, height:320,
      });
      state.camera.start();
    }
  } catch(e) {
    $("webcam-status").textContent = "Camera error: " + e.message;
    showToast("Could not access camera: " + e.message);
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(t=>t.stop());
    state.webcamStream = null;
  }
  if (state.camera) { state.camera.stop(); state.camera = null; }
  state.webcamActive = false;
}

window.captureWebcam = function() {
  const video = $("webcam-video");
  const cvs = document.createElement("canvas");
  cvs.width = video.videoWidth; cvs.height = video.videoHeight;
  cvs.getContext("2d").drawImage(video, 0, 0);
  cvs.toBlob(blob => {
    const file = new File([blob], "webcam.jpg", { type:"image/jpeg" });
    state.imageFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      state.imageBytes = e.target.result;
      switchMode("upload");
      preview.src = e.target.result;
      preview.classList.remove("hidden");
      dropCont.classList.add("hidden");
      analyzeFaceFromImage(e.target.result);
    };
    reader.readAsDataURL(blob);
  }, "image/jpeg", 0.92);
};

// ── Skip face analysis ───────────────────────────────────────
window.skipFaceAnalysis = function() {
  state.faceMeasured = false;
  state.faceRatio = null;
  startQuiz();
};

// ── Quiz ─────────────────────────────────────────────────────
window.startQuiz = async function() {
  const btn = $("start-quiz-btn");
  btn.disabled = true;
  $("start-btn-text").textContent = "Starting quiz…";
  $("start-spinner").classList.remove("hidden");

  const preAnswered = {};
  if (state.faceMeasured && state.faceRatio !== null) {
    preAnswered[9] = state.faceRatio;   // face_width_ratio
  }

  try {
    const resp = await fetch(`${API}/quiz/start`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ pre_answered: preAnswered }),
    });
    const data = await resp.json();
    state.quizState   = data.state;
    state.currentQ    = data.question;
    state.quizAnswers = { ...preAnswered };

    goStep(2);
    displayQuestion(data.question, data.state);
  } catch(e) {
    showToast("Failed to start quiz: " + e.message);
    btn.disabled = false;
    $("start-btn-text").textContent = "➡ Start Adaptive Quiz";
    $("start-spinner").classList.add("hidden");
  }
};

function displayQuestion(q, qs) {
  $("question-label").textContent = q.label;
  $("scale-low").textContent  = q.low;
  $("scale-high").textContent = q.high;
  $("quiz-slider").value = 50;
  updateSliderDisplay(0.5, q);
  updatePosterior(qs.posterior, qs.entropy);

  const n = qs.n_answered;
  const maxQ = 7;
  const pct  = (n / maxQ) * 100;
  $("quiz-progress-bar").style.width = pct + "%";
  $("quiz-progress-label").textContent = `Question ${n+1} of ≤${maxQ}`;

  // Re-animate card
  const card = $("question-card");
  card.style.animation = "none";
  void card.offsetHeight;
  card.style.animation = "slideIn 0.3s ease";
}

function updateSliderDisplay(val, q) {
  const pct = val * 100;
  $("quiz-slider").style.setProperty("--pct", pct + "%");
  $("slider-num-live").textContent = val.toFixed(2);
  // Semantic label
  if (val < 0.33)      $("slider-label-live").textContent = q ? q.low.split("—")[0].trim()  : "Low";
  else if (val > 0.67) $("slider-label-live").textContent = q ? q.high.split("—")[0].trim() : "High";
  else                 $("slider-label-live").textContent = "Moderate";
}

// Slider input
$("quiz-slider").addEventListener("input", function() {
  updateSliderDisplay(this.value/100, state.currentQ);
});

function updatePosterior(posterior, entropy) {
  const labels = ["vata","pitta","kapha"];
  labels.forEach((d,i) => {
    const pct = Math.round(posterior[i]*100);
    $(`post-bar-${d}`).style.width  = pct + "%";
    $(`post-pct-${d}`).textContent  = pct + "%";
  });
  $("entropy-val").textContent = entropy.toFixed(3) + " bits";
  // Max entropy = log2(3) = 1.585 bits
  $("entropy-bar").style.width = ((entropy / 1.585) * 100) + "%";
}

window.submitAnswer = async function() {
  const answer = $("quiz-slider").value / 100;
  const qIdx   = state.currentQ.idx;

  $("quiz-next-btn").disabled = true;

  try {
    const resp = await fetch(`${API}/quiz/next`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        state:        state.quizState,
        question_idx: qIdx,
        answer:       answer,
      }),
    });
    
    if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || `HTTP Error ${resp.status}`);
    }

    const data = await resp.json();
    state.quizState        = data.state;
    state.quizAnswers[qIdx] = answer;
    updatePosterior(data.state.posterior, data.state.entropy);

    if (data.done) {
      await fetchRealPrediction(data);
    } else {
      state.currentQ = data.question;
      displayQuestion(data.question, data.state);
    }
  } catch(e) {
    showToast("Quiz error: " + e.message);
  } finally {
    $("quiz-next-btn").disabled = false;
  }
};

// ── Results ───────────────────────────────────────────────────
async function showResults(quizData) {
  goStep(3);

  const pred = quizData.prediction;
  const conf = quizData.confidence;
  const info = DOSHA[pred];

  // Prediction card
  $("pred-emoji").textContent = info.emoji;
  $("pred-name").textContent  = pred;
  $("pred-name").className    = "pred-name " + pred.toLowerCase();
  $("pred-desc").textContent  = info.desc;
  $("unc-badge").style.display = "none";

  // Recommendations
  $("recom-diet").textContent      = info.diet;
  $("recom-herbs").textContent     = info.herbs;
  $("recom-lifestyle").textContent = info.lifestyle;

  // Ternary triangle
  drawTernaryTriangle(
    conf["Vata"]/100, conf["Pitta"]/100, conf["Kapha"]/100,
    info.color
  );

  // Confidence chart
  drawConfidenceChart(conf);

  // SHAP explanation
  renderExplanation(quizData.explanation || []);

  // Update uncertainty panel if mlData exists
  if (quizData.mlData) {
    const d = quizData.mlData;
    const ep  = Math.min(d.epistemic * 1000, 100);
    const al  = Math.min((d.aleatoric / 1.585) * 100, 100);
    $("unc-epistemic-bar").style.width = ep + "%";
    $("unc-aleatoric-bar").style.width = al + "%";
    $("unc-epistemic-val").textContent = d.epistemic.toFixed(5);
    $("unc-aleatoric-val").textContent = d.aleatoric.toFixed(3) + " nats";
    $("mc-note").textContent = `MC-Dropout T=50 | Level: ${d.uncertainty_level}`;

    const badge = $("unc-badge");
    badge.style.display = "flex";
    const lv = $("unc-level");
    lv.textContent = d.uncertainty_level;
    lv.className = "unc-level " + d.uncertainty_level;

    drawTernaryTriangle(conf["Vata"]/100, conf["Pitta"]/100, conf["Kapha"]/100, info.color, d.epistemic);
  } else {
    $("unc-badge").style.display = "none";
    $("mc-note").textContent = "ML Inference failed — MC-Dropout skipped";
    drawTernaryTriangle(conf["Vata"]/100, conf["Pitta"]/100, conf["Kapha"]/100, info.color, 0);
  }

  // Confidence chart
  drawConfidenceChart(conf);

  // Run GradCAM
  runGradCAM(pred);
}

async function fetchRealPrediction(quizData) {
  $("mc-note").textContent = "🔄 Running Multimodal Inference...";
  const btn = $("quiz-next-btn");
  btn.disabled = true;
  btn.textContent = "Analyzing...";

  try {
    const feat = buildFeatureArray();
    const fd   = new FormData();
    if (state.imageBytes) {
      fd.append("image_b64", state.imageBytes);
    }
    fd.append("features", JSON.stringify(feat));

    const resp = await fetch(`${API}/predict/uncertainty`, { method:"POST", body:fd });
    if (!resp.ok) throw new Error("Inference failed");
    
    const mlData = await resp.json();
    
    // Override quiz result with REAL ML result
    quizData.prediction = mlData.prediction;
    quizData.confidence = mlData.confidence;
    quizData.explanation = mlData.explanation;
    quizData.mlData = mlData; 
    
  } catch (err) {
    console.warn("ML Model failed, using quiz fallback", err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Next Question →";
    state.prediction = quizData.prediction;
    state.confidence = quizData.confidence;
    await showResults(quizData);
  }
}

async function runGradCAM(prediction) {
  if (!state.imageBytes) {
    $("gradcam-placeholder").innerHTML = "<div>📷</div><div>Upload face image for GradCAM</div>";
    return;
  }
  $("gradcam-placeholder").innerHTML = "<div>🔄</div><div>Generating GradCAM heatmap...</div>";

  const classIdx = ["Vata","Pitta","Kapha"].indexOf(prediction);
  try {
    const feat = buildFeatureArray();
    const fd   = new FormData();
    fd.append("image_b64", state.imageBytes);
    fd.append("features", JSON.stringify(feat));
    fd.append("target_class", classIdx);

    const resp = await fetch(`${API}/gradcam`, { method:"POST", body:fd });
    if (!resp.ok) return;
    const d = await resp.json();

    $("gradcam-img").src = "data:image/jpeg;base64," + d.heatmap_b64;
    $("gradcam-img").classList.remove("hidden");
    $("gradcam-placeholder").classList.add("hidden");
  } catch(e) {
    $("gradcam-placeholder").innerHTML = "<div>⚠️</div><div>GradCAM generation failed</div>";
    console.warn("GradCAM failed:", e);
  }
}

function buildFeatureArray() {
  // Fill from quiz answers; default 0.5 for unanswered
  return Array.from({length:10}, (_,i) =>
    state.quizAnswers[i] !== undefined ? state.quizAnswers[i] : 0.5
  );
}

// ── Confidence chart ─────────────────────────────────────────
function drawConfidenceChart(conf) {
  const labels = Object.keys(conf);
  const values = Object.values(conf);
  const colors = labels.map(l => DOSHA[l].color);

  if (state.confChart) state.confChart.destroy();
  state.confChart = new Chart($("confidence-chart").getContext("2d"), {
    type:"bar",
    data:{
      labels,
      datasets:[{
        data:values,
        backgroundColor: colors.map(c=>c+"44"),
        borderColor:     colors,
        borderWidth:2, borderRadius:8, barPercentage:0.55,
      }],
    },
    options:{
      responsive:true,
      plugins:{ legend:{display:false} },
      scales:{
        y:{ min:0,max:100,
            ticks:{color:"#94a3b8",callback:v=>`${v}%`},
            grid:{color:"rgba(255,255,255,0.05)"}},
        x:{ ticks:{color:"#eef2ff",font:{size:13,weight:"600"}},
            grid:{display:false}},
      },
    },
  });
}

// ── Ternary plot (pure canvas) ────────────────────────────────
function drawTernaryTriangle(vata, pitta, kapha, pointColor, epistemic=0) {
  const cvs = $("ternary-canvas");
  const ctx = cvs.getContext("2d");
  const W = cvs.width, H = cvs.height;
  ctx.clearRect(0,0,W,H);

  const mg = 36;
  // Vertices: Vata=top, Pitta=bottom-right, Kapha=bottom-left
  const V = {x:W/2,   y:mg};
  const P = {x:W-mg,  y:H-mg};
  const K = {x:mg,    y:H-mg};

  // Triangle outline
  ctx.beginPath();
  ctx.moveTo(V.x,V.y); ctx.lineTo(P.x,P.y); ctx.lineTo(K.x,K.y); ctx.closePath();
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1.5; ctx.stroke();

  // Grid lines (trilinear grid at 33%)
  const thirds = [1/3, 2/3];
  ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
  thirds.forEach(t => {
    // Lines parallel to each side
    [
      [lerp2(V,P,t), lerp2(V,K,t)],
      [lerp2(V,P,t), lerp2(K,P,t)],
      [lerp2(V,K,t), lerp2(K,P,t)],
    ].forEach(([a,b]) => {
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    });
  });

  // Vertex labels
  [
    {pt:V, label:"V", color:"#8b5cf6"},
    {pt:P, label:"P", color:"#f97316"},
    {pt:K, label:"K", color:"#14b8a6"},
  ].forEach(({pt,label,color}) => {
    const ox = pt===V?0 : pt===P?14:-14;
    const oy = pt===V?-12:12;
    ctx.fillStyle = color; ctx.font = "bold 13px Inter,sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, pt.x+ox, pt.y+oy);
  });

  // Barycentric → cartesian
  const px = vata*V.x + pitta*P.x + kapha*K.x;
  const py = vata*V.y + pitta*P.y + kapha*K.y;

  // Uncertainty ellipse behind dot
  if (epistemic > 0) {
    const r = Math.min(40, epistemic * 800 + 6);
    ctx.beginPath();
    ctx.ellipse(px, py, r, r*0.7, 0, 0, Math.PI*2);
    ctx.fillStyle = pointColor + "22"; ctx.fill();
  }

  // Point
  const g = ctx.createRadialGradient(px, py, 2, px, py, 12);
  g.addColorStop(0, pointColor + "ff");
  g.addColorStop(1, pointColor + "33");
  ctx.beginPath(); ctx.arc(px,py,10,0,Math.PI*2);
  ctx.fillStyle = g; ctx.fill();
  ctx.strokeStyle = pointColor; ctx.lineWidth = 2; ctx.stroke();
}

function lerp2(a,b,t) { return {x:a.x+(b.x-a.x)*t, y:a.y+(b.y-a.y)*t}; }

// ── SHAP explanation ─────────────────────────────────────────
function renderExplanation(items) {
  const container = $("explanation-cards");
  container.innerHTML = "";
  if (!items || !items.length) {
    container.innerHTML = `<p style="color:var(--text-dim);font-size:.85rem">Explanation unavailable.</p>`;
    return;
  }
  items.forEach(e => {
    const card = document.createElement("div");
    card.className = "explain-card";
    card.innerHTML = `
      <div>
        <div class="explain-feature">${e.feature.replace(/_/g," ")}</div>
        <div class="explain-desc">${e.description}</div>
      </div>
      <span class="explain-dir ${e.direction}">${e.direction}</span>
    `;
    container.appendChild(card);
  });
}

// ── Reset ────────────────────────────────────────────────────
// ── Reset ────────────────────────────────────────────────────
window.resetAll = function() {
  state = {
    imageFile:null, imageBytes:null, faceMeasured:false, faceRatio:null,
    quizState:null, currentQ:null, prediction:null, confidence:null,
    quizAnswers:{}, confChart:null, mediapipe:null, camera:null, webcamActive:false,
  };
  // Wait, if deep linked, these elements might not exist.
  try {
    $("preview-img").classList.add("hidden");
    $("drop-content").classList.remove("hidden");
    $("image-input").value = "";
    $("start-quiz-btn").disabled = true;
    $("start-btn-text").textContent = "➡ Start Adaptive Quiz";
    $("start-spinner").classList.add("hidden");
    $("val-ratio").textContent = "—";
    $("val-sym").textContent = "—";
    $("val-skin").textContent = "—";
    $("bar-ratio").style.width = "50%";
    $("bar-sym").style.width = "50%";
    $("metrics-note").textContent = "Upload a face photo to auto-extract metrics.";
    $("gradcam-img").classList.add("hidden");
    $("gradcam-placeholder").classList.remove("hidden");
  } catch(e) {} // Ignore if deep linked without face
  goStep(1);
  if (state.confChart) { state.confChart.destroy(); state.confChart = null; }
  
  // Strip url param
  window.history.replaceState({}, document.title, window.location.pathname);
};

// ── Viral Growth Mechanics ───────────────────────────────────

window.addEventListener('DOMContentLoaded', async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const profileId = urlParams.get('profile');
  if (profileId && supabaseClient) {
    try {
      const { data, error } = await supabaseClient.from('doshanet_profiles').select('payload').eq('id', profileId).single();
      if (!error && data) {
        const payload = data.payload;
        state.prediction = payload.prediction;
        state.confidence = payload.confidence;
        
        goStep(3);
        
        const info = DOSHA[payload.prediction];
        $("pred-emoji").textContent = info.emoji;
        $("pred-name").textContent = payload.prediction;
        $("pred-name").className = "pred-name " + payload.prediction.toLowerCase();
        $("pred-desc").textContent = info.desc;
        $("recom-diet").textContent = info.diet;
        $("recom-herbs").textContent = info.herbs;
        $("recom-lifestyle").textContent = info.lifestyle;
        
        drawConfidenceChart(payload.confidence);
        drawTernaryTriangle(payload.confidence["Vata"]/100, payload.confidence["Pitta"]/100, payload.confidence["Kapha"]/100, info.color, payload.epistemic || 0);
        
        if (payload.gradcamDataUrl && payload.gradcamDataUrl.startsWith('data:image')) {
          $("gradcam-placeholder").classList.add("hidden");
          $("gradcam-img").src = payload.gradcamDataUrl;
          $("gradcam-img").classList.remove("hidden");
        }
        
        showToast("Loaded shared Dosha profile!", "ok");
      }
    } catch(e) {
      console.error(e);
      showToast("Could not load profile.");
    }
  }
});

window.shareProfile = async function() {
  const btn = $("btn-share");
  btn.disabled = true;
  btn.textContent = "⏳ Saving...";
  try {
    const gcImg = $("gradcam-img");
    const payload = {
      prediction: state.prediction,
      confidence: state.confidence,
      gradcamDataUrl: (!gcImg.classList.contains("hidden")) ? gcImg.src : null,
      epistemic: parseFloat($("unc-epistemic-val").textContent) || 0,
      answers: state.quizAnswers
    };
    
    const shortId = Math.random().toString(36).substring(2, 10);
    const { error } = await supabaseClient.from('doshanet_profiles').insert({ id: shortId, payload });
    
    if (!error) {
      const url = new URL(window.location.href);
      url.searchParams.set("profile", shortId);
      try {
        await navigator.clipboard.writeText(url.toString());
        showToast("Link copied to clipboard!", "ok");
      } catch(er) {
        showToast("Profile saved but could not copy to clipboard. ID: " + shortId);
      }
    } else {
      throw error;
    }
  } catch(e) {
    console.error(e);
    showToast("Failed to share profile.");
  } finally {
    btn.disabled = false;
    btn.textContent = "🔗 Share URL";
  }
};

window.downloadPDF = function() {
  const btn = $("btn-pdf");
  if (typeof html2pdf === "undefined") {
      showToast("PDF engine still loading, try again.");
      return;
  }
  btn.disabled = true;
  btn.textContent = "⏳ Generating...";
  
  const element = $("panel-results");
  const opt = {
    margin:       [10, 5, 10, 5],
    filename:     'DoshaNet-Profile.pdf',
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, backgroundColor: "#0d1220" },
    jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };
  
  // Hide buttons for PDF
  const actions = document.querySelector(".results-actions");
  if(actions) actions.style.display = "none";

  html2pdf().set(opt).from(element).save().then(() => {
    btn.disabled = false;
    btn.textContent = "📄 Download PDF";
    if(actions) actions.style.display = "flex";
    showToast("PDF Downloaded!", "ok");
  }).catch((e) => {
    btn.disabled = false;
    btn.textContent = "📄 Download PDF";
    if(actions) actions.style.display = "flex";
  });
};
