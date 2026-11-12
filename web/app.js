import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.mjs";
import { decide, decode, drise, letterboxGeometry, toTensor } from "./inspect.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
// GitHub Pages cannot send the headers multi-threaded WASM needs, so use one thread unless isolated.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) >> 1)) : 1;

const $ = (id) => document.getElementById(id);
const view = $("view");
const ctx = view.getContext("2d");
const state = { cfg: null, session: null, img: null, geo: null, tensor: null, dets: [], sample: null, heat: null, saved: null };

function status(msg) {
  $("status").textContent = msg;
}

const pct = (v) => (v == null ? "n/a" : `${(100 * v).toFixed(1)}%`);

async function boot() {
  try {
    state.cfg = await (await fetch("assets/config.json")).json();
  } catch {
    status("No model packaged yet. Run `python -m synthinspect demo --category <name>` and reload.");
    return;
  }
  const c = state.cfg;
  const t0 = performance.now();
  status("Loading model…");
  state.session = await ort.InferenceSession.create("assets/model.onnx", { executionProviders: ["wasm"] });
  const loadMs = performance.now() - t0;
  $("tags").innerHTML = [
    `category: ${c.category.replace("_", " ")}`,
    `training data: ${c.condition}`,
    `model ${c.version}`,
    `${c.imgsz}px input`,
    `runs on this device (WASM)`,
  ].map((t) => `<span class="tag">${t}</span>`).join("");
  const m = c.metrics;
  const r = m.review_cross_fitted || {};
  $("metrics").innerHTML = [
    ["Image AP", m.image_ap?.toFixed(3)],
    ["Image AUROC", m.image_auroc?.toFixed(3)],
    ["Box AP@0.1", m.box_ap10?.toFixed(3)],
    ["Automated", pct(r.automation)],
    ["Escaped defects", pct(r.escape_rate)],
    ["False rejects", pct(r.false_reject_rate)],
    ["Band", `${c.band.t_pass.toFixed(3)} – ${c.band.t_reject.toFixed(3)}`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const tp = Math.max(0, Math.min(1, c.band.t_pass));
  const tr = Math.max(tp, Math.min(1, c.band.t_reject));
  $("zpass").style.width = `${100 * tp}%`;
  $("zreview").style.width = `${100 * (tr - tp)}%`;
  $("zreject").style.width = `${100 * (1 - tr)}%`;

  const gallery = $("gallery");
  c.samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "thumb";
    b.setAttribute("aria-pressed", "false");
    b.setAttribute("aria-label", `Sample ${i + 1}: ${s.label === "good" ? "good part" : s.defect_type.replace("_", " ")}`);
    b.innerHTML = `<img src="assets/samples/${s.file}" alt=""><span>${s.label === "good" ? "good" : s.defect_type.replace("_", " ")}</span>`;
    b.onclick = () => selectSample(i, b);
    gallery.appendChild(b);
  });
  status(`Model ready in ${loadMs.toFixed(0)} ms. Pick a part or upload an image.`);
  if (c.samples.length) selectSample(c.samples.length - 1, gallery.lastChild);
}

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}

async function selectSample(i, btn) {
  document.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
  btn.setAttribute("aria-pressed", "true");
  const s = state.cfg.samples[i];
  state.sample = s;
  state.saved = await loadImage(`assets/samples/${s.explanation}`);
  await inspect(await loadImage(`assets/samples/${s.file}`));
}

async function run(tensor) {
  const size = state.cfg.imgsz;
  const out = await state.session.run({ [state.session.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, size, size]) });
  return decode(out[state.session.outputNames[0]].data, state.geo, state.img.naturalWidth, state.img.naturalHeight, 0.001);
}

async function inspect(img) {
  state.img = img;
  state.heat = null;
  const size = state.cfg.imgsz;
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  state.geo = letterboxGeometry(w, h, size);
  const off = new OffscreenCanvas(state.geo.nw, state.geo.nh);
  const octx = off.getContext("2d");
  octx.imageSmoothingQuality = "high";
  octx.drawImage(img, 0, 0, state.geo.nw, state.geo.nh);
  state.tensor = toTensor(octx.getImageData(0, 0, state.geo.nw, state.geo.nh).data, state.geo);
  const t0 = performance.now();
  state.dets = await run(state.tensor);
  const ms = performance.now() - t0;
  const score = state.dets.length ? state.dets[0].score : 0;
  const decision = decide(score, state.cfg.band);
  draw();
  const stamp = $("stamp");
  stamp.hidden = false;
  stamp.className = `stamp ${decision}`;
  stamp.textContent = decision.toUpperCase();
  $("marker").style.left = `calc(${(100 * Math.min(1, score)).toFixed(2)}% - 1px)`;
  const truth = state.sample ? (state.sample.label === "good" ? "good part" : state.sample.defect_type.replace("_", " ")) : "unknown (uploaded)";
  $("readout").innerHTML = `<span>score <b>${score.toFixed(3)}</b></span><span>inference <b>${ms.toFixed(0)} ms</b></span><span>truth <b>${truth}</b></span>`;
  $("explainLive").disabled = state.dets.length === 0;
  $("showSaved").disabled = !state.sample;
  $("clearExp").disabled = true;
  status(decision === "review" ? "Unsure: this part would go to a person." : `Auto-${decision}.`);
}

function draw(overlay = null) {
  const img = state.img;
  const scale = Math.min(640 / img.naturalWidth, 640 / img.naturalHeight, 1);
  view.width = Math.round(img.naturalWidth * scale);
  view.height = Math.round(img.naturalHeight * scale);
  if (overlay === "saved" && state.saved) {
    ctx.drawImage(state.saved, 0, 0, view.width, view.height);
  } else {
    ctx.drawImage(img, 0, 0, view.width, view.height);
    if (state.heat) {
      ctx.globalAlpha = 0.5;
      ctx.drawImage(state.heat, 0, 0, view.width, view.height);
      ctx.globalAlpha = 1;
    }
  }
  const top = state.dets[0]?.score ?? 0;
  ctx.lineWidth = 2;
  ctx.font = "600 12px 'IBM Plex Mono', monospace";
  state.dets.filter((d, i) => i < 5 && d.score >= Math.max(0.05, 0.5 * top)).forEach((d, i) => {
    const [x1, y1, x2, y2] = d.box.map((v) => v * scale);
    ctx.strokeStyle = i === 0 ? "#e6a443" : "rgba(230,164,67,.55)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = d.score.toFixed(2);
    ctx.fillStyle = "rgba(0,0,0,.7)";
    ctx.fillRect(x1, Math.max(0, y1 - 16), ctx.measureText(label).width + 8, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 4, Math.max(12, y1 - 4));
  });
}

function heatmapCanvas(sal) {
  const { size, padX, padY, nw, nh } = state.geo;
  const c = new OffscreenCanvas(nw, nh);
  const g = c.getContext("2d");
  const data = g.createImageData(nw, nh);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = sal[(y + padY) * size + (x + padX)];
      const o = (y * nw + x) * 4;
      data.data[o] = 255 * Math.min(1, 1.6 * v);
      data.data[o + 1] = 255 * Math.max(0, Math.min(1, 2 * v - 0.6));
      data.data[o + 2] = 255 * Math.max(0, 0.5 - v);
      data.data[o + 3] = 255 * Math.min(1, 0.25 + v);
    }
  }
  g.putImageData(data, 0, 0);
  return c;
}

$("explainLive").onclick = async () => {
  const btn = $("explainLive");
  btn.disabled = true;
  const prog = $("prog");
  prog.hidden = false;
  const n = 120;
  const t0 = performance.now();
  const sal = await drise(state.tensor, state.dets[0].box, run, {
    size: state.cfg.imgsz,
    n,
    grid: 10,
    onProgress: (f) => {
      prog.value = f;
      status(`Explaining: ${Math.round(100 * f)}% (${n} masked runs)`);
    },
  });
  state.heat = heatmapCanvas(sal);
  prog.hidden = true;
  draw();
  status(`D-RISE explanation of the top detection in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
  btn.disabled = false;
  $("clearExp").disabled = false;
};

$("showSaved").onclick = () => {
  draw("saved");
  $("clearExp").disabled = false;
  status("ODAM map from the PyTorch model (computed when the demo was packaged).");
};

$("clearExp").onclick = () => {
  state.heat = null;
  draw();
  $("clearExp").disabled = true;
};

$("upload").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  document.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
  state.sample = null;
  state.saved = null;
  const url = URL.createObjectURL(f);
  await inspect(await loadImage(url));
  URL.revokeObjectURL(url);
};

boot().catch((err) => {
  console.error(err);
  status(`Could not start: ${err.message}`);
});
