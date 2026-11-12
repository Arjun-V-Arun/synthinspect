// Shared inference logic for the browser demo. Pure functions over typed arrays, so the same
// file runs in Node (tests/test_web.py checks it against the Python pipeline) and in the browser.
// It mirrors synthinspect/detect/base.py (letterbox, decode) and synthinspect/eval/review.py (band).

export const PAD = 114;

// Python's round() rounds halves to even; Math.round rounds them up. The letterbox
// geometry must match the Python side exactly, so use Python's rule.
export function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function letterboxGeometry(w, h, size) {
  const r = Math.min(size / h, size / w);
  const nw = pyRound(w * r);
  const nh = pyRound(h * r);
  const dw = (size - nw) / 2;
  const dh = (size - nh) / 2;
  return { ratio: r, nw, nh, padX: pyRound(dw - 0.1), padY: pyRound(dh - 0.1), size };
}

// rgba: Uint8ClampedArray of an image ALREADY resized to (nw, nh) by the caller (canvas drawImage).
// Returns the (1, 3, size, size) float tensor the model expects.
export function toTensor(rgba, geo) {
  const { nw, nh, padX, padY, size } = geo;
  const plane = size * size;
  const t = new Float32Array(3 * plane).fill(PAD / 255);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const s = (y * nw + x) * 4;
      const d = (y + padY) * size + (x + padX);
      t[d] = rgba[s] / 255;
      t[plane + d] = rgba[s + 1] / 255;
      t[2 * plane + d] = rgba[s + 2] / 255;
    }
  }
  return t;
}

// YOLO26 NMS-free output: (1, maxDet, 6) = x1 y1 x2 y2 score class, letterboxed pixels.
export function decode(raw, geo, origW, origH, minScore = 0.05) {
  const dets = [];
  for (let i = 0; i + 5 < raw.length; i += 6) {
    const score = raw[i + 4];
    if (score < minScore) continue;
    const unbox = (v, pad, max) => Math.min(Math.max((v - pad) / geo.ratio, 0), max);
    dets.push({
      box: [unbox(raw[i], geo.padX, origW), unbox(raw[i + 1], geo.padY, origH), unbox(raw[i + 2], geo.padX, origW), unbox(raw[i + 3], geo.padY, origH)],
      score,
      cls: raw[i + 5],
    });
  }
  dets.sort((a, b) => b.score - a.score);
  return dets;
}

export function decide(score, band) {
  if (score < band.t_pass) return "pass";
  if (score > band.t_reject) return "reject";
  return "review";
}

export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

// D-RISE weight of one masked run: best IoU x score against the target box.
export function targetWeight(dets, box) {
  let best = 0;
  for (const d of dets) best = Math.max(best, iou(box, d.box) * d.score);
  return best;
}

// Small seeded PRNG so the demo's explanations are repeatable.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One RISE mask over the letterboxed image: a coarse random grid, bilinearly upsampled and shifted.
export function riseMask(size, grid, p, rand) {
  const cell = Math.ceil(size / grid);
  const g = new Float32Array((grid + 1) * (grid + 1));
  for (let i = 0; i < g.length; i++) g[i] = rand() < p ? 1 : 0;
  const dx = Math.floor(rand() * cell);
  const dy = Math.floor(rand() * cell);
  const m = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const gy = (y + dy) / cell;
    const y0 = Math.min(Math.floor(gy), grid - 1);
    const fy = gy - y0;
    for (let x = 0; x < size; x++) {
      const gx = (x + dx) / cell;
      const x0 = Math.min(Math.floor(gx), grid - 1);
      const fx = gx - x0;
      const a = g[y0 * (grid + 1) + x0];
      const b = g[y0 * (grid + 1) + x0 + 1];
      const c = g[(y0 + 1) * (grid + 1) + x0];
      const d = g[(y0 + 1) * (grid + 1) + x0 + 1];
      m[y * size + x] = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
    }
  }
  return m;
}

// Masked copy of the input tensor: hidden pixels become the image's mean colour.
export function applyMask(tensor, mask, mean, size) {
  const plane = size * size;
  const out = new Float32Array(tensor.length);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < plane; i++) out[c * plane + i] = tensor[c * plane + i] * mask[i] + mean[c] * (1 - mask[i]);
  }
  return out;
}

export function channelMeans(tensor, size) {
  const plane = size * size;
  return [0, 1, 2].map((c) => {
    let s = 0;
    for (let i = 0; i < plane; i++) s += tensor[c * plane + i];
    return s / plane;
  });
}

// D-RISE in the browser. `run(tensor)` returns decoded detections for a masked tensor.
export async function drise(tensor, box, run, { size, n = 120, grid = 10, p = 0.5, seed = 0, onProgress } = {}) {
  const rand = mulberry32(seed);
  const mean = channelMeans(tensor, size);
  const sal = new Float32Array(size * size);
  for (let k = 0; k < n; k++) {
    const m = riseMask(size, grid, p, rand);
    const w = targetWeight(await run(applyMask(tensor, m, mean, size)), box);
    if (w > 0) for (let i = 0; i < sal.length; i++) sal[i] += w * m[i];
    if (onProgress) onProgress((k + 1) / n);
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of sal) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = hi - lo || 1;
  for (let i = 0; i < sal.length; i++) sal[i] = (sal[i] - lo) / span;
  return sal;
}
