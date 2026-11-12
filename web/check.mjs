// Runs inspect.js under Node with onnxruntime-web (the browser's WASM runtime) for tests/test_web.py.
// Usage: node check.mjs <job.json>   job = {model, size, tensor: number[], origW, origH, band, box, drise_n}
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { decide, decode, drise, letterboxGeometry } from "./inspect.js";

ort.env.wasm.numThreads = 1;
const job = JSON.parse(readFileSync(process.argv[2], "utf8"));
const session = await ort.InferenceSession.create(readFileSync(job.model));
const size = job.size;
const geo = letterboxGeometry(job.origW, job.origH, size);
const run = async (t) => {
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", t, [1, 3, size, size]) });
  return decode(out[session.outputNames[0]].data, geo, job.origW, job.origH, 0.001);
};
const tensor = Float32Array.from(job.tensor);
const dets = await run(tensor);
const result = {
  geometry: geo,
  geometries: job.sizes.map(([w, h]) => letterboxGeometry(w, h, size)),
  dets: dets.slice(0, 10),
  decision: decide(dets.length ? dets[0].score : 0, job.band),
};
if (job.drise_n) {
  const sal = await drise(tensor, dets[0].box, run, { size, n: job.drise_n, grid: 6 });
  let hi = 0;
  let lo = 1;
  for (const v of sal) {
    hi = Math.max(hi, v);
    lo = Math.min(lo, v);
  }
  result.drise = { length: sal.length, max: hi, min: lo };
}
console.log(JSON.stringify(result));
