// Compare the JavaScript model with reference values from the Python code.
// Usage: node tests/validate.mjs
import { readFileSync } from 'node:fs';
import { LambdaTables } from '../web/js/lambda.js';
import { DEFAULTS, pointModel } from '../web/js/model.js';

const dir = new URL('../web/data/', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('lambda_tables.json', dir)));
const buf = (f) => { const b = readFileSync(new URL(f, dir)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const T = new LambdaTables(meta, buf(meta.lambda3d.file), buf(meta.lambda2d.file));
const ref = JSON.parse(readFileSync(new URL('reference.json', import.meta.url)));

const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor((q / 100) * s.length))]; };
let failed = false;
for (const c of ref) {
  const p = { ...DEFAULTS, vfrag: c.params.vfrag, epsMax: c.params.eps_max, sigEps: c.params.sigma_log,
              epsBkg: c.params.eps_floor, mstar: c.params.mstar };
  const err = { St_max: [], eps: [], cdcs: [], mJ: [] };
  let ddgiMismatch = 0;
  for (const r of c.points) {
    const q = pointModel(T, r.mdot, r.r, p);
    if (q.stMax === undefined) continue; // MRI cells (the reference only masks frag / no GI)
    const rel = (a, b) => Math.abs(a / b - 1);
    err.St_max.push(rel(q.stMax, r.St_max));
    err.eps.push(rel(q.eps, r.eps));
    err.cdcs.push(rel(q.cdcs, r.cdcs));
    err.mJ.push(rel(q.mJ, r.mJ));
    if ((q.cdcs < q.eps) !== (r.cdcs < r.eps)) ddgiMismatch++;
  }
  console.log(`v_frag=${p.vfrag} eps_max=${p.epsMax} sigma=${p.sigEps} eps_bkg=${p.epsBkg} M*=${p.mstar}  (${err.cdcs.length} pts)`);
  for (const [k, v] of Object.entries(err))
    console.log(`  ${k.padEnd(7)} rel.err median ${pct(v, 50).toExponential(1)}  90% ${pct(v, 90).toExponential(1)}  max ${pct(v, 100).toExponential(1)}`);
  console.log(`  DDGI classification mismatches: ${ddgiMismatch}`);
  if (pct(err.cdcs, 90) > 0.01 || ddgiMismatch > 2) failed = true;
}
process.exit(failed ? 1 : 0);
