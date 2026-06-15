'use strict';

// ─── Seeded PRNG (mulberry32) ──────────────────────────────────────────────
function makePRNG(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng = makePRNG(1);

function randn() {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Math helpers ──────────────────────────────────────────────────────────
function torus(x, w) {
  return ((x + w * 0.5) % w + w) % w - w * 0.5;
}
function wrapToPi(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }
function hypot2(dx, dy) { return Math.sqrt(dx * dx + dy * dy); }
function posmod(x, w) { return ((x % w) + w) % w; }

// ─── Colour helpers ────────────────────────────────────────────────────────
function hsvToRgb(h, s, v) {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
}

// ─── Matrix (row-major, Float64Array backed) ───────────────────────────────
class Matrix {
  constructor(rows, cols, data) {
    this.rows = rows;
    this.cols = cols;
    this.data = (data instanceof Float64Array) ? data : new Float64Array(rows * cols);
  }

  static zeros(rows, cols) { return new Matrix(rows, cols); }

  static randn(rows, cols, scale) {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = randn() * (scale ?? 1);
    return m;
  }

  clone() { return new Matrix(this.rows, this.cols, new Float64Array(this.data)); }

  get(r, c) { return this.data[r * this.cols + c]; }
  set(r, c, v) { this.data[r * this.cols + c] = v; }

  addNoise(sigma) {
    for (let i = 0; i < this.data.length; i++) this.data[i] += randn() * sigma;
  }

  meanAll() {
    if (!this.data.length) return 0;
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return s / this.data.length;
  }

  stdAll() {
    if (this.data.length < 2) return 0;
    const mu = this.meanAll();
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += (this.data[i] - mu) ** 2;
    return Math.sqrt(s / this.data.length);
  }

  sumSign() {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += Math.sign(this.data[i]);
    return s;
  }

  frobenius(other) {
    const R = Math.max(this.rows, other.rows);
    const C = Math.max(this.cols, other.cols);
    let s = 0;
    for (let r = 0; r < R; r++)
      for (let c = 0; c < C; c++) {
        const a = (r < this.rows && c < this.cols) ? this.get(r, c) : 0;
        const b = (r < other.rows && c < other.cols) ? other.get(r, c) : 0;
        s += (a - b) ** 2;
      }
    return Math.sqrt(s);
  }

  addRow(scale) {
    scale = scale ?? 0.3;
    const nd = new Float64Array((this.rows + 1) * this.cols);
    nd.set(this.data);
    for (let c = 0; c < this.cols; c++) nd[this.rows * this.cols + c] = randn() * scale;
    this.rows++;
    this.data = nd;
  }

  addCol(scale) {
    scale = scale ?? 0.3;
    const nc = this.cols + 1;
    const nd = new Float64Array(this.rows * nc);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) nd[r * nc + c] = this.get(r, c);
      nd[r * nc + this.cols] = randn() * scale;
    }
    this.cols = nc;
    this.data = nd;
  }

  toJSON() { return { rows: this.rows, cols: this.cols, data: Array.from(this.data) }; }
  static fromJSON(o) { return new Matrix(o.rows, o.cols, new Float64Array(o.data)); }
}

// matrix × column vector, tanh activation
function matMulTanh(M, x) {
  const out = new Float64Array(M.rows);
  for (let r = 0; r < M.rows; r++) {
    let s = 0;
    for (let c = 0; c < M.cols; c++) s += M.data[r * M.cols + c] * x[c];
    out[r] = Math.tanh(s);
  }
  return out;
}

// Forward pass — output only (fast path for non-selected creatures)
function nnForward(W, inp) {
  let x = inp;
  for (let l = 0; l < W.length - 1; l++) x = matMulTanh(W[l], x);
  return matMulTanh(W[W.length - 1], x);
}

// Forward pass — returns output + all per-layer activations for brain visualisation
function nnForwardFull(W, inp) {
  const acts = [inp];
  let x = inp;
  for (let l = 0; l < W.length - 1; l++) {
    x = matMulTanh(W[l], x);
    acts.push(x);
  }
  const out = matMulTanh(W[W.length - 1], x);
  acts.push(out);
  return { out, acts };
}

// ─── Simulation parameters ─────────────────────────────────────────────────
const P = {
  N0: 30, Npred: 6, worldSize: 120,
  foodCount: 80, foodEnergy: 22,
  bodyRadius: 1.4, dt: 0.2, maxSpeed: 10,
  energyDecay: 0.1, moveCost: 0.05, predMoveCost: 0.07,
  reproEnergy: 100, carnReproEnergy: 80, mutationSigma: 0.3, nnHidden: 8,
  senseRadius: 25,
  addLayerProb: 0.18, delLayerProb: 0.02, addNeuronProb: 0.22,
  maxHiddenLayers: 5, maxNeurons: 70,
  fitFood: 5, fitTick: 0.01, fitRepro: 20,
  maxAgeHerb: 1500, maxAgeCarn: 3000, modeThresh: 0.10,
  nInput: 5, nOutput: 3,
};

// FIX 1 — Speed: 11 steps, ¼× (0.25 ticks/frame) is the MAXIMUM
// At 60 fps: step 5 ≈ 3 ticks/sec (default — visible individual movement)
const SPEED_STEPS  = [0.001, 0.003, 0.007, 0.015, 0.035, 0.05, 0.09, 0.14, 0.19, 0.22, 0.25];
const SPEED_LABELS = ['1/1000','1/333','1/143','1/67','1/29','1/20','1/11','1/7','1/5','1/4.5','¼×'];

// ─── Stable creature identity counter (FIX 6) ─────────────────────────────
let nextCreatureId = 0;

// ─── Creature helpers ──────────────────────────────────────────────────────
function geneColor(cre) {
  const h = ((cre.W[0].meanAll() * 0.4 + 0.5) % 1 + 1) % 1;
  const s = 0.6 + 0.3 / (1 + Math.exp(-cre.W[cre.W.length - 1].stdAll() * 2));
  return hsvToRgb(h, s, 1);
}

function makeInitBrain() {
  return [
    Matrix.randn(P.nnHidden, P.nInput, 0.5),
    Matrix.randn(P.nOutput, P.nnHidden, 0.5),
  ];
}

function makeCreature(posX, posY, W, type, lineage, energy) {
  const c = {
    id: nextCreatureId++,          // FIX 6: stable identity that survives array compaction
    posX, posY, velX: 0, velY: 0,
    angle: rng() * 2 * Math.PI,
    W,
    energy: (energy !== undefined) ? energy : 55 + rng() * 12,
    act:  new Float64Array(P.nOutput),
    acts: null,
    color: [1, 1, 1],
    age: 0, fitness: 30,
    type,
    busyTime: 0, fadeTick: 0, fadeInit: 0, immature: 0,
    eatCooldown: 0, deathStart: 0,
    lineage,
  };
  c.color = type === 'carn' ? [1, 0.1, 0.1] : geneColor(c);
  return c;
}

// Creature shape — encodes brain complexity + weight personality in geometry
function polyVerts(cre) {
  const W     = cre.W;
  const Wlast = W[W.length - 1];

  // Sides (3–8): total hidden neurons reflects brain complexity
  const totalHidden = W.slice(0, -1).reduce((s, m) => s + m.rows, 0);
  const ns = Math.max(3, Math.min(8, 3 + Math.floor(totalHidden / 6)));

  // Base size: average weight std across all layers
  const allStd = W.reduce((s, m) => s + m.stdAll(), 0) / W.length;
  const sG     = 0.7 + 0.8 / (1 + Math.exp(-allStd * 2));

  // Growth: 0.25× at birth → 1.0× after 300 ticks
  const grow = 0.25 + 0.75 * Math.min(1, cre.age / 300);

  // Smooth time-based fade (real-time ms, not tick-discrete)
  let fadeScale = 1;
  if (cre.deathStart > 0) {
    fadeScale = Math.max(0, 1 - (performance.now() - cre.deathStart) / 1400);
  } else if (cre.fadeInit > 0 && cre.fadeTick === 0) {
    return { vx: [], vy: [], R: 0 };
  }

  const Rbase = P.bodyRadius * sG * grow * fadeScale;
  if (Rbase <= 0) return { vx: [], vy: [], R: 0 };

  // Irregular per-vertex radii — each maps to a specific weight value
  // so every creature has a recognisably unique silhouette
  const wdata = Wlast.data;
  const wlen  = wdata.length || 1;
  const vx = new Array(ns), vy = new Array(ns);
  for (let i = 0; i < ns; i++) {
    const th   = (i / ns) * 2 * Math.PI + cre.angle;
    const wIdx = Math.floor(i * wlen / ns) % wlen;
    const spike = Math.tanh(wdata[wIdx]) * 0.40;   // ±40% radius variation
    const Ri   = Math.max(P.bodyRadius * 0.25, Rbase * (1 + spike));
    vx[i] = cre.posX + Ri * Math.cos(th);
    vy[i] = cre.posY + Ri * Math.sin(th);
  }
  return { vx, vy, R: Rbase };
}

// ─── World state ───────────────────────────────────────────────────────────
let creatures = [];
let food      = [];
let tick      = 0;
let baseW     = null;
let rngSeed   = 1;

const POP_MAX  = 1000;
const popHerb  = new Int16Array(POP_MAX);
const popCarn  = new Int16Array(POP_MAX);
let   popHead  = 0;
let   popCount = 0;

function initWorld(seed) {
  rngSeed = seed ?? 1;
  rng     = makePRNG(rngSeed);
  tick    = 0;
  popHead = 0; popCount = 0;
  nextCreatureId = 0;          // reset ID counter on new world
  creatures = [];
  for (let k = 0; k < P.N0; k++)
    creatures.push(makeCreature(rng() * P.worldSize, rng() * P.worldSize, makeInitBrain(), 'herb', k));
  for (let k = 0; k < P.Npred; k++) {
    const i = P.N0 + k;
    creatures.push(makeCreature(rng() * P.worldSize, rng() * P.worldSize, makeInitBrain(), 'carn', i, 60 + rng() * 20));
  }
  baseW = creatures[0].W.map(m => m.clone());
  food  = [];
  spawnFood();
}

function spawnFood() {
  while (food.length < P.foodCount)
    food.push({ x: rng() * P.worldSize, y: rng() * P.worldSize });
}

// ─── Structural mutation ───────────────────────────────────────────────────
function mutateNetworkStructure(W) {
  const outSize = P.nOutput;
  let L = W.length - 1;

  if (L < P.maxHiddenLayers && rng() < P.addLayerProb) {
    const nPrev = W[W.length - 2].rows;
    const nNew  = Math.max(2, Math.min(P.maxNeurons, Math.round(nPrev * (0.7 + 0.6 * rng()))));
    W = [...W.slice(0, W.length - 1), Matrix.randn(nNew, nPrev, 0.3), Matrix.randn(outSize, nNew, 0.3)];
    L++;
  }

  if (L > 1 && rng() < P.delLayerProb) {
    const kill     = Math.floor(rng() * L);
    const prevCols = kill === 0 ? P.nInput : W[kill - 1].rows;
    if (kill === L - 1)
      W[W.length - 1] = Matrix.randn(outSize, prevCols, 0.3);
    else
      W[kill + 1] = Matrix.randn(W[kill + 1].rows, prevCols, 0.3);
    W = [...W.slice(0, kill), ...W.slice(kill + 1)];
    L--;
  }

  if (L >= 1 && rng() < P.addNeuronProb) {
    const which = Math.floor(rng() * L);
    if (W[which].rows < P.maxNeurons) {
      W[which].addRow(0.3);
      W[which + 1].addCol(0.3);
    }
  }

  return W;
}

// ─── Simulation step ───────────────────────────────────────────────────────
function simStep() {
  const n = creatures.length;
  if (n === 0) { spawnFood(); return; }

  const posX     = new Float64Array(n);
  const posY     = new Float64Array(n);
  const velX     = new Float64Array(n);
  const velY     = new Float64Array(n);
  const ang      = new Float64Array(n);
  const E        = new Float64Array(n);
  const fit      = new Float64Array(n);
  const age      = new Float64Array(n);
  const busy     = new Float64Array(n);
  const fade     = new Float64Array(n);
  const fadeInit = new Float64Array(n);
  const imm      = new Float64Array(n);
  const isCarn   = new Uint8Array(n);
  const lineage  = new Int32Array(n);
  const eatCD    = new Float64Array(n);   // cooldown ticks before carnivore can eat again

  for (let k = 0; k < n; k++) {
    const c = creatures[k];
    posX[k] = c.posX; posY[k] = c.posY;
    velX[k] = c.velX; velY[k] = c.velY;
    ang[k]  = c.angle;
    E[k]    = c.energy;   fit[k] = c.fitness;
    age[k]  = c.age;      busy[k] = c.busyTime;
    fade[k] = c.fadeTick; fadeInit[k] = c.fadeInit;
    imm[k]  = c.immature;
    eatCD[k] = c.eatCooldown | 0;
    isCarn[k] = c.type === 'carn' ? 1 : 0;
    lineage[k] = c.lineage | 0;
  }

  for (let k = 0; k < n; k++) {
    age[k]++;
    fit[k] += P.fitTick;
    if (busy[k]  > 0) busy[k]--;
    if (fade[k]  > 0) fade[k]--;
    if (imm[k]   > 0) imm[k]--;
    if (eatCD[k] > 0) eatCD[k]--;
  }

  const WS = P.worldSize, SR = P.senseRadius;

  // ── Sense ─────────────────────────────────────────────────────────────
  const dx = new Float64Array(n);
  const dy = new Float64Array(n);
  const d  = new Float64Array(n).fill(SR);

  // herbivores: toward food, flee if predator is closer
  for (let h = 0; h < n; h++) {
    if (isCarn[h]) continue;
    let foodD2 = Infinity, fdx = 0, fdy = 0;
    for (let f = 0; f < food.length; f++) {
      const fx = torus(posX[h] - food[f].x, WS);
      const fy = torus(posY[h] - food[f].y, WS);
      const d2 = fx * fx + fy * fy;
      if (d2 < foodD2) { foodD2 = d2; fdx = fx; fdy = fy; }
    }
    let predD2 = Infinity, pdx = 0, pdy = 0;
    for (let p = 0; p < n; p++) {
      if (!isCarn[p]) continue;
      const px = torus(posX[h] - posX[p], WS);
      const py = torus(posY[h] - posY[p], WS);
      const d2 = px * px + py * py;
      if (d2 < predD2) { predD2 = d2; pdx = px; pdy = py; }
    }
    if (predD2 < foodD2 && predD2 < SR * SR) {
      dx[h] = pdx; dy[h] = pdy; d[h] = Math.sqrt(predD2);
    } else if (foodD2 < SR * SR) {
      dx[h] = -fdx; dy[h] = -fdy; d[h] = Math.sqrt(foodD2);
    }
  }

  // carnivores: sense nearest live herbivore within senseRadius
  for (let cc = 0; cc < n; cc++) {
    if (!isCarn[cc]) continue;
    let bestD = SR, bx = 0, by = 0;
    for (let p = 0; p < n; p++) {
      if (isCarn[p] || fade[p] > 0 || imm[p] > 0) continue;
      const ddx = torus(posX[p] - posX[cc], WS);
      const ddy = torus(posY[p] - posY[cc], WS);
      const dist = hypot2(ddx, ddy);
      if (dist < bestD) { bestD = dist; bx = ddx; by = ddy; }
    }
    if (bestD < SR) { dx[cc] = bx; dy[cc] = by; d[cc] = bestD; }
  }

  // ── NN forward pass ────────────────────────────────────────────────────
  const act  = new Array(n);
  const acts = new Array(n);
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) {
      act[k]  = new Float64Array(P.nOutput);
      acts[k] = null;
      continue;
    }
    const inp = new Float64Array([dx[k], dy[k], d[k] / SR, E[k] / P.reproEnergy, 1]);
    const res = nnForwardFull(creatures[k].W, inp);
    act[k]  = res.out;
    acts[k] = res.acts;
  }

  // ── Interpret outputs → motion ─────────────────────────────────────────
  const MT = P.modeThresh;
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) { velX[k] = 0; velY[k] = 0; continue; }
    const modeRaw  = act[k][0];
    const thrustIn = Math.max(act[k][1], 0);
    const mode     = modeRaw > MT ? 1 : modeRaw < -MT ? -1 : 0;
    let theta = Math.atan2(dy[k], dx[k]);
    if (mode === -1) theta += Math.PI;
    ang[k] += 3 * P.dt * wrapToPi(theta - ang[k]);
    if (mode !== 0) {
      velX[k] += thrustIn * Math.cos(ang[k]);
      velY[k] += thrustIn * Math.sin(ang[k]);
    }
    const spd = hypot2(velX[k], velY[k]);
    if (spd > P.maxSpeed) {
      const sc = P.maxSpeed / spd;
      velX[k] *= sc; velY[k] *= sc;
    }
  }

  // ── Physics ────────────────────────────────────────────────────────────
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) continue;
    posX[k] = posmod(posX[k] + velX[k] * P.dt, WS);
    posY[k] = posmod(posY[k] + velY[k] * P.dt, WS);
    const spd = hypot2(velX[k], velY[k]);
    E[k] -= P.energyDecay + P.moveCost * spd;
    if (isCarn[k]) E[k] -= P.predMoveCost * spd;
    if (E[k] > 200) E[k] = 200;
  }

  // ── Herbivores eat food ────────────────────────────────────────────────
  const r2    = P.bodyRadius * P.bodyRadius;
  const eaten = new Uint8Array(food.length);
  for (let h = 0; h < n; h++) {
    if (isCarn[h] || fade[h] > 0) continue;
    for (let f = 0; f < food.length; f++) {
      if (eaten[f]) continue;
      const fx = torus(posX[h] - food[f].x, WS);
      const fy = torus(posY[h] - food[f].y, WS);
      if (fx * fx + fy * fy <= r2) {
        eaten[f] = 1;
        E[h] += P.foodEnergy;
        fit[h] += P.fitFood;
      }
    }
  }
  food = food.filter((_, f) => !eaten[f]);

  // ── Predator attacks ────────────────────────────────────────────────────
  // EAT_TICKS: carnivore briefly frozen during kill strike
  // EAT_COOLDOWN: ticks carnivore must wait before next attack (can still move)
  // FADE_TICKS: sim keeps dying herb alive long enough for smooth visual fade
  const EAT_TICKS    = 4;
  const EAT_COOLDOWN = 36;               // total eat cycle ≈ 40 ticks
  const ATTACK_R     = P.bodyRadius * 5;
  const FADE_TICKS   = 30;              // creature stays in sim during visual shrink
  for (let cc = 0; cc < n; cc++) {
    if (!isCarn[cc] || busy[cc] > 0 || fade[cc] > 0 || eatCD[cc] > 0) continue;
    let bestD = ATTACK_R, bestPrey = -1;
    for (let p = 0; p < n; p++) {
      if (isCarn[p] || fade[p] > 0 || imm[p] > 0) continue;
      const dist = hypot2(torus(posX[p] - posX[cc], WS), torus(posY[p] - posY[cc], WS));
      if (dist < bestD) { bestD = dist; bestPrey = p; }
    }
    if (bestPrey >= 0) {
      busy[cc]  = EAT_TICKS;
      eatCD[cc] = EAT_COOLDOWN;
      fade[bestPrey]     = FADE_TICKS;
      fadeInit[bestPrey] = FADE_TICKS;
      creatures[bestPrey].deathStart = performance.now();  // smooth real-time fade
      E[cc] += E[bestPrey] / 2;
      fit[cc] += P.fitFood;
    }
  }

  // ── Deaths ─────────────────────────────────────────────────────────────
  const keep = new Uint8Array(n).fill(1);
  for (let k = 0; k < n; k++) {
    const tooOld   = isCarn[k] ? age[k] >= P.maxAgeCarn : age[k] >= P.maxAgeHerb;
    const fadeDead = fade[k] === 0 && fadeInit[k] > 0;
    if (tooOld || fadeDead || E[k] <= 0) keep[k] = 0;
  }

  const alive = [];
  for (let k = 0; k < n; k++) {
    if (!keep[k]) continue;
    const c = creatures[k];
    c.posX = posX[k]; c.posY = posY[k];
    c.velX = velX[k]; c.velY = velY[k];
    c.angle    = ang[k];
    c.energy   = E[k];   c.fitness = fit[k];
    c.age      = age[k]; c.act     = act[k];
    c.acts     = acts[k] || c.acts;
    c.busyTime    = busy[k];
    c.fadeTick    = fade[k]; c.fadeInit = fadeInit[k];
    c.immature    = imm[k];
    c.eatCooldown = eatCD[k];
    alive.push(c);
  }
  creatures = alive;

  // ── Reproduction ───────────────────────────────────────────────────────
  const toRepro = [];
  for (let k = 0; k < creatures.length; k++) {
    const c = creatures[k];
    if (c.fadeTick > 0 || c.busyTime > 0) continue;
    const thresh = c.type === 'carn' ? P.carnReproEnergy : P.reproEnergy;
    const wantsRepro = c.act[2] > 0 || c.energy >= thresh * 1.5;
    if (c.energy >= thresh && wantsRepro) toRepro.push(k);
  }
  for (const k of toRepro) reproduce(k);

  spawnFood();
}

function reproduce(i) {
  const par = creatures[i];
  const eHalf = par.energy / 2;
  par.energy = eHalf;
  par.fitness += P.fitRepro;
  par.immature = Math.max(par.immature, 5);

  const childW  = par.W.map(m => { const mc = m.clone(); mc.addNoise(P.mutationSigma); return mc; });
  const mutated = mutateNetworkStructure(childW);

  const chi = makeCreature(
    posmod(par.posX + Math.cos(par.angle) * P.bodyRadius * 3, P.worldSize),
    posmod(par.posY + Math.sin(par.angle) * P.bodyRadius * 3, P.worldSize),
    mutated, par.type, par.lineage, eHalf,
  );
  chi.fitness  = par.fitness / 2;
  chi.immature = Math.round(2 / P.dt);
  chi.angle    = rng() * 2 * Math.PI;
  if (par.type === 'herb') chi.color = geneColor(chi);
  creatures.push(chi);
}

// ─── Diversity score ───────────────────────────────────────────────────────
function diversityScore(cre) {
  const Lc = cre.W.length - 1, Lb = baseW.length - 1;
  const dL = Math.abs(Lc - Lb);
  let dN = 0;
  const mL = Math.max(Lc, Lb);
  for (let k = 0; k < mL; k++)
    dN += Math.abs((k < Lc ? cre.W[k].rows : 0) - (k < Lb ? baseW[k].rows : 0));
  let dF = 0;
  for (let k = 0; k < Math.min(Lc + 1, Lb + 1); k++) dF += cre.W[k].frobenius(baseW[k]);
  return dL + 0.2 * dN + 0.002 * dF;
}

// ─── Renderer ─────────────────────────────────────────────────────────────
class Renderer {
  constructor() {
    this.wCanvas   = document.getElementById('worldCanvas');
    this.wCtx      = this.wCanvas.getContext('2d');
    this.pCanvas   = document.getElementById('popCanvas');
    this.pCtx      = this.pCanvas.getContext('2d');
    this.bCanvases = [0,1,2].map(i => document.getElementById(`brain${i}`));
    this.bCtxs     = this.bCanvases.map(c => c.getContext('2d'));
    this.dCanvas   = document.getElementById('brainDetail');
    this.dCtx      = this.dCanvas.getContext('2d');

    this.dpr     = 1;
    this.scale   = 1;
    this.offsetX = 0;
    this.offsetY = 0;

    this._lastSel = -1;
    this._pulseT  = 0;
  }

  resize() {
    this.dpr = Math.min(window.devicePixelRatio || 1, 3);
    this._size(this.wCanvas, this.wCtx);
    this._size(this.pCanvas, this.pCtx);
    this.bCanvases.forEach((c, i) => this._size(c, this.bCtxs[i]));
    this._size(this.dCanvas, this.dCtx);

    const ww = this.wCanvas.clientWidth  || 1;
    const wh = this.wCanvas.clientHeight || 1;
    this.scale   = Math.min(ww / P.worldSize, wh / P.worldSize);
    this.offsetX = (ww - P.worldSize * this.scale) / 2;
    this.offsetY = (wh - P.worldSize * this.scale) / 2;
  }

  _size(canvas, ctx) {
    const d = this.dpr;
    const w = Math.max(1, canvas.clientWidth);
    const h = Math.max(1, canvas.clientHeight);
    const pw = Math.round(w * d), ph = Math.round(h * d);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw; canvas.height = ph;
    }
    ctx.setTransform(d, 0, 0, d, 0, 0);
  }

  _wx(x) { return this.offsetX + x * this.scale; }
  _wy(y) { return this.offsetY + (P.worldSize - y) * this.scale; }

  canvasToWorld(cx, cy) {
    return {
      x: (cx - this.offsetX) / this.scale,
      y: P.worldSize - (cy - this.offsetY) / this.scale,
    };
  }

  // ── World ──────────────────────────────────────────────────────────────
  // FIX 4: no zoom transform — world canvas always shows full boundary
  drawWorld(selectedIdx) {
    const ctx = this.wCtx;
    const cw  = this.wCanvas.clientWidth;
    const ch  = this.wCanvas.clientHeight;

    ctx.fillStyle = '#050509';
    ctx.fillRect(0, 0, cw, ch);

    // Food (batched into one path)
    ctx.fillStyle = '#2ecc40';
    ctx.beginPath();
    for (const f of food) {
      const fx = this._wx(f.x), fy = this._wy(f.y);
      ctx.moveTo(fx + 2.5, fy);
      ctx.arc(fx, fy, 2.5, 0, 6.2832);
    }
    ctx.fill();

    // Creatures
    for (let k = 0; k < creatures.length; k++) {
      const c = creatures[k];
      const { vx, vy, R } = polyVerts(c);
      if (R <= 0 || vx.length === 0) continue;
      const [r, g, b] = c.color;
      ctx.fillStyle = `rgb(${(r*255)|0},${(g*255)|0},${(b*255)|0})`;
      ctx.beginPath();
      ctx.moveTo(this._wx(vx[0]), this._wy(vy[0]));
      for (let i = 1; i < vx.length; i++) ctx.lineTo(this._wx(vx[i]), this._wy(vy[i]));
      ctx.closePath();
      ctx.fill();
    }

    // Selected creature — pulsing halo, no zoom
    if (selectedIdx >= 0 && selectedIdx < creatures.length) {
      const c = creatures[selectedIdx];
      this._pulseT = (this._pulseT + 0.05) % (2 * Math.PI);
      const pulse = 1 + 0.18 * Math.sin(this._pulseT);
      const haloR = P.bodyRadius * 4 * this.scale * pulse;

      ctx.shadowColor = '#ffdd00';
      ctx.shadowBlur  = 14;
      ctx.strokeStyle = '#ffdd00';
      ctx.lineWidth   = 2.5;
      ctx.beginPath();
      ctx.arc(this._wx(c.posX), this._wy(c.posY), haloR, 0, 6.2832);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Direction dot
      const ax = this._wx(c.posX) + Math.cos(-c.angle) * haloR * 1.4;
      const ay = this._wy(c.posY) + Math.sin(-c.angle) * haloR * 1.4;
      ctx.fillStyle = '#ffdd00';
      ctx.beginPath();
      ctx.arc(ax, ay, 3, 0, 6.2832);
      ctx.fill();
    }
  }

  // ── Population graph ───────────────────────────────────────────────────
  drawPopGraph() {
    const ctx = this.pCtx;
    const cw  = this.pCanvas.clientWidth;
    const ch  = this.pCanvas.clientHeight;
    ctx.fillStyle = '#06060d';
    ctx.fillRect(0, 0, cw, ch);
    if (popCount < 2) return;

    let maxVal = 4;
    for (let i = 0; i < popCount; i++) {
      const idx = (popHead - popCount + i + POP_MAX) % POP_MAX;
      const tot = popHerb[idx] + popCarn[idx];
      if (tot > maxVal) maxVal = tot;
    }

    const xStep  = cw / (popCount - 1);
    const yScale = (ch - 6) / (maxVal * 1.15);

    const line = (arr, color, lw) => {
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      ctx.beginPath();
      for (let i = 0; i < popCount; i++) {
        const idx = (popHead - popCount + i + POP_MAX) % POP_MAX;
        i === 0 ? ctx.moveTo(i * xStep, ch - 3 - arr[idx] * yScale)
                : ctx.lineTo(i * xStep, ch - 3 - arr[idx] * yScale);
      }
      ctx.stroke();
    };

    line(popHerb, '#3b3', 1.5);
    line(popCarn, '#c33', 1.5);

    ctx.strokeStyle = '#556'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < popCount; i++) {
      const idx = (popHead - popCount + i + POP_MAX) % POP_MAX;
      const tot = popHerb[idx] + popCarn[idx];
      i === 0 ? ctx.moveTo(i * xStep, ch - 3 - tot * yScale)
              : ctx.lineTo(i * xStep, ch - 3 - tot * yScale);
    }
    ctx.stroke();

    ctx.font = '9px -apple-system,sans-serif';
    ctx.fillStyle = '#3b3'; ctx.fillText('herb', 4, 11);
    ctx.fillStyle = '#c33'; ctx.fillText('carn', 4, 22);
  }

  // ── Brain diagram ──────────────────────────────────────────────────────
  drawBrain(ctx, cre, w, h, detail) {
    ctx.clearRect(0, 0, w, h);
    if (!cre || !w || !h) return;

    const W    = cre.W;
    const acts = cre.acts;
    const L    = W.length - 1;

    const layers = [W[0].cols];
    for (let l = 0; l < L; l++) layers.push(W[l].rows);
    layers.push(W[W.length - 1].rows);

    const nL   = layers.length;
    const xGap = w / (nL + 1);
    const xPos = layers.map((_, li) => xGap * (li + 1));
    const yPos = layers.map(cnt => {
      const gap = h / (cnt + 1);
      return Array.from({ length: cnt }, (_, i) => gap * (i + 1));
    });

    const MAX_VIS = 28;

    if (detail) {
      // Rich view for selected creature (shown in sidebar creatureInfo panel)
      const INPUT_LABELS  = ['dx', 'dy', 'dist', 'nrg', '1'];
      const OUTPUT_LABELS = ['mode', 'push', 'rep'];

      // Edges: green=positive, red=negative, width ∝ magnitude
      for (let li = 0; li < nL - 1; li++) {
        const sN = layers[li], dN = layers[li + 1];
        if (sN > MAX_VIS || dN > MAX_VIS) continue;
        for (let s = 0; s < sN; s++) {
          for (let dst = 0; dst < dN; dst++) {
            const wv  = W[li].get(dst, s);
            const mag = Math.abs(wv);
            if (mag < 0.06) continue;
            const alpha = Math.min(0.85, mag * 0.9);
            const lw    = Math.max(0.4, Math.min(3.5, mag * 2.8));
            ctx.strokeStyle = wv > 0
              ? `rgba(40,210,90,${alpha})`
              : `rgba(220,55,55,${alpha})`;
            ctx.lineWidth = lw;
            ctx.beginPath();
            ctx.moveTo(xPos[li], yPos[li][s]);
            ctx.lineTo(xPos[li + 1], yPos[li + 1][dst]);
            ctx.stroke();
          }
        }
      }

      const nodeR = Math.max(5, Math.min(13, xGap * 0.26));
      const FONT  = `bold ${Math.max(7, Math.round(nodeR * 0.85))}px -apple-system,sans-serif`;

      for (let li = 0; li < nL; li++) {
        const cnt    = layers[li];
        const actArr = acts ? acts[li] : null;

        for (let ni = 0; ni < cnt; ni++) {
          if (cnt > MAX_VIS) {
            ctx.fillStyle = '#444466';
            ctx.font = `${Math.round(nodeR * 1.8)}px sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(`×${cnt}`, xPos[li], h / 2);
            break;
          }

          const x   = xPos[li], y = yPos[li][ni];
          const act = actArr ? actArr[ni] : 0;
          const absAct = Math.abs(act);

          if (absAct > 0.3) {
            ctx.shadowColor = act > 0 ? '#44ff88' : '#ff4444';
            ctx.shadowBlur  = nodeR * absAct * 1.4;
          }

          let fillColor;
          if (li === 0)        fillColor = '#1a3a8a';
          else if (li === nL-1) fillColor = '#7a3000';
          else {
            const L_pct = Math.round(12 + ((act + 1) / 2) * 55);
            fillColor = `hsl(230,60%,${L_pct}%)`;
          }
          ctx.beginPath(); ctx.arc(x, y, nodeR, 0, 6.2832);
          ctx.fillStyle = fillColor; ctx.fill();
          ctx.shadowBlur = 0;

          if (actArr) {
            ctx.beginPath(); ctx.arc(x, y, nodeR * 0.55, 0, 6.2832);
            ctx.fillStyle = act > 0
              ? `rgba(60,255,120,${Math.min(0.9, act * 1.1)})`
              : `rgba(255,60,60,${Math.min(0.9, -act * 1.1)})`;
            ctx.fill();
          }

          ctx.beginPath(); ctx.arc(x, y, nodeR, 0, 6.2832);
          ctx.strokeStyle = li === 0 ? '#4488ff' : li === nL-1 ? '#ff8833' : '#555588';
          ctx.lineWidth = 1.2; ctx.stroke();

          ctx.font = FONT; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          if (li === 0 && ni < INPUT_LABELS.length) {
            ctx.fillStyle = '#7799cc';
            ctx.fillText(INPUT_LABELS[ni], x, y + nodeR + 2);
          } else if (li === nL - 1 && ni < OUTPUT_LABELS.length) {
            ctx.fillStyle = '#cc7733';
            ctx.fillText(OUTPUT_LABELS[ni], x, y + nodeR + 2);
          }
        }
      }
    } else {
      // Fast mini view for the top-3 panels
      ctx.lineWidth = 0.7;
      for (let li = 0; li < nL - 1; li++) {
        const sN = layers[li], dN = layers[li + 1];
        if (sN > MAX_VIS || dN > MAX_VIS) continue;
        const buckets = {};
        for (let s = 0; s < sN; s++) {
          for (let dst = 0; dst < dN; dst++) {
            const wv  = W[li].get(dst, s);
            const col = wv > 0 ? '#2a9' : '#933';
            (buckets[col] = buckets[col] || []).push(
              xPos[li], yPos[li][s], xPos[li+1], yPos[li+1][dst]);
          }
        }
        for (const [col, pts] of Object.entries(buckets)) {
          ctx.strokeStyle = col; ctx.beginPath();
          for (let p = 0; p < pts.length; p += 4) {
            ctx.moveTo(pts[p], pts[p+1]); ctx.lineTo(pts[p+2], pts[p+3]);
          }
          ctx.stroke();
        }
      }
      const nodeR = Math.max(2, Math.min(5, xGap * 0.18));
      for (let li = 0; li < nL; li++) {
        if (layers[li] > 40) {
          ctx.fillStyle = '#445566';
          ctx.font = `${Math.round(nodeR * 2.5)}px sans-serif`;
          ctx.fillText(`×${layers[li]}`, xPos[li] - nodeR * 2, h / 2);
          continue;
        }
        ctx.fillStyle = li === 0 ? '#3366aa' : li === nL-1 ? '#aa6622' : '#888';
        ctx.beginPath();
        for (let ni = 0; ni < layers[li]; ni++) {
          ctx.moveTo(xPos[li] + nodeR, yPos[li][ni]);
          ctx.arc(xPos[li], yPos[li][ni], nodeR, 0, 6.2832);
        }
        ctx.fill();
      }
    }
  }

  drawBrainPanels(topIdxs, selectedIdx) {
    for (let k = 0; k < 3; k++) {
      const c   = this.bCanvases[k];
      const cre = (topIdxs[k] !== undefined) ? creatures[topIdxs[k]] : null;
      this.drawBrain(this.bCtxs[k], cre, c.clientWidth, c.clientHeight, false);
    }
    if (selectedIdx >= 0 && selectedIdx < creatures.length) {
      const dc = this.dCanvas;
      this.drawBrain(this.dCtx, creatures[selectedIdx], dc.clientWidth, dc.clientHeight, true);
    }
  }
}

// ─── UI Controller ─────────────────────────────────────────────────────────
class UIController {
  constructor(renderer) {
    this.renderer   = renderer;
    // FIX 6: store stable creature ID instead of fragile array index
    this.selectedId = -1;
    this._paused    = false;

    this._tickEl      = document.getElementById('tickDisplay');
    this._popEl       = document.getElementById('popDisplay');
    this._brainsRow   = document.getElementById('brainsRow');
    this._infoPanel   = document.getElementById('creatureInfo');
    this._panelTitle  = document.getElementById('panelTitle');
    this._panelStats  = document.getElementById('panelStats');
    this._hint        = document.getElementById('hint');
    this._hintShown   = false;
    this._brainLabels = [0,1,2].map(i => document.getElementById(`brainLabel${i}`));

    const wc = renderer.wCanvas;

    // Tap / click to select a creature
    wc.addEventListener('touchstart', e => {
      e.preventDefault();
      if (e.touches.length !== 1) return;
      const t    = e.changedTouches[0];
      const rect = wc.getBoundingClientRect();
      this._onTap(renderer.canvasToWorld(t.clientX - rect.left, t.clientY - rect.top));
    }, { passive: false });

    wc.addEventListener('touchmove', e => { e.preventDefault(); }, { passive: false });

    wc.addEventListener('click', e => {
      const rect = wc.getBoundingClientRect();
      this._onTap(renderer.canvasToWorld(e.clientX - rect.left, e.clientY - rect.top));
    });

    // Close creature info
    document.getElementById('closePanelBtn').addEventListener('click', () => this._closePanel());

    // Pause
    document.getElementById('pauseBtn').addEventListener('click', () => {
      this._paused = !this._paused;
      document.getElementById('pauseBtn').textContent = this._paused ? '▶' : '⏸';
    });

    // Speed slider — initialise label immediately
    const speedSlider = document.getElementById('speedSlider');
    const speedLabel  = document.getElementById('speedLabel');
    speedLabel.textContent = SPEED_LABELS[speedSlider.value | 0];
    speedSlider.addEventListener('input', () => {
      speedLabel.textContent = SPEED_LABELS[speedSlider.value | 0];
    });

    // Settings panel
    const configPanel = document.getElementById('configPanel');
    document.getElementById('settingsBtn').addEventListener('click', () => {
      configPanel.classList.toggle('open');
    });
    document.getElementById('closeConfigBtn').addEventListener('click', () => {
      configPanel.classList.remove('open');
    });

    // Config sliders — live label updates
    const cfgPairs = [
      ['cfgN0','cfgN0Val'], ['cfgNpred','cfgNpredVal'],
      ['cfgFood','cfgFoodVal'], ['cfgWorld','cfgWorldVal'], ['cfgSeed','cfgSeedVal'],
    ];
    for (const [id, valId] of cfgPairs) {
      const el = document.getElementById(id);
      const vl = document.getElementById(valId);
      el.addEventListener('input', () => { vl.textContent = el.value; });
    }

    // Restart button
    document.getElementById('restartBtn').addEventListener('click', () => {
      P.N0        = parseInt(document.getElementById('cfgN0').value);
      P.Npred     = parseInt(document.getElementById('cfgNpred').value);
      P.foodCount = parseInt(document.getElementById('cfgFood').value);
      P.worldSize = parseInt(document.getElementById('cfgWorld').value);
      const seed  = parseInt(document.getElementById('cfgSeed').value);
      configPanel.classList.remove('open');
      this._closePanel();
      initWorld(seed);
      renderer.resize();
    });

    // Save / Load
    document.getElementById('saveBtn').addEventListener('click', () => {
      configPanel.classList.remove('open');
      this._save();
    });
    document.getElementById('loadBtn').addEventListener('click', () => {
      configPanel.classList.remove('open');
      document.getElementById('fileInput').click();
    });
    document.getElementById('fileInput').addEventListener('change', e => {
      const f = e.target.files[0];
      if (!f) return;
      const r = new FileReader();
      r.onload = ev => this._load(ev.target.result);
      r.readAsText(f);
      e.target.value = '';
    });

    // Prevent body scroll on touch
    document.addEventListener('touchmove', e => e.preventDefault(), { passive: false });
  }

  get paused() { return this._paused; }

  get ticksPerFrame() {
    const v = document.getElementById('speedSlider').value | 0;
    return SPEED_STEPS[v];
  }

  // FIX 6: computed property — looks up current index from stable ID each call
  get selectedIdx() {
    if (this.selectedId < 0) return -1;
    return creatures.findIndex(c => c.id === this.selectedId);
  }

  _onTap({ x, y }) {
    if (!this._hintShown) {
      this._hint.classList.add('hidden');
      this._hintShown = true;
    }

    const hitR = Math.max(P.bodyRadius * 8, 12);
    let bestD = hitR, bestK = -1;
    for (let k = 0; k < creatures.length; k++) {
      const c = creatures[k];
      const dist = hypot2(torus(x - c.posX, P.worldSize), torus(y - c.posY, P.worldSize));
      if (dist < bestD) { bestD = dist; bestK = k; }
    }

    if (bestK >= 0) {
      this.selectedId = creatures[bestK].id;   // store stable ID
      this._showInfo();
    } else {
      this._closePanel();
    }
  }

  // FIX 5: show creature info inside sidebar (never covers world canvas)
  _showInfo() {
    this._brainsRow.style.display = 'none';
    this._infoPanel.classList.add('open');
  }

  _closePanel() {
    this.selectedId = -1;
    this._infoPanel.classList.remove('open');
    this._brainsRow.style.display = '';   // restore flex layout
  }

  updateHeader(nHerb, nCarn) {
    this._tickEl.textContent = `Tick ${tick}`;
    this._popEl.textContent  = `H:${nHerb}  C:${nCarn}`;
  }

  updateBrainLabels(topIdxs) {
    for (let k = 0; k < 3; k++) {
      const idx = topIdxs[k];
      if (idx !== undefined && idx < creatures.length) {
        const c = creatures[idx];
        this._brainLabels[k].textContent =
          `${c.type === 'carn' ? '🔴' : '🟢'} F${c.fitness.toFixed(0)} E${c.energy.toFixed(0)}`;
      } else {
        this._brainLabels[k].textContent = `Top Fit #${k+1}`;
      }
    }
  }

  updatePanelStats() {
    const idx = this.selectedIdx;   // computed from selectedId
    if (idx < 0) return;
    const c = creatures[idx];
    const layerStr = [c.W[0].cols, ...c.W.map(m => m.rows)].join('→');
    const speed    = hypot2(c.velX, c.velY).toFixed(1);
    const typeName = c.type === 'carn' ? '🔴 Carnivore' : '🟢 Herbivore';
    this._panelTitle.textContent = `${typeName}  (id ${c.id})`;
    this._panelStats.innerHTML =
      `E:${c.energy.toFixed(0)} · F:${c.fitness.toFixed(0)} · Age:${c.age} · Spd:${speed}<br>` +
      `Brain: [${layerStr}]`;
  }

  _save() {
    const state = {
      tick, rngSeed, nextCreatureId,
      creatures: creatures.map(c => ({
        id: c.id, posX: c.posX, posY: c.posY, velX: c.velX, velY: c.velY,
        angle: c.angle, energy: c.energy, fitness: c.fitness,
        age: c.age, type: c.type, lineage: c.lineage,
        busyTime: c.busyTime, fadeTick: c.fadeTick, fadeInit: c.fadeInit,
        immature: c.immature, eatCooldown: c.eatCooldown, color: c.color,
        W: c.W.map(m => m.toJSON()), act: Array.from(c.act),
      })),
      food,
      popHerb: Array.from(popHerb.subarray(0, popCount)),
      popCarn: Array.from(popCarn.subarray(0, popCount)),
      popHead, popCount,
    };
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(new Blob([JSON.stringify(state)], { type: 'application/json' }));
    a.download = `bibites_${tick}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }

  _load(text) {
    try {
      const s = JSON.parse(text);
      tick           = s.tick || 0;
      rngSeed        = s.rngSeed || 1;
      nextCreatureId = s.nextCreatureId || 0;
      food           = s.food || [];
      creatures = s.creatures.map(sc => ({
        id: sc.id ?? nextCreatureId++,
        posX: sc.posX, posY: sc.posY, velX: sc.velX || 0, velY: sc.velY || 0,
        angle: sc.angle, energy: sc.energy, fitness: sc.fitness,
        age: sc.age, type: sc.type, lineage: sc.lineage || 0,
        busyTime: sc.busyTime || 0, fadeTick: sc.fadeTick || 0,
        fadeInit: sc.fadeInit || 0, immature: sc.immature || 0,
        eatCooldown: sc.eatCooldown || 0, deathStart: 0,
        color: sc.color || [1,1,1],
        W: sc.W.map(m => Matrix.fromJSON(m)),
        act: new Float64Array(sc.act || [0,0,0]),
        acts: null,
      }));
      baseW = creatures[0]?.W.map(m => m.clone()) ?? baseW;
      popHead = s.popHead || 0; popCount = s.popCount || 0;
      if (s.popHerb) for (let i = 0; i < s.popHerb.length; i++) popHerb[i] = s.popHerb[i];
      if (s.popCarn) for (let i = 0; i < s.popCarn.length; i++) popCarn[i] = s.popCarn[i];
    } catch (e) {
      alert('Load failed: ' + e.message);
    }
  }
}

// ─── Game Loop ─────────────────────────────────────────────────────────────
class GameLoop {
  constructor(renderer, ui) {
    this.renderer    = renderer;
    this.ui          = ui;
    this._rafId      = null;
    this._lastTs     = 0;
    this._tickAccum  = 0;
    this._brainFrame = 0;
    this._BRAIN_EVERY = 6;
  }

  start() {
    this._lastTs = performance.now();
    this._rafId  = requestAnimationFrame(ts => this._loop(ts));
  }

  _loop(ts) {
    this._rafId = requestAnimationFrame(t => this._loop(t));

    const dt = Math.min(ts - this._lastTs, 100);
    this._lastTs = ts;

    if (!this.ui.paused) {
      const tpf = this.ui.ticksPerFrame;
      this._tickAccum += tpf;
      const steps = Math.floor(this._tickAccum);
      this._tickAccum -= steps;
      for (let i = 0; i < steps; i++) { simStep(); tick++; }

      const nCarn = countType('carn');
      const nHerb = creatures.length - nCarn;
      popHerb[popHead] = nHerb; popCarn[popHead] = nCarn;
      popHead = (popHead + 1) % POP_MAX;
      if (popCount < POP_MAX) popCount++;
      this.ui.updateHeader(nHerb, nCarn);
    }

    // FIX 6: detect creature death via stable ID lookup (not array-length clamp)
    if (this.ui.selectedId >= 0 && this.ui.selectedIdx < 0) {
      // The selected creature has died — close the info panel
      this.ui._closePanel();
    }

    // Cache selectedIdx (calls findIndex once per frame)
    const selectedIdx = this.ui.selectedIdx;

    const topIdxs = topByFitness(3);
    this.renderer.drawWorld(selectedIdx);
    this.renderer.drawPopGraph();

    this._brainFrame++;
    const brainDirty = selectedIdx !== this.renderer._lastSel;
    if (brainDirty || this._brainFrame % this._BRAIN_EVERY === 0) {
      this.renderer.drawBrainPanels(topIdxs, selectedIdx);
      this.ui.updateBrainLabels(topIdxs);
      if (selectedIdx >= 0) this.ui.updatePanelStats();
      this.renderer._lastSel = selectedIdx;
    }

    if (creatures.length === 0) {
      this.ui._tickEl.textContent = 'All extinct — tap ⚙ to restart';
      cancelAnimationFrame(this._rafId);
    }
  }
}

function countType(type) {
  let c = 0;
  for (const cr of creatures) if (cr.type === type) c++;
  return c;
}

function topByFitness(n) {
  if (!creatures.length) return [];
  return creatures
    .map((c, i) => ({ i, f: c.fitness }))
    .sort((a, b) => b.f - a.f)
    .slice(0, n)
    .map(r => r.i);
}

// ─── Bootstrap ─────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initWorld(1);

  const renderer = new Renderer();
  const ui       = new UIController(renderer);
  const loop     = new GameLoop(renderer, ui);

  const worldWrap = document.getElementById('worldWrap');

  function applyLayout() {
    const isLandscape = window.innerWidth > window.innerHeight;
    if (!isLandscape) {
      worldWrap.style.height = worldWrap.clientWidth + 'px';
    } else {
      worldWrap.style.height = '';
    }
  }

  function onResize() {
    applyLayout();
    requestAnimationFrame(() => renderer.resize());
  }

  window.addEventListener('resize', onResize);
  window.addEventListener('orientationchange', () => setTimeout(onResize, 100));

  requestAnimationFrame(() => {
    applyLayout();
    requestAnimationFrame(() => {
      renderer.resize();
      loop.start();
      setTimeout(() => document.getElementById('hint').classList.add('hidden'), 8000);
    });
  });
});
