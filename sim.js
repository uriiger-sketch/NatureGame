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
  // signed toroidal delta: equivalent to MATLAB mod(x + w/2, w) - w/2
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

function weightColor(w) {
  const g = Math.round(((Math.max(-1, Math.min(1, w)) + 1) / 2) * 255);
  return `rgb(${g},${g},${g})`;
}

// ─── Matrix (row-major, Float64Array backed) ───────────────────────────────
class Matrix {
  constructor(rows, cols, data) {
    this.rows = rows;
    this.cols = cols;
    this.data = (data instanceof Float64Array)
      ? data
      : new Float64Array(rows * cols);
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

// matrix × column vector, applying tanh to output
function matMulTanh(M, x) {
  const out = new Float64Array(M.rows);
  for (let r = 0; r < M.rows; r++) {
    let s = 0;
    for (let c = 0; c < M.cols; c++) s += M.data[r * M.cols + c] * x[c];
    out[r] = Math.tanh(s);
  }
  return out;
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
    posX, posY, velX: 0, velY: 0,
    angle: rng() * 2 * Math.PI,
    W,
    energy: (energy !== undefined) ? energy : 55 + rng() * 12,
    act: new Float64Array(P.nOutput),
    color: [1, 1, 1],
    age: 0, fitness: 30,
    type,
    busyTime: 0, fadeTick: 0, fadeInit: 0, immature: 0,
    lineage,
  };
  c.color = type === 'carn' ? [1, 0.1, 0.1] : geneColor(c);
  return c;
}

function polyVerts(cre) {
  const Wlast = cre.W[cre.W.length - 1];
  const sig = Math.tanh(Wlast.sumSign() / (Wlast.data.length || 1));
  const ns  = Math.max(3, Math.min(8, 3 + Math.round((sig + 1) * 2.5)));
  const sG  = 0.8 + 0.6 / (1 + Math.exp(-Wlast.stdAll() * 3));
  let R = P.bodyRadius * sG;
  if (cre.fadeTick > 0 && cre.fadeInit > 0)      R *= cre.fadeTick / cre.fadeInit;
  else if (cre.fadeInit > 0)                       R = 0;
  const vx = new Array(ns), vy = new Array(ns);
  for (let i = 0; i < ns; i++) {
    const th = (i / ns) * 2 * Math.PI + cre.angle;
    vx[i] = cre.posX + R * Math.cos(th);
    vy[i] = cre.posY + R * Math.sin(th);
  }
  return { vx, vy, R };
}

// ─── World state ───────────────────────────────────────────────────────────
let creatures = [];
let food      = [];  // [{x, y}]
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

// ─── Neural network ────────────────────────────────────────────────────────
function nnForward(W, inp) {
  let x = inp;
  for (let l = 0; l < W.length - 1; l++) x = matMulTanh(W[l], x);
  return matMulTanh(W[W.length - 1], x);
}

// ─── Structural mutation ───────────────────────────────────────────────────
function mutateNetworkStructure(W) {
  const outSize = P.nOutput;
  let L = W.length - 1;  // hidden layer count

  // maybe add a hidden layer before output
  if (L < P.maxHiddenLayers && rng() < P.addLayerProb) {
    const nPrev = W[W.length - 2].rows;
    const nNew  = Math.max(2, Math.min(P.maxNeurons, Math.round(nPrev * (0.7 + 0.6 * rng()))));
    W = [...W.slice(0, W.length - 1), Matrix.randn(nNew, nPrev, 0.3), Matrix.randn(outSize, nNew, 0.3)];
    L++;
  }

  // maybe delete a hidden layer
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

  // maybe add a neuron to a random hidden layer
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

  // extract state
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

  for (let k = 0; k < n; k++) {
    const c = creatures[k];
    posX[k] = c.posX; posY[k] = c.posY;
    velX[k] = c.velX; velY[k] = c.velY;
    ang[k]  = c.angle;
    E[k]    = c.energy;   fit[k] = c.fitness;
    age[k]  = c.age;      busy[k] = c.busyTime;
    fade[k] = c.fadeTick; fadeInit[k] = c.fadeInit;
    imm[k]  = c.immature;
    isCarn[k] = c.type === 'carn' ? 1 : 0;
    lineage[k] = c.lineage | 0;
  }

  // tick timers
  for (let k = 0; k < n; k++) {
    age[k]++;
    fit[k] += P.fitTick;
    if (busy[k] > 0) busy[k]--;
    if (fade[k] > 0) fade[k]--;
    if (imm[k]  > 0) imm[k]--;
  }

  const WS = P.worldSize, SR = P.senseRadius;

  // ── sense ──────────────────────────────────────────────────────────────
  const dx = new Float64Array(n);
  const dy = new Float64Array(n);
  const d  = new Float64Array(n).fill(SR);

  // herbivores: toward food, or flee nearest predator if closer
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
      // flee: vector away from predator (herb - pred)
      dx[h] = pdx; dy[h] = pdy; d[h] = Math.sqrt(predD2);
    } else if (foodD2 < SR * SR) {
      // forage: vector toward food (food - herb = -(herb - food))
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
  const act = new Array(n);
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) {
      act[k] = new Float64Array(P.nOutput);
      continue;
    }
    act[k] = nnForward(creatures[k].W, new Float64Array([
      dx[k], dy[k],
      d[k] / SR,
      E[k] / P.reproEnergy,
      1,
    ]));
  }

  // ── interpret outputs → motion ─────────────────────────────────────────
  const MT = P.modeThresh;
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) { velX[k] = 0; velY[k] = 0; continue; }
    const modeRaw  = act[k][0];
    const thrustIn = Math.max(act[k][1], 0);
    const mode     = modeRaw > MT ? 1 : modeRaw < -MT ? -1 : 0;

    let theta = Math.atan2(dy[k], dx[k]);
    if (mode === -1) theta += Math.PI;  // FLIGHT: reverse direction

    ang[k] += 3 * P.dt * wrapToPi(theta - ang[k]);

    if (mode !== 0) {
      velX[k] += thrustIn * Math.cos(ang[k]);
      velY[k] += thrustIn * Math.sin(ang[k]);
    }

    // clamp speed
    const spd = hypot2(velX[k], velY[k]);
    if (spd > P.maxSpeed) {
      const sc = P.maxSpeed / spd;
      velX[k] *= sc; velY[k] *= sc;
    }
  }

  // ── physics (integrate + energy decay) ────────────────────────────────
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) continue;
    posX[k] = posmod(posX[k] + velX[k] * P.dt, WS);
    posY[k] = posmod(posY[k] + velY[k] * P.dt, WS);
    const spd = hypot2(velX[k], velY[k]);
    E[k] -= P.energyDecay + P.moveCost * spd;
    if (isCarn[k]) E[k] -= P.predMoveCost * spd;
    if (E[k] > 200) E[k] = 200;  // energy cap prevents runaway accumulation
  }

  // ── herbivores eat food ────────────────────────────────────────────────
  const r2     = P.bodyRadius * P.bodyRadius;
  const eaten  = new Uint8Array(food.length);
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

  // ── predator attacks ───────────────────────────────────────────────────
  const EAT_TICKS = 10;
  for (let cc = 0; cc < n; cc++) {
    if (!isCarn[cc] || busy[cc] > 0 || fade[cc] > 0) continue;
    let bestD = 2 * P.bodyRadius, bestPrey = -1;
    for (let p = 0; p < n; p++) {
      if (isCarn[p] || fade[p] > 0 || imm[p] > 0) continue;  // only eat herbs
      const dist = hypot2(torus(posX[p] - posX[cc], WS), torus(posY[p] - posY[cc], WS));
      if (dist < bestD) { bestD = dist; bestPrey = p; }
    }
    if (bestPrey >= 0) {
      busy[cc]           = EAT_TICKS;
      fade[bestPrey]     = EAT_TICKS;
      fadeInit[bestPrey] = EAT_TICKS;
      E[cc] += E[bestPrey] / 2;  // absorb half of prey's energy
      fit[cc] += P.fitFood;
    }
  }

  // ── deaths ─────────────────────────────────────────────────────────────
  const keep = new Uint8Array(n).fill(1);
  for (let k = 0; k < n; k++) {
    const tooOld  = isCarn[k] ? age[k] >= P.maxAgeCarn : age[k] >= P.maxAgeHerb;
    const fadeDead = fade[k] === 0 && fadeInit[k] > 0;
    if (tooOld || fadeDead || E[k] <= 0) keep[k] = 0;
  }

  // write back survivors
  const alive = [];
  for (let k = 0; k < n; k++) {
    if (!keep[k]) continue;
    const c = creatures[k];
    c.posX = posX[k]; c.posY = posY[k];
    c.velX = velX[k]; c.velY = velY[k];
    c.angle    = ang[k];
    c.energy   = E[k];   c.fitness = fit[k];
    c.age      = age[k]; c.act     = act[k];
    c.busyTime = busy[k];
    c.fadeTick = fade[k]; c.fadeInit = fadeInit[k];
    c.immature = imm[k];
    alive.push(c);
  }
  creatures = alive;

  // ── reproduction ───────────────────────────────────────────────────────
  // reproduce when NN says so OR when energy exceeds threshold (safety net
  // for creatures whose NN never outputs a positive reproduce signal)
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
  chi.immature = Math.round(2 / P.dt);  // 10-tick invulnerability shield
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
    this.wCanvas  = document.getElementById('worldCanvas');
    this.wCtx     = this.wCanvas.getContext('2d');
    this.pCanvas  = document.getElementById('popCanvas');
    this.pCtx     = this.pCanvas.getContext('2d');
    this.bCanvases = [0,1,2].map(i => document.getElementById(`brain${i}`));
    this.bCtxs     = this.bCanvases.map(c => c.getContext('2d'));
    this.dCanvas  = document.getElementById('brainDetail');
    this.dCtx     = this.dCanvas.getContext('2d');

    this.dpr     = 1;
    this.scale   = 1;
    this.offsetX = 0;
    this.offsetY = 0;
  }

  // Call after layout is settled
  resize() {
    this.dpr = Math.min(window.devicePixelRatio || 1, 3);
    this._size(this.wCanvas,  this.wCtx);
    this._size(this.pCanvas,  this.pCtx);
    this.bCanvases.forEach((c, i) => this._size(c, this.bCtxs[i]));
    this._size(this.dCanvas,  this.dCtx);

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
      canvas.width  = pw;
      canvas.height = ph;
    }
    // always reset transform so DPR scaling is correct
    ctx.setTransform(d, 0, 0, d, 0, 0);
  }

  // world → canvas logical
  wx(x) { return this.offsetX + x * this.scale; }
  wy(y) { return this.offsetY + (P.worldSize - y) * this.scale; }

  // canvas logical → world (for tap detection)
  canvasToWorld(cx, cy) {
    return {
      x: (cx - this.offsetX) / this.scale,
      y: P.worldSize - (cy - this.offsetY) / this.scale,
    };
  }

  // ── World ──────────────────────────────────────────────────────────────
  drawWorld(selectedIdx) {
    const ctx = this.wCtx;
    const cw  = this.wCanvas.clientWidth;
    const ch  = this.wCanvas.clientHeight;

    ctx.fillStyle = '#050509';
    ctx.fillRect(0, 0, cw, ch);

    // food (all in one path)
    ctx.fillStyle = '#2ecc40';
    ctx.beginPath();
    for (const f of food) {
      const fx = this.wx(f.x), fy = this.wy(f.y);
      ctx.moveTo(fx + 2.5, fy);
      ctx.arc(fx, fy, 2.5, 0, 6.2832);
    }
    ctx.fill();

    // creatures
    for (let k = 0; k < creatures.length; k++) {
      const c = creatures[k];
      const { vx, vy, R } = polyVerts(c);
      if (R <= 0 || vx.length === 0) continue;
      const [r, g, b] = c.color;
      ctx.fillStyle = `rgb(${(r*255)|0},${(g*255)|0},${(b*255)|0})`;
      ctx.beginPath();
      ctx.moveTo(this.wx(vx[0]), this.wy(vy[0]));
      for (let i = 1; i < vx.length; i++) ctx.lineTo(this.wx(vx[i]), this.wy(vy[i]));
      ctx.closePath();
      ctx.fill();
    }

    // selection halo
    if (selectedIdx >= 0 && selectedIdx < creatures.length) {
      const c = creatures[selectedIdx];
      ctx.strokeStyle = '#ffdd00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(this.wx(c.posX), this.wy(c.posY), P.bodyRadius * 3 * this.scale, 0, 6.2832);
      ctx.stroke();
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
      ctx.strokeStyle = color;
      ctx.lineWidth   = lw;
      ctx.beginPath();
      for (let i = 0; i < popCount; i++) {
        const idx = (popHead - popCount + i + POP_MAX) % POP_MAX;
        const x = i * xStep;
        const y = ch - 3 - arr[idx] * yScale;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    line(popHerb, '#3b3', 1.5);
    line(popCarn, '#c33', 1.5);

    // total
    ctx.strokeStyle = '#667';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    for (let i = 0; i < popCount; i++) {
      const idx = (popHead - popCount + i + POP_MAX) % POP_MAX;
      const tot = popHerb[idx] + popCarn[idx];
      i === 0 ? ctx.moveTo(i * xStep, ch - 3 - tot * yScale)
              : ctx.lineTo(i * xStep, ch - 3 - tot * yScale);
    }
    ctx.stroke();

    // legend
    ctx.font = '9px -apple-system,sans-serif';
    ctx.fillStyle = '#3b3'; ctx.fillText('herb', 4, 11);
    ctx.fillStyle = '#c33'; ctx.fillText('carn', 4, 22);
  }

  // ── Brain diagram ──────────────────────────────────────────────────────
  drawBrain(ctx, cre, w, h) {
    ctx.clearRect(0, 0, w, h);
    if (!cre || !w || !h) return;

    const W = cre.W;
    const L = W.length - 1;  // hidden layer count

    // layer sizes: [nInput, h0, h1, ..., nOutput]
    const layers = [W[0].cols];
    for (let l = 0; l < L; l++) layers.push(W[l].rows);
    layers.push(W[W.length - 1].rows);

    const nL    = layers.length;
    const xGap  = w / (nL + 1);
    const xPos  = layers.map((_, li) => xGap * (li + 1));
    const yPos  = layers.map(cnt => {
      const gap = h / (cnt + 1);
      return Array.from({ length: cnt }, (_, i) => gap * (i + 1));
    });

    // edges (grouped by color bucket for fewer stroke calls)
    const MAX_N = 18;
    ctx.lineWidth = 0.7;
    for (let li = 0; li < nL - 1; li++) {
      const sN = layers[li], dN = layers[li + 1];
      if (sN > MAX_N || dN > MAX_N) continue;
      const buckets = {};
      for (let s = 0; s < sN; s++) {
        for (let dst = 0; dst < dN; dst++) {
          const col = weightColor(W[li].get(dst, s));
          (buckets[col] = buckets[col] || []).push(xPos[li], yPos[li][s], xPos[li+1], yPos[li+1][dst]);
        }
      }
      for (const [col, pts] of Object.entries(buckets)) {
        ctx.strokeStyle = col;
        ctx.beginPath();
        for (let p = 0; p < pts.length; p += 4) {
          ctx.moveTo(pts[p], pts[p+1]);
          ctx.lineTo(pts[p+2], pts[p+3]);
        }
        ctx.stroke();
      }
    }

    // nodes
    const nodeR = Math.max(2, Math.min(5, xGap * 0.18));
    for (let li = 0; li < nL; li++) {
      if (layers[li] > 40) {
        // just label the count
        ctx.fillStyle = '#556';
        ctx.font = `${Math.round(nodeR * 2.5)}px sans-serif`;
        ctx.fillText(`×${layers[li]}`, xPos[li] - nodeR * 2, h / 2);
        continue;
      }
      ctx.fillStyle = li === 0 ? '#5599ff' : li === nL - 1 ? '#ffaa33' : '#cccccc';
      ctx.beginPath();
      for (let ni = 0; ni < layers[li]; ni++) {
        ctx.moveTo(xPos[li] + nodeR, yPos[li][ni]);
        ctx.arc(xPos[li], yPos[li][ni], nodeR, 0, 6.2832);
      }
      ctx.fill();
    }
  }

  // ── Brain panel updates ────────────────────────────────────────────────
  drawBrainPanels(topIdxs, selectedIdx) {
    for (let k = 0; k < 3; k++) {
      const c  = this.bCanvases[k];
      const cre = (topIdxs[k] !== undefined) ? creatures[topIdxs[k]] : null;
      this.drawBrain(this.bCtxs[k], cre, c.clientWidth, c.clientHeight);
    }
    if (selectedIdx >= 0 && selectedIdx < creatures.length) {
      const dc = this.dCanvas;
      this.drawBrain(this.dCtx, creatures[selectedIdx], dc.clientWidth, dc.clientHeight);
    }
  }
}

// ─── UI Controller ─────────────────────────────────────────────────────────
class UIController {
  constructor(renderer) {
    this.renderer    = renderer;
    this.selectedIdx = -1;
    this._paused     = false;

    this._tickEl    = document.getElementById('tickDisplay');
    this._popEl     = document.getElementById('popDisplay');
    this._panel     = document.getElementById('creaturePanel');
    this._panelTitle = document.getElementById('panelTitle');
    this._panelStats = document.getElementById('panelStats');
    this._hint      = document.getElementById('hint');
    this._hintShown = false;
    this._brainLabels = [0,1,2].map(i => document.getElementById(`brainLabel${i}`));

    // touch on world canvas
    const wc = renderer.wCanvas;
    wc.addEventListener('touchstart', e => {
      e.preventDefault();
      const t    = e.changedTouches[0];
      const rect = wc.getBoundingClientRect();
      this._onTap(
        renderer.canvasToWorld(t.clientX - rect.left, t.clientY - rect.top)
      );
    }, { passive: false });

    wc.addEventListener('click', e => {
      const rect = wc.getBoundingClientRect();
      this._onTap(renderer.canvasToWorld(e.clientX - rect.left, e.clientY - rect.top));
    });

    // close panel on handle drag or tap outside
    document.getElementById('closePanelBtn').addEventListener('click', () => this._closePanel());
    document.getElementById('panelHandle').addEventListener('click', () => this._closePanel());

    document.getElementById('pauseBtn').addEventListener('click', () => {
      this._paused = !this._paused;
      document.getElementById('pauseBtn').textContent = this._paused ? 'Resume' : 'Pause';
    });

    document.getElementById('saveBtn').addEventListener('click', () => this._save());
    document.getElementById('loadBtn').addEventListener('click', () => {
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

    // prevent all body scrolling on touch
    document.addEventListener('touchmove', e => e.preventDefault(), { passive: false });
  }

  get paused() { return this._paused; }

  _onTap({ x, y }) {
    // show hint once creature selected
    if (!this._hintShown) {
      this._hint.classList.add('hidden');
      this._hintShown = true;
    }

    // hit radius in world units: at least ~12 units for comfortable tap on phone
    const hitR = Math.max(P.bodyRadius * 8, 12);
    let bestD = hitR, bestK = -1;
    for (let k = 0; k < creatures.length; k++) {
      const c = creatures[k];
      const dist = hypot2(torus(x - c.posX, P.worldSize), torus(y - c.posY, P.worldSize));
      if (dist < bestD) { bestD = dist; bestK = k; }
    }

    this.selectedIdx = bestK;
    if (bestK >= 0) {
      this._panel.classList.add('open');
    } else {
      this._closePanel();
    }
  }

  _closePanel() {
    this._panel.classList.remove('open');
    this.selectedIdx = -1;
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
          `${c.type === 'carn' ? 'C' : 'H'} F${c.fitness.toFixed(0)} E${c.energy.toFixed(0)}`;
      } else {
        this._brainLabels[k].textContent = `Top Fit #${k+1}`;
      }
    }
  }

  updatePanelStats() {
    const idx = this.selectedIdx;
    if (idx < 0 || idx >= creatures.length) return;
    const c = creatures[idx];
    const layerStr = [c.W[0].cols, ...c.W.map(m => m.rows)].join('→');
    this._panelTitle.textContent =
      (c.type === 'carn' ? 'Carnivore' : 'Herbivore') + `  #${idx}`;
    this._panelStats.innerHTML =
      `Energy: ${c.energy.toFixed(1)}  &middot;  Fitness: ${c.fitness.toFixed(0)}<br>` +
      `Age: ${c.age}  &middot;  Brain: [${layerStr}]`;
  }

  _save() {
    const state = {
      tick, rngSeed,
      creatures: creatures.map(c => ({
        posX: c.posX, posY: c.posY, velX: c.velX, velY: c.velY,
        angle: c.angle, energy: c.energy, fitness: c.fitness,
        age: c.age, type: c.type, lineage: c.lineage,
        busyTime: c.busyTime, fadeTick: c.fadeTick, fadeInit: c.fadeInit,
        immature: c.immature, color: c.color,
        W: c.W.map(m => m.toJSON()),
        act: Array.from(c.act),
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
      tick    = s.tick || 0;
      rngSeed = s.rngSeed || 1;
      food    = s.food || [];
      creatures = s.creatures.map(sc => ({
        posX: sc.posX, posY: sc.posY, velX: sc.velX, velY: sc.velY,
        angle: sc.angle, energy: sc.energy, fitness: sc.fitness,
        age: sc.age, type: sc.type, lineage: sc.lineage || 0,
        busyTime: sc.busyTime || 0, fadeTick: sc.fadeTick || 0,
        fadeInit: sc.fadeInit || 0, immature: sc.immature || 0,
        color: sc.color || [1,1,1],
        W: sc.W.map(m => Matrix.fromJSON(m)),
        act: new Float64Array(sc.act || [0,0,0]),
      }));
      baseW = creatures[0]?.W.map(m => m.clone()) ?? baseW;
      popHead = s.popHead || 0;
      popCount = s.popCount || 0;
      if (s.popHerb) for (let i = 0; i < s.popHerb.length; i++) popHerb[i] = s.popHerb[i];
      if (s.popCarn) for (let i = 0; i < s.popCarn.length; i++) popCarn[i] = s.popCarn[i];
    } catch (e) {
      alert('Load failed: ' + e.message);
    }
  }
}

// ─── Game Loop ────────────────────────────────────────────────────────────
class GameLoop {
  constructor(renderer, ui) {
    this.renderer    = renderer;
    this.ui          = ui;
    this._rafId      = null;
    this._lastTs     = 0;
    this._brainFrame = 0;
    this._BRAIN_EVERY = 8;  // redraw brain panels every N frames
  }

  start() {
    this._lastTs = performance.now();
    this._rafId  = requestAnimationFrame(ts => this._loop(ts));
  }

  _loop(ts) {
    this._rafId = requestAnimationFrame(t => this._loop(t));

    const dt = Math.min(ts - this._lastTs, 100);  // cap at 100ms to survive tab-switch
    this._lastTs = ts;

    if (!this.ui.paused) {
      // run 2 sim ticks per frame (matching graphicsStride:2)
      simStep(); simStep();
      tick += 2;

      // record population
      const nCarn = countType('carn');
      const nHerb = creatures.length - nCarn;
      popHerb[popHead] = nHerb;
      popCarn[popHead] = nCarn;
      popHead = (popHead + 1) % POP_MAX;
      if (popCount < POP_MAX) popCount++;

      this.ui.updateHeader(nHerb, nCarn);
    }

    // clamp selectedIdx if that creature died
    if (this.ui.selectedIdx >= creatures.length) {
      this.ui.selectedIdx = -1;
      document.getElementById('creaturePanel').classList.remove('open');
    }

    // top-3 by fitness
    const topIdxs = topByFitness(3);

    this.renderer.drawWorld(this.ui.selectedIdx);
    this.renderer.drawPopGraph();

    this._brainFrame++;
    const brainDirty = this.ui.selectedIdx !== this.renderer._lastSel;
    if (brainDirty || this._brainFrame % this._BRAIN_EVERY === 0) {
      this.renderer.drawBrainPanels(topIdxs, this.ui.selectedIdx);
      this.ui.updateBrainLabels(topIdxs);
      if (this.ui.selectedIdx >= 0) this.ui.updatePanelStats();
      this.renderer._lastSel = this.ui.selectedIdx;
    }

    if (creatures.length === 0) {
      this.ui._tickEl.textContent = 'All extinct — reload to restart';
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

// ─── Bootstrap ────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initWorld(1);

  const renderer = new Renderer();
  const ui       = new UIController(renderer);
  const loop     = new GameLoop(renderer, ui);

  const worldWrap = document.getElementById('worldWrap');

  function applyLayout() {
    const isLandscape = window.innerWidth > window.innerHeight;
    if (!isLandscape) {
      // portrait: world canvas is a square equal to viewport width
      worldWrap.style.height = worldWrap.clientWidth + 'px';
    } else {
      worldWrap.style.height = '';
    }
  }

  function onResize() {
    applyLayout();
    // defer canvas sizing one frame so layout has settled
    requestAnimationFrame(() => renderer.resize());
  }

  window.addEventListener('resize', onResize);
  window.addEventListener('orientationchange', () => setTimeout(onResize, 100));

  // initial layout + resize, deferred two frames to be sure layout is ready
  requestAnimationFrame(() => {
    applyLayout();
    requestAnimationFrame(() => {
      renderer.resize();
      loop.start();
      // hide hint after 6 seconds
      setTimeout(() => {
        document.getElementById('hint').classList.add('hidden');
      }, 6000);
    });
  });
});
