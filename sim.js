'use strict';
// ─── Seeded PRNG (mulberry32) ──────────────────────────────────────────────
function makePRNG(seed) {
  let s = seed >>> 0;
  return function() {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng = makePRNG(1);

function randn() {
  // Box-Muller
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Math helpers ──────────────────────────────────────────────────────────
function torus(x, w) {
  // signed toroidal delta: mod(x + w/2, w) - w/2
  return ((x + w * 0.5) % w + w) % w - w * 0.5;
}

function wrapToPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

function hypot2(dx, dy) { return Math.sqrt(dx * dx + dy * dy); }

function mod(x, w) { return ((x % w) + w) % w; }

// ─── Colour helpers ────────────────────────────────────────────────────────
function hsvToRgb(h, s, v) {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const tbl = [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]];
  const [r,g,b] = tbl[i % 6];
  return [r, g, b];
}

function weightColor(w) {
  // map [-1,1] → greyscale
  const g = Math.round(((Math.max(-1, Math.min(1, w)) + 1) / 2) * 255);
  return `rgb(${g},${g},${g})`;
}

// ─── Matrix (row-major, Float64Array backed) ───────────────────────────────
class Matrix {
  constructor(rows, cols, data) {
    this.rows = rows;
    this.cols = cols;
    this.data = data instanceof Float64Array ? data : new Float64Array(rows * cols);
    if (data && !(data instanceof Float64Array)) {
      // init from nested array
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          this.data[r * cols + c] = data[r][c];
    }
  }

  static zeros(rows, cols) { return new Matrix(rows, cols); }

  static randn(rows, cols, scale = 1) {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) m.data[i] = randn() * scale;
    return m;
  }

  clone() {
    return new Matrix(this.rows, this.cols, new Float64Array(this.data));
  }

  get(r, c) { return this.data[r * this.cols + c]; }
  set(r, c, v) { this.data[r * this.cols + c] = v; }

  addNoise(sigma) {
    for (let i = 0; i < this.data.length; i++) this.data[i] += randn() * sigma;
  }

  meanAll() {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return s / this.data.length;
  }

  stdAll() {
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
    // zero-pad to common size and compute ||A-B||_F
    const R = Math.max(this.rows, other.rows);
    const C = Math.max(this.cols, other.cols);
    let s = 0;
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const a = (r < this.rows && c < this.cols) ? this.get(r, c) : 0;
        const b = (r < other.rows && c < other.cols) ? other.get(r, c) : 0;
        s += (a - b) ** 2;
      }
    }
    return Math.sqrt(s);
  }

  addRow(scale = 0.3) {
    // append one row of randn values
    const newData = new Float64Array((this.rows + 1) * this.cols);
    newData.set(this.data);
    for (let c = 0; c < this.cols; c++) newData[this.rows * this.cols + c] = randn() * scale;
    this.rows++;
    this.data = newData;
  }

  addCol(scale = 0.3) {
    // append one column of randn values
    const newData = new Float64Array(this.rows * (this.cols + 1));
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) newData[r * (this.cols + 1) + c] = this.get(r, c);
      newData[r * (this.cols + 1) + this.cols] = randn() * scale;
    }
    this.cols++;
    this.data = newData;
  }

  toJSON() {
    return { rows: this.rows, cols: this.cols, data: Array.from(this.data) };
  }

  static fromJSON(obj) {
    return new Matrix(obj.rows, obj.cols, new Float64Array(obj.data));
  }
}

// multiply Matrix by Float64Array column vector, apply tanh
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
  N0: 30,
  Npred: 6,
  worldSize: 120,
  foodCount: 80,
  foodEnergy: 22,
  predEnergyGain: -10,   // negative — predator loses extra energy per kill
  bodyRadius: 1.4,
  dt: 0.2,               // 1/5
  maxSpeed: 10,
  energyDecay: 0.1,
  moveCost: 0.05,
  predMoveCost: 0.07,
  reproEnergy: 100,
  mutationSigma: 0.3,
  nnHidden: 8,
  graphicsStride: 2,
  senseRadius: 25,
  addLayerProb: 0.18,
  delLayerProb: 0.02,
  addNeuronProb: 0.22,
  maxHiddenLayers: 5,
  maxNeurons: 70,
  fitFood: 5,
  fitTick: 0.01,
  fitRepro: 20,
  maxAgeHerb: 1500,
  maxAgeCarn: 500,
  modeThresh: 0.10,
  nInput: 5,
  nOutput: 3,
};

// ─── Creature factory ──────────────────────────────────────────────────────
function makeCreature(posX, posY, W, type, lineage, energy) {
  const c = {
    posX, posY,
    velX: 0, velY: 0,
    angle: rng() * 2 * Math.PI,
    W,                           // Array<Matrix>
    energy: energy ?? (55 + rng() * 12),
    act: new Float64Array(P.nOutput),
    color: [1, 1, 1],
    age: 0,
    fitness: 30,
    type,                        // 'herb' | 'carn'
    busyTime: 0,
    fadeTick: 0,
    fadeInit: 0,
    immature: 0,
    lineage,
  };
  c.color = type === 'carn' ? [1, 0, 0] : geneColor(c);
  return c;
}

function makeInitBrain() {
  return [
    Matrix.randn(P.nnHidden, P.nInput, 0.5),
    Matrix.randn(P.nOutput, P.nnHidden, 0.5),
  ];
}

function geneColor(cre) {
  const W1 = cre.W[0];
  const Wlast = cre.W[cre.W.length - 1];
  const h = ((W1.meanAll() * 0.4 + 0.5) % 1 + 1) % 1;
  const s = 0.6 + 0.3 / (1 + Math.exp(-Wlast.stdAll() * 2));
  return hsvToRgb(h, s, 1);
}

// ─── World state ───────────────────────────────────────────────────────────
let creatures = [];
let food = [];   // Array of {x, y}
let tick = 0;
let baseW = null;
let rngSeed = 1;

const POP_HISTORY_MAX = 1000;
let popT    = new Float32Array(POP_HISTORY_MAX);
let popHerb = new Float32Array(POP_HISTORY_MAX);
let popCarn = new Float32Array(POP_HISTORY_MAX);
let popHead = 0;
let popCount = 0;

function initWorld(seed) {
  rngSeed = seed ?? 1;
  rng = makePRNG(rngSeed);
  tick = 0;
  popHead = 0; popCount = 0;

  creatures = [];
  for (let k = 0; k < P.N0; k++) {
    const W = makeInitBrain();
    creatures.push(makeCreature(rng() * P.worldSize, rng() * P.worldSize, W, 'herb', k));
  }
  for (let k = 0; k < P.Npred; k++) {
    const idx = P.N0 + k;
    const W = makeInitBrain();
    const c = makeCreature(rng() * P.worldSize, rng() * P.worldSize, W, 'carn', idx, 60 + rng() * 20);
    creatures.push(c);
  }

  baseW = creatures[0].W.map(m => m.clone());

  food = [];
  spawnFood();
}

function spawnFood() {
  while (food.length < P.foodCount) {
    food.push({ x: rng() * P.worldSize, y: rng() * P.worldSize });
  }
}

// ─── Neural network forward pass ───────────────────────────────────────────
function nnForward(W, inp) {
  let x = inp;
  for (let l = 0; l < W.length - 1; l++) x = matMulTanh(W[l], x);
  return matMulTanh(W[W.length - 1], x);
}

// ─── Mutation ──────────────────────────────────────────────────────────────
function mutateNetworkStructure(W) {
  const inSize = P.nInput;
  const outSize = P.nOutput;
  let L = W.length - 1; // number of hidden layers

  // maybe add hidden layer (insert before output)
  if (L < P.maxHiddenLayers && rng() < P.addLayerProb) {
    const nPrev = W[W.length - 2].rows; // rows of last hidden
    const nNew = Math.max(2, Math.min(P.maxNeurons, Math.round(nPrev * (0.7 + 0.6 * rng()))));
    const Wnew  = Matrix.randn(nNew, nPrev, 0.3);
    const Wout  = Matrix.randn(outSize, nNew, 0.3);
    W = [...W.slice(0, W.length - 1), Wnew, Wout];
    L++;
  }

  // maybe delete one hidden
  if (L > 1 && rng() < P.delLayerProb) {
    const kill = Math.floor(rng() * L); // which hidden to kill (0-indexed among hiddens)
    const prevCols = kill === 0 ? inSize : W[kill - 1].rows;
    if (kill === L - 1) {
      // killing last hidden → output attaches to prev
      W[W.length - 1] = Matrix.randn(outSize, prevCols, 0.3);
    } else {
      // killing middle hidden → next layer gets prev cols
      const nextRows = W[kill + 1].rows;
      W[kill + 1] = Matrix.randn(nextRows, prevCols, 0.3);
    }
    W = [...W.slice(0, kill), ...W.slice(kill + 1)];
    L--;
  }

  // maybe grow a neuron in a random hidden layer
  if (L >= 1 && rng() < P.addNeuronProb) {
    const which = Math.floor(rng() * L); // which hidden layer (0-indexed)
    if (W[which].rows < P.maxNeurons) {
      W[which].addRow(0.3);          // new hidden neuron
      W[which + 1].addCol(0.3);     // downstream layer gains a new input column
    }
  }

  return W;
}

// ─── Simulation step ───────────────────────────────────────────────────────
function simStep() {
  const n = creatures.length;
  if (n === 0) { spawnFood(); return; }

  // ── extract state into flat arrays ────────────────────────────────────
  const posX = new Float64Array(n);
  const posY = new Float64Array(n);
  const velX = new Float64Array(n);
  const velY = new Float64Array(n);
  const ang  = new Float64Array(n);
  const E    = new Float64Array(n);
  const fit  = new Float64Array(n);
  const age  = new Float64Array(n);
  const busy = new Float64Array(n);
  const fade = new Float64Array(n);
  const fadeInit = new Float64Array(n);
  const immature = new Float64Array(n);
  const isCarn = new Uint8Array(n);
  const lineage = new Int32Array(n);

  for (let k = 0; k < n; k++) {
    const c = creatures[k];
    posX[k] = c.posX; posY[k] = c.posY;
    velX[k] = c.velX; velY[k] = c.velY;
    ang[k]  = c.angle;
    E[k]    = c.energy;
    fit[k]  = c.fitness;
    age[k]  = c.age;
    busy[k] = c.busyTime;
    fade[k] = c.fadeTick;
    fadeInit[k] = c.fadeInit;
    immature[k] = c.immature;
    isCarn[k] = c.type === 'carn' ? 1 : 0;
    lineage[k] = c.lineage;
  }

  // ── age, fitness, timers ───────────────────────────────────────────────
  for (let k = 0; k < n; k++) {
    age[k]++;
    fit[k] += P.fitTick;
    if (busy[k] > 0) busy[k]--;
    if (fade[k] > 0) fade[k]--;
    if (immature[k] > 0) immature[k]--;
  }

  const WS = P.worldSize;
  const SR = P.senseRadius;

  // ── sensing ────────────────────────────────────────────────────────────
  const dx = new Float64Array(n);
  const dy = new Float64Array(n);
  const d  = new Float64Array(n).fill(SR);

  // herbivores → nearest food; flee if predator closer
  for (let h = 0; h < n; h++) {
    if (isCarn[h]) continue;
    // nearest food
    let bestFoodD2 = Infinity, bfx = 0, bfy = 0;
    for (let f = 0; f < food.length; f++) {
      const fx = torus(posX[h] - food[f].x, WS);
      const fy = torus(posY[h] - food[f].y, WS);
      const d2 = fx * fx + fy * fy;
      if (d2 < bestFoodD2) { bestFoodD2 = d2; bfx = fx; bfy = fy; }
    }
    // nearest predator
    let bestPredD2 = Infinity, bpx = 0, bpy = 0;
    for (let p = 0; p < n; p++) {
      if (!isCarn[p]) continue;
      const px = torus(posX[h] - posX[p], WS);
      const py = torus(posY[h] - posY[p], WS);
      const d2 = px * px + py * py;
      if (d2 < bestPredD2) { bestPredD2 = d2; bpx = px; bpy = py; }
    }
    if (bestPredD2 < bestFoodD2 && bestPredD2 < SR * SR) {
      // flee: vector AWAY from predator (from pred to herb)
      dx[h] = bpx; dy[h] = bpy;
      d[h]  = Math.sqrt(bestPredD2);
    } else if (bestFoodD2 < SR * SR) {
      // approach food (vector FROM herb TO food, negative = toward)
      // MATLAB uses dx = foodX - herbX (toward), then mode interprets fight as approach
      dx[h] = -bfx; dy[h] = -bfy;
      d[h]  = Math.sqrt(bestFoodD2);
    }
  }

  // carnivores → nearest weaker herbivore (different lineage)
  for (let cc = 0; cc < n; cc++) {
    if (!isCarn[cc]) continue;
    let bestD = Infinity, bx = 0, by = 0;
    for (let p = 0; p < n; p++) {
      if (isCarn[p]) continue;          // target herbs only
      if (fade[p] > 0) continue;        // already fading
      if (immature[p] > 0) continue;    // protected newborn
      if (E[p] >= E[cc]) continue;      // only weaker
      if (fit[p] >= fit[cc]) continue;  // only lower fitness
      if (lineage[p] === lineage[cc]) continue; // lineage veto
      const ddx = torus(posX[p] - posX[cc], WS);
      const ddy = torus(posY[p] - posY[cc], WS);
      const dist = hypot2(ddx, ddy);
      if (dist < bestD && dist < SR) { bestD = dist; bx = ddx; by = ddy; }
    }
    if (bestD < SR) { dx[cc] = bx; dy[cc] = by; d[cc] = bestD; }
  }

  // ── NN forward pass ────────────────────────────────────────────────────
  const act = [];
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) {
      act.push(new Float64Array(P.nOutput));
      continue;
    }
    const inp = new Float64Array([
      dx[k], dy[k],
      d[k] / SR,
      E[k] / P.reproEnergy,
      1,
    ]);
    act.push(nnForward(creatures[k].W, inp));
  }

  // ── interpret outputs ──────────────────────────────────────────────────
  const MT = P.modeThresh;
  for (let k = 0; k < n; k++) {
    const modeRaw  = act[k][0];
    const thrustIn = Math.max(act[k][1], 0);
    let mode = 0;
    if (modeRaw >  MT) mode =  1;
    if (modeRaw < -MT) mode = -1;

    const thetaTarget = Math.atan2(dy[k], dx[k]) + (mode === -1 ? Math.PI : 0);
    const dAng = wrapToPi(thetaTarget - ang[k]);
    ang[k] += 3 * P.dt * dAng;

    const thrust = thrustIn * (mode !== 0 ? 1 : 0);
    velX[k] += thrust * Math.cos(ang[k]);
    velY[k] += thrust * Math.sin(ang[k]);
  }

  // ── physics ────────────────────────────────────────────────────────────
  for (let k = 0; k < n; k++) {
    if (busy[k] > 0 || fade[k] > 0) { velX[k] = 0; velY[k] = 0; continue; }
    const spd = hypot2(velX[k], velY[k]);
    if (spd > P.maxSpeed) {
      const scl = P.maxSpeed / spd;
      velX[k] *= scl; velY[k] *= scl;
    }
    posX[k] = mod(posX[k] + velX[k] * P.dt, WS);
    posY[k] = mod(posY[k] + velY[k] * P.dt, WS);
    const spd2 = hypot2(velX[k], velY[k]);
    E[k] -= P.energyDecay + P.moveCost * spd2;
    if (isCarn[k]) E[k] -= P.predMoveCost * spd2;
  }

  // ── herbivores eat food ────────────────────────────────────────────────
  const r2 = P.bodyRadius * P.bodyRadius;
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

  // ── predator attacks ───────────────────────────────────────────────────
  const eatTicks = 10;
  for (let cc = 0; cc < n; cc++) {
    if (!isCarn[cc] || busy[cc] > 0 || fade[cc] > 0) continue;
    let bestD = Infinity, bestPrey = -1;
    for (let p = 0; p < n; p++) {
      if (p === cc) continue;
      if (fade[p] > 0) continue;
      if (immature[p] > 0) continue;
      const canEat = !isCarn[p] || (E[p] < E[cc] && fit[p] < fit[cc]);
      if (!canEat) continue;
      const ddx = torus(posX[p] - posX[cc], WS);
      const ddy = torus(posY[p] - posY[cc], WS);
      const dist = hypot2(ddx, ddy);
      if (dist < bestD) { bestD = dist; bestPrey = p; }
    }
    if (bestPrey >= 0 && bestD <= 2 * P.bodyRadius) {
      busy[cc] = eatTicks;
      fade[bestPrey] = eatTicks;
      fadeInit[bestPrey] = eatTicks;
      E[cc] += E[bestPrey] / 4 + P.predEnergyGain; // predEnergyGain is negative
      fit[cc] += P.fitFood;
    }
  }

  // ── death ──────────────────────────────────────────────────────────────
  const keep = new Uint8Array(n).fill(1);
  for (let k = 0; k < n; k++) {
    const oldAgeDeath = isCarn[k]
      ? age[k] >= P.maxAgeCarn
      : age[k] >= P.maxAgeHerb;
    const fadeDeath   = fade[k] === 0 && fadeInit[k] > 0;
    const energyDeath = E[k] <= 0;
    if (oldAgeDeath || fadeDeath || energyDeath) keep[k] = 0;
  }

  // ── write back survivors ───────────────────────────────────────────────
  const newCreatures = [];
  for (let k = 0; k < n; k++) {
    if (!keep[k]) continue;
    const c = creatures[k];
    c.posX = posX[k]; c.posY = posY[k];
    c.velX = velX[k]; c.velY = velY[k];
    c.angle = ang[k];
    c.energy = E[k];
    c.fitness = fit[k];
    c.age = age[k];
    c.act = act[k];
    c.busyTime = busy[k];
    c.fadeTick = fade[k];
    c.fadeInit = fadeInit[k];
    c.immature = immature[k];
    newCreatures.push(c);
  }
  creatures = newCreatures;

  // ── reproduction ───────────────────────────────────────────────────────
  const toReproduce = [];
  for (let k = 0; k < creatures.length; k++) {
    const c = creatures[k];
    if (c.fadeTick > 0 || c.busyTime > 0) continue;
    if (c.energy >= P.reproEnergy && c.act[2] > 0) toReproduce.push(k);
  }
  // process in reverse to keep indices valid when appending
  for (const k of toReproduce) {
    reproduce(k);
  }

  spawnFood();
}

// ─── Reproduce ────────────────────────────────────────────────────────────
function reproduce(i) {
  const par = creatures[i];
  const eHalf = par.energy / 2;
  par.energy = eHalf;
  par.fitness += P.fitRepro;
  par.immature = Math.max(par.immature, 5);

  // clone weights and mutate
  const childW = par.W.map(m => {
    const mc = m.clone();
    mc.addNoise(P.mutationSigma);
    return mc;
  });
  const mutatedW = mutateNetworkStructure(childW);

  const chi = makeCreature(
    mod(par.posX + Math.cos(par.angle) * P.bodyRadius * 3, P.worldSize),
    mod(par.posY + Math.sin(par.angle) * P.bodyRadius * 3, P.worldSize),
    mutatedW,
    par.type,
    par.lineage,
    eHalf,
  );
  chi.fitness = par.fitness / 2;
  chi.immature = Math.round(2 / P.dt);  // 10 ticks invulnerability
  chi.angle = rng() * 2 * Math.PI;
  if (par.type === 'herb') chi.color = geneColor(chi);

  creatures.push(chi);
}

// ─── Diversity score ───────────────────────────────────────────────────────
function diversityScore(cre) {
  const Lcre  = cre.W.length - 1;
  const Lbase = baseW.length - 1;
  const dL = Math.abs(Lcre - Lbase);
  const maxL = Math.max(Lcre, Lbase);
  let dN = 0;
  for (let k = 0; k < maxL; k++) {
    const nCre  = k < Lcre  ? cre.W[k].rows  : 0;
    const nBase = k < Lbase ? baseW[k].rows   : 0;
    dN += Math.abs(nCre - nBase);
  }
  let dF = 0;
  const commonLayers = Math.min(Lcre + 1, Lbase + 1);
  for (let k = 0; k < commonLayers; k++) {
    dF += cre.W[k].frobenius(baseW[k]);
  }
  return dL + 0.2 * dN + 0.002 * dF;
}

// ─── Polygon verts ────────────────────────────────────────────────────────
function polyVerts(cre) {
  const Wlast = cre.W[cre.W.length - 1];
  const sig   = Math.tanh(Wlast.sumSign() / Wlast.data.length);
  let ns = Math.max(3, Math.min(8, 3 + Math.round((sig + 1) * 2.5)));
  const sizeG = 0.8 + 0.6 / (1 + Math.exp(-Wlast.stdAll() * 3));
  let R = P.bodyRadius * sizeG;

  if (cre.fadeTick > 0 && cre.fadeInit > 0) {
    R *= cre.fadeTick / cre.fadeInit;
  } else if (cre.fadeInit > 0) {
    R = 0;
  }

  const vx = [], vy = [];
  for (let i = 0; i < ns; i++) {
    const th = (i / ns) * 2 * Math.PI + cre.angle;
    vx.push(cre.posX + R * Math.cos(th));
    vy.push(cre.posY + R * Math.sin(th));
  }
  return { vx, vy };
}

// ─── Renderer ─────────────────────────────────────────────────────────────
class Renderer {
  constructor() {
    this.worldCanvas  = document.getElementById('worldCanvas');
    this.wCtx         = this.worldCanvas.getContext('2d');
    this.popCanvas    = document.getElementById('popCanvas');
    this.pCtx         = this.popCanvas.getContext('2d');
    this.brainCanvases = [0, 1, 2].map(i => document.getElementById(`brain${i}`));
    this.brainCtxs     = this.brainCanvases.map(c => c.getContext('2d'));
    this.detailCanvas  = document.getElementById('brainDetail');
    this.dCtx          = this.detailCanvas.getContext('2d');

    this.scale = 1;
    this.dpr   = window.devicePixelRatio || 1;
    this._lastSelectedIdx = -1;
    this._lastBrainTick = -1;
    this.renderFrame = 0;

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    this.dpr = window.devicePixelRatio || 1;
    this._sizeCanvas(this.worldCanvas, this.wCtx);
    this._sizeCanvas(this.popCanvas,   this.pCtx);
    this.brainCanvases.forEach((c, i) => this._sizeCanvas(c, this.brainCtxs[i]));
    this._sizeCanvas(this.detailCanvas, this.dCtx);

    const ww = this.worldCanvas.clientWidth;
    const wh = this.worldCanvas.clientHeight;
    this.scale = Math.min(ww / P.worldSize, wh / P.worldSize);
    this.offsetX = (ww - P.worldSize * this.scale) / 2;
    this.offsetY = (wh - P.worldSize * this.scale) / 2;
  }

  _sizeCanvas(canvas, ctx) {
    const dpr = this.dpr;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width  = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  // world coords → canvas logical coords
  wx(x) { return this.offsetX + x * this.scale; }
  wy(y) { return this.offsetY + (P.worldSize - y) * this.scale; }

  // canvas logical → world coords (for touch)
  canvasToWorld(cx, cy) {
    return {
      x: (cx - this.offsetX) / this.scale,
      y: P.worldSize - (cy - this.offsetY) / this.scale,
    };
  }

  drawWorld(selectedIdx) {
    const ctx = this.wCtx;
    const cw  = this.worldCanvas.clientWidth;
    const ch  = this.worldCanvas.clientHeight;

    ctx.fillStyle = '#060608';
    ctx.fillRect(0, 0, cw, ch);

    // food (batched)
    ctx.fillStyle = '#38e838';
    ctx.beginPath();
    for (const f of food) {
      const fx = this.wx(f.x);
      const fy = this.wy(f.y);
      ctx.moveTo(fx + 2, fy);
      ctx.arc(fx, fy, 2, 0, Math.PI * 2);
    }
    ctx.fill();

    // creatures
    for (let k = 0; k < creatures.length; k++) {
      const c = creatures[k];
      const { vx, vy } = polyVerts(c);
      if (vx.length === 0) continue;
      const [r, g, b] = c.color;
      ctx.fillStyle = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
      ctx.beginPath();
      ctx.moveTo(this.wx(vx[0]), this.wy(vy[0]));
      for (let i = 1; i < vx.length; i++) ctx.lineTo(this.wx(vx[i]), this.wy(vy[i]));
      ctx.closePath();
      ctx.fill();
    }

    // halo for selected
    if (selectedIdx >= 0 && selectedIdx < creatures.length) {
      const c = creatures[selectedIdx];
      ctx.strokeStyle = '#ff0';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(this.wx(c.posX), this.wy(c.posY), P.bodyRadius * 2.5 * this.scale, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  drawPopGraph() {
    const ctx = this.pCtx;
    const cw = this.popCanvas.clientWidth;
    const ch = this.popCanvas.clientHeight;
    ctx.fillStyle = '#06060c';
    ctx.fillRect(0, 0, cw, ch);

    if (popCount < 2) return;

    let maxVal = 2;
    for (let i = 0; i < popCount; i++) {
      const idx = (popHead - popCount + i + POP_HISTORY_MAX) % POP_HISTORY_MAX;
      const tot = popHerb[idx] + popCarn[idx];
      if (tot > maxVal) maxVal = tot;
    }

    const xStep = cw / (popCount - 1);
    const yScale = (ch - 4) / (maxVal * 1.1);

    const drawLine = (arr, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < popCount; i++) {
        const idx = (popHead - popCount + i + POP_HISTORY_MAX) % POP_HISTORY_MAX;
        const x = i * xStep;
        const y = ch - arr[idx] * yScale - 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    drawLine(popHerb, '#3d3');
    drawLine(popCarn, '#d33');

    // total
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < popCount; i++) {
      const idx = (popHead - popCount + i + POP_HISTORY_MAX) % POP_HISTORY_MAX;
      const x = i * xStep;
      const y = ch - (popHerb[idx] + popCarn[idx]) * yScale - 2;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawBrain(ctx, cre, w, h) {
    ctx.clearRect(0, 0, w, h);
    if (!cre) return;
    const W = cre.W;
    const L = W.length - 1;  // hidden layer count

    // build layer node counts
    const layers = [];
    layers.push(W[0].cols);          // input count
    for (let l = 0; l < L; l++) layers.push(W[l].rows);
    layers.push(W[W.length - 1].rows); // output

    const nLayers = layers.length;
    const xSpacing = w / (nLayers + 1);

    // y positions for each layer
    const yPos = layers.map(count => {
      const spacing = h / (count + 1);
      return Array.from({ length: count }, (_, i) => spacing * (i + 1));
    });

    const xPos = layers.map((_, li) => xSpacing * (li + 1));

    // draw edges (grouped by color to reduce stroke calls)
    ctx.lineWidth = 0.8;
    const MAX_NODES_FOR_LINES = 20;
    for (let li = 0; li < nLayers - 1; li++) {
      const srcN = layers[li];
      const dstN = layers[li + 1];
      if (srcN > MAX_NODES_FOR_LINES || dstN > MAX_NODES_FOR_LINES) continue;
      // bucket edges by greyscale level
      const buckets = {};
      for (let s = 0; s < srcN; s++) {
        for (let dst = 0; dst < dstN; dst++) {
          const wv = W[li].get(dst, s);
          const col = weightColor(wv);
          if (!buckets[col]) buckets[col] = [];
          buckets[col].push([xPos[li], yPos[li][s], xPos[li + 1], yPos[li + 1][dst]]);
        }
      }
      for (const [col, segs] of Object.entries(buckets)) {
        ctx.strokeStyle = col;
        ctx.beginPath();
        for (const [x0, y0, x1, y1] of segs) {
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
        }
        ctx.stroke();
      }
    }

    // draw nodes
    const nodeR = Math.max(2, Math.min(5, w / 40));
    for (let li = 0; li < nLayers; li++) {
      for (let ni = 0; ni < layers[li]; ni++) {
        if (layers[li] > 50) {
          // too many nodes — just show count
          ctx.fillStyle = '#557';
          ctx.font = `${Math.max(8, nodeR * 2)}px sans-serif`;
          ctx.fillText(`${layers[li]}`, xPos[li] - nodeR, h / 2);
          break;
        }
        ctx.fillStyle = li === 0 ? '#66aaff' : li === nLayers - 1 ? '#ffaa44' : '#eeeeee';
        ctx.beginPath();
        ctx.arc(xPos[li], yPos[li][ni], nodeR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  drawBrainPanels(topIdxs, selectedIdx) {
    // top-3 by fitness
    for (let k = 0; k < 3; k++) {
      const bw = this.brainCanvases[k].clientWidth;
      const bh = this.brainCanvases[k].clientHeight;
      const ctx = this.brainCtxs[k];
      const cre = topIdxs[k] !== undefined ? creatures[topIdxs[k]] : null;
      this.drawBrain(ctx, cre, bw, bh);
    }

    // detail panel
    if (selectedIdx >= 0 && selectedIdx < creatures.length) {
      const bw = this.detailCanvas.clientWidth;
      const bh = this.detailCanvas.clientHeight;
      this.drawBrain(this.dCtx, creatures[selectedIdx], bw, bh);
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
    this._brainLabels = [0,1,2].map(i => document.getElementById(`brainLabel${i}`));

    document.getElementById('worldCanvas').addEventListener('touchstart', e => {
      e.preventDefault();
      const touch = e.changedTouches[0];
      const rect  = renderer.worldCanvas.getBoundingClientRect();
      const cx = touch.clientX - rect.left;
      const cy = touch.clientY - rect.top;
      const { x, y } = renderer.canvasToWorld(cx, cy);
      this._onTap(x, y);
    }, { passive: false });

    document.getElementById('worldCanvas').addEventListener('mousedown', e => {
      const rect = renderer.worldCanvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const { x, y } = renderer.canvasToWorld(cx, cy);
      this._onTap(x, y);
    });

    document.getElementById('closePanelBtn').addEventListener('click', () => {
      this.selectedIdx = -1;
      this._panel.classList.remove('open');
    });

    document.getElementById('pauseBtn').addEventListener('click', () => {
      this._paused = !this._paused;
      document.getElementById('pauseBtn').textContent = this._paused ? 'Resume' : 'Pause';
    });

    document.getElementById('saveBtn').addEventListener('click', () => this._save());
    document.getElementById('loadBtn').addEventListener('click', () => {
      document.getElementById('fileInput').click();
    });
    document.getElementById('fileInput').addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => this._load(ev.target.result);
      reader.readAsText(file);
      e.target.value = '';
    });

    // block body scroll on touch
    document.addEventListener('touchmove', e => e.preventDefault(), { passive: false });
  }

  get paused() { return this._paused; }

  _onTap(wx, wy) {
    const hitR = P.bodyRadius * 4;
    let bestD = hitR, bestK = -1;
    for (let k = 0; k < creatures.length; k++) {
      const c = creatures[k];
      const ddx = torus(wx - c.posX, P.worldSize);
      const ddy = torus(wy - c.posY, P.worldSize);
      const dist = hypot2(ddx, ddy);
      if (dist < bestD) { bestD = dist; bestK = k; }
    }
    this.selectedIdx = bestK;
    if (bestK >= 0) {
      this._panel.classList.add('open');
    } else {
      this._panel.classList.remove('open');
    }
  }

  updateHeader(nHerb, nCarn) {
    this._tickEl.textContent = `Tick ${tick}`;
    this._popEl.textContent  = `H:${nHerb} C:${nCarn}`;
  }

  updateBrainLabels(topIdxs) {
    for (let k = 0; k < 3; k++) {
      const idx = topIdxs[k];
      if (idx !== undefined && idx < creatures.length) {
        const c = creatures[idx];
        this._brainLabels[k].textContent = `Fit ${c.fitness.toFixed(0)} E${c.energy.toFixed(0)}`;
      } else {
        this._brainLabels[k].textContent = `Fit #${k+1}`;
      }
    }
  }

  updatePanelStats() {
    const idx = this.selectedIdx;
    if (idx < 0 || idx >= creatures.length) return;
    const c = creatures[idx];
    const layers = c.W.map(m => m.rows).join('→');
    this._panelTitle.textContent = `${c.type === 'carn' ? '🔴 Carnivore' : '🟢 Herbivore'}  #${idx}`;
    this._panelStats.textContent =
      `Energy: ${c.energy.toFixed(1)} · Fit: ${c.fitness.toFixed(0)} · Age: ${c.age} · Layers: [${c.W[0].cols}→${layers}]`;
  }

  _save() {
    const state = {
      tick, rngSeed,
      creatures: creatures.map(c => ({
        ...c,
        W: c.W.map(m => m.toJSON()),
        act: Array.from(c.act),
      })),
      food,
      popT:    Array.from(popT.slice(0, popCount)),
      popHerb: Array.from(popHerb.slice(0, popCount)),
      popCarn: Array.from(popCarn.slice(0, popCount)),
      popHead, popCount,
    };
    const blob = new Blob([JSON.stringify(state)], { type: 'application/json' });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = `bibites_${tick}.json`;
    a.click();
  }

  _load(text) {
    try {
      const s = JSON.parse(text);
      tick     = s.tick;
      rngSeed  = s.rngSeed;
      food     = s.food;
      creatures = s.creatures.map(sc => {
        const c = { ...sc, W: sc.W.map(m => Matrix.fromJSON(m)), act: new Float64Array(sc.act) };
        return c;
      });
      baseW = creatures[0]?.W.map(m => m.clone()) ?? baseW;
      if (s.popCount !== undefined) {
        popHead = s.popHead; popCount = s.popCount;
        for (let i = 0; i < s.popT.length; i++) { popT[i] = s.popT[i]; popHerb[i] = s.popHerb[i]; popCarn[i] = s.popCarn[i]; }
      }
    } catch (e) {
      alert('Failed to load state: ' + e.message);
    }
  }
}

// ─── Game Loop ────────────────────────────────────────────────────────────
class GameLoop {
  constructor(renderer, ui) {
    this.renderer = renderer;
    this.ui       = ui;
    this._rafId   = null;
    this._last    = 0;
    this._accumulator = 0;
    this._renderEvery = 10;  // update brain panels every N render frames
    this._brainFrame  = 0;
  }

  start() {
    this._last = performance.now();
    this._rafId = requestAnimationFrame(ts => this._loop(ts));
  }

  _loop(ts) {
    this._rafId = requestAnimationFrame(t => this._loop(t));

    const elapsed = Math.min(ts - this._last, 50);
    this._last = ts;

    if (!this.ui.paused) {
      // run 2 sim ticks per render frame (matches graphicsStride:2)
      simStep();
      simStep();
      tick += 2;

      // record population
      const nCarn = creatures.filter(c => c.type === 'carn').length;
      const nHerb = creatures.length - nCarn;
      popT[popHead]    = tick;
      popHerb[popHead] = nHerb;
      popCarn[popHead] = nCarn;
      popHead = (popHead + 1) % POP_HISTORY_MAX;
      if (popCount < POP_HISTORY_MAX) popCount++;

      this.ui.updateHeader(nHerb, nCarn);
    }

    // clamp selectedIdx if creature died
    if (this.ui.selectedIdx >= creatures.length) {
      this.ui.selectedIdx = -1;
      document.getElementById('creaturePanel').classList.remove('open');
    }

    // sort by fitness for top-3
    let topIdxs = [];
    if (creatures.length > 0) {
      const ranked = creatures
        .map((c, i) => ({ i, f: c.fitness }))
        .sort((a, b) => b.f - a.f);
      topIdxs = ranked.slice(0, 3).map(r => r.i);
    }

    // render
    this.renderer.drawWorld(this.ui.selectedIdx);
    this.renderer.drawPopGraph();

    this._brainFrame++;
    if (this._brainFrame % this._renderEvery === 0 || this.ui.selectedIdx !== this.renderer._lastSelectedIdx) {
      this.renderer.drawBrainPanels(topIdxs, this.ui.selectedIdx);
      this.ui.updateBrainLabels(topIdxs);
      this.ui.updatePanelStats();
      this.renderer._lastSelectedIdx = this.ui.selectedIdx;
    }

    if (creatures.length === 0) {
      document.getElementById('tickDisplay').textContent = 'Extinct! Reload to restart.';
      cancelAnimationFrame(this._rafId);
    }
  }
}

// ─── Bootstrap ────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  // iOS Safari vh fix
  const setVH = () => {
    document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`);
  };
  setVH();
  window.addEventListener('resize', setVH);

  initWorld(1);

  const renderer = new Renderer();
  const ui       = new UIController(renderer);
  const loop     = new GameLoop(renderer, ui);

  // set world canvas height = available width (square world) — must be after renderer is created
  const worldWrap = document.getElementById('worldWrap');
  const setLayout = () => {
    const isLandscape = window.innerWidth > window.innerHeight;
    if (!isLandscape) {
      // portrait: square world canvas
      const w = worldWrap.clientWidth || window.innerWidth;
      worldWrap.style.height = w + 'px';
    } else {
      // landscape: fill container height
      worldWrap.style.height = '';
    }
    renderer.resize();
  };
  setLayout();
  window.addEventListener('resize', setLayout);

  loop.start();
});
