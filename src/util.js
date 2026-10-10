// ---------- Utilities: RNG, noise, math helpers ----------
'use strict';

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let RNG = mulberry32(Date.now() & 0xffffffff);
const rand = () => RNG();
const randi = (a, b) => a + Math.floor(RNG() * (b - a + 1));
const randf = (a, b) => a + RNG() * (b - a);
const chance = (p) => RNG() < p;
const pick = (arr) => arr[Math.floor(RNG() * arr.length)];
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const dist2 = (ax, ay, bx, by) => (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
const dist = (ax, ay, bx, by) => Math.sqrt(dist2(ax, ay, bx, by));

function weightedPick(items, weightFn) {
  let total = 0;
  for (const it of items) total += Math.max(0, weightFn(it));
  if (total <= 0) return null;
  let r = RNG() * total;
  for (const it of items) {
    r -= Math.max(0, weightFn(it));
    if (r <= 0) return it;
  }
  return items[items.length - 1];
}

// Value noise (seeded) for terrain generation
function makeNoise(seed) {
  const r = mulberry32(seed);
  const SIZE = 256;
  const perm = new Uint8Array(SIZE * 2);
  const vals = new Float32Array(SIZE);
  for (let i = 0; i < SIZE; i++) { perm[i] = i; vals[i] = r(); }
  for (let i = SIZE - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (let i = 0; i < SIZE; i++) perm[i + SIZE] = perm[i];
  const smooth = (t) => t * t * (3 - 2 * t);
  function n2(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = x - xi, yf = y - yi;
    const X = xi & 255, Y = yi & 255;
    const v00 = vals[perm[X + perm[Y]]];
    const v10 = vals[perm[X + 1 + perm[Y]]];
    const v01 = vals[perm[X + perm[Y + 1]]];
    const v11 = vals[perm[X + 1 + perm[Y + 1]]];
    const u = smooth(xf), v = smooth(yf);
    return lerp(lerp(v00, v10, u), lerp(v01, v11, u), v);
  }
  return function fbm(x, y, oct = 4) {
    let amp = 1, freq = 1, sum = 0, norm = 0;
    for (let o = 0; o < oct; o++) {
      sum += n2(x * freq, y * freq) * amp;
      norm += amp; amp *= 0.5; freq *= 2;
    }
    return sum / norm;
  };
}

function fmtMoney(v) {
  const neg = v < 0; v = Math.abs(Math.round(v));
  let s;
  if (v >= 1e6) s = (v / 1e6).toFixed(v >= 1e7 ? 1 : 2) + 'M';
  else if (v >= 1e4) s = (v / 1e3).toFixed(0) + 'k';
  else s = v.toLocaleString('en-US');
  return (neg ? '-$' : '$') + s;
}
function fmtMoneyFull(v) {
  const neg = v < 0; v = Math.abs(Math.round(v));
  return (neg ? '-$' : '$') + v.toLocaleString('en-US');
}

const DIRS4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const DIRS8 = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
