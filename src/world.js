// ---------- World: terrain, construction layers, regions, power, pathfinding ----------
'use strict';

class World {
  constructor(seed) {
    this.seed = seed;
    this.W = MAP_W; this.H = MAP_H;
    const N = this.W * this.H;
    this.terrain = new Uint8Array(N);
    this.variant = new Uint8Array(N);
    this.fence = new Uint8Array(N);
    this.fenceHp = new Float32Array(N);
    this.fenceOrig = new Uint8Array(N);
    this.fencePowered = new Uint8Array(N);
    this.path = new Uint8Array(N);
    this.bld = new Int32Array(N); // building id or 0
    this.region = new Int32Array(N);
    this.covered = new Uint8Array(N); // power coverage
    this.lavaT = new Float32Array(N); // lava cooling timers
    this.buildings = new Map();
    this.nextBid = 1;
    this.regions = [];
    this.dirtyTerrain = true;
    this.pathVersion = 0;
    this.flowCache = new Map();
    this.power = { supply: 0, demand: 0, outage: 0, backupFuel: 0, ratio: 1 };
    this.volcano = null;
    // BFS scratch
    this._seen = new Int32Array(N);
    this._prev = new Int32Array(N);
    this._stamp = 1;
    this._queue = new Int32Array(N);
  }

  idx(x, y) { return y * this.W + x; }
  inb(x, y) { return x >= 0 && y >= 0 && x < this.W && y < this.H; }

  generate() {
    const noise = makeNoise(this.seed);
    const noise2 = makeNoise(this.seed + 99);
    const noise3 = makeNoise(this.seed + 7);
    const W = this.W, H = this.H;
    const r = mulberry32(this.seed + 5);
    const hmap = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const nx = (x / W) * 2 - 1, ny = (y / H) * 2 - 1;
      const d = Math.sqrt(nx * nx * 0.9 + ny * ny * 1.1);
      let h = noise(x * 0.06, y * 0.06, 5) * 0.75 + (1 - d * d) * 0.85 - 0.45;
      hmap[this.idx(x, y)] = h;
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = this.idx(x, y), h = hmap[i];
      let t;
      if (h < 0.05) t = T_DEEP;
      else if (h < 0.13) t = T_WATER;
      else if (h < 0.18) t = T_SAND;
      else {
        const m = noise2(x * 0.09, y * 0.09, 3);
        const f = noise3(x * 0.12, y * 0.12, 3);
        if (h > 0.52 && m > 0.55) t = T_ROCK;
        else if (m < 0.3 && h > 0.25 && h < 0.45) t = T_WATER; // inland lakes
        else if (f > 0.57) t = T_FOREST;
        else t = T_GRASS;
      }
      this.terrain[i] = t;
      this.variant[i] = Math.floor(r() * 251);
    }
    // Map border: always deep water
    for (let x = 0; x < W; x++) { this.terrain[this.idx(x, 0)] = T_DEEP; this.terrain[this.idx(x, H - 1)] = T_DEEP; }
    for (let y = 0; y < H; y++) { this.terrain[this.idx(0, y)] = T_DEEP; this.terrain[this.idx(W - 1, y)] = T_DEEP; }

    // Volcano: highest point that is rock or near high ground
    let best = -1, bx = 0, by = 0;
    for (let y = 4; y < H - 4; y++) for (let x = 4; x < W - 4; x++) {
      const h = hmap[this.idx(x, y)] + (this.terrain[this.idx(x, y)] === T_ROCK ? 0.2 : 0);
      if (h > best) { best = h; bx = x; by = y; }
    }
    // keep volcano away from the southern starting area
    if (by > H * 0.6) by = Math.floor(H * 0.35);
    this.volcano = { x: bx, y: by };
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      const x = bx + dx, y = by + dy;
      if (!this.inb(x, y)) continue;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d <= 1.5) this.terrain[this.idx(x, y)] = T_VOLCANO;
      else if (d <= 3.2) this.terrain[this.idx(x, y)] = T_ROCK;
    }
    this.removeTinyLakes();
    this.dirtyTerrain = true;
  }

  removeTinyLakes() {
    // Inland water bodies of 1-2 tiles look noisy: fill them in
    const W = this.W, H = this.H;
    const seen = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) {
      if (seen[i] || this.terrain[i] !== T_WATER) continue;
      const comp = []; const q = [i]; seen[i] = 1;
      let touchesDeep = false;
      while (q.length) {
        const c = q.pop(); comp.push(c);
        const x = c % W, y = (c / W) | 0;
        for (const [dx, dy] of DIRS4) {
          const nx = x + dx, ny = y + dy;
          if (!this.inb(nx, ny)) continue;
          const j = this.idx(nx, ny);
          if (this.terrain[j] === T_DEEP) touchesDeep = true;
          if (!seen[j] && this.terrain[j] === T_WATER) { seen[j] = 1; q.push(j); }
        }
      }
      if (!touchesDeep && comp.length < 4) for (const c of comp) this.terrain[c] = T_GRASS;
    }
  }

  // Find a good start location for the gate: southern grass near coast
  findStartSpot() {
    const W = this.W, H = this.H;
    let best = null, bestScore = -1e9;
    for (let y = Math.floor(H * 0.55); y < H - 4; y++) for (let x = 8; x < W - 8; x++) {
      let ok = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -4; dx <= 4; dx++) {
        const t = this.terrain[this.idx(x + dx, y + dy)];
        if (t === T_GRASS || t === T_FOREST || t === T_SAND) ok++;
      }
      const score = ok * 10 + y * 2 - Math.abs(x - W / 2) * 1.5;
      if (ok >= 60 && score > bestScore) { bestScore = score; best = { x, y }; }
    }
    return best || { x: Math.floor(W / 2), y: Math.floor(H * 0.75) };
  }

  terrainAt(x, y) { return this.inb(x, y) ? this.terrain[this.idx(x, y)] : T_DEEP; }
  isLand(t) { return t === T_SAND || t === T_GRASS || t === T_FOREST || t === T_BASALT; }

  // ---------- passability ----------
  dinoPass(x, y) {
    if (!this.inb(x, y)) return false;
    const i = this.idx(x, y);
    const t = this.terrain[i];
    if (!(t === T_SAND || t === T_GRASS || t === T_FOREST || t === T_BASALT)) return false;
    const f = this.fence[i];
    if (f === F_ELECTRIC || f === F_WALL) return false;
    if (this.bld[i]) return false;
    return true;
  }
  humanPass(x, y) { return this.dinoPass(x, y); }

  canBuildAt(x, y, allowForest = true) {
    if (!this.inb(x, y)) return false;
    const i = this.idx(x, y);
    const t = this.terrain[i];
    if (!(t === T_SAND || t === T_GRASS || t === T_BASALT || (allowForest && t === T_FOREST))) return false;
    if (this.fence[i] === F_ELECTRIC || this.fence[i] === F_WALL) return false;
    if (this.bld[i]) return false;
    return true;
  }

  // ---------- buildings ----------
  canPlaceBuilding(type, x, y) {
    const def = BUILDINGS[type];
    for (let dy = 0; dy < def.h; dy++) for (let dx = 0; dx < def.w; dx++) {
      if (!this.canBuildAt(x + dx, y + dy)) return false;
      if (this.path[this.idx(x + dx, y + dy)]) return false;
      if (this.fence[this.idx(x + dx, y + dy)] === F_BROKEN) return false;
    }
    return true;
  }

  addBuilding(type, x, y) {
    const def = BUILDINGS[type];
    const b = { id: this.nextBid++, type, x, y, w: def.w, h: def.h, hp: def.hp, maxHp: def.hp, powered: false, visitors: 0, inside: 0, offline: 0, built: 0 };
    this.buildings.set(b.id, b);
    for (let dy = 0; dy < def.h; dy++) for (let dx = 0; dx < def.w; dx++) {
      const i = this.idx(x + dx, y + dy);
      this.bld[i] = b.id;
      if (this.terrain[i] === T_FOREST) this.terrain[i] = T_GRASS;
    }
    this.invalidate();
    return b;
  }

  removeBuilding(b) {
    for (let dy = 0; dy < b.h; dy++) for (let dx = 0; dx < b.w; dx++) this.bld[this.idx(b.x + dx, b.y + dy)] = 0;
    this.buildings.delete(b.id);
    this.invalidate();
  }

  buildingAt(x, y) { return this.inb(x, y) ? this.buildings.get(this.bld[this.idx(x, y)]) : null; }

  // Path tiles adjacent to a building's footprint
  accessTiles(b) {
    const out = [];
    for (let dy = -1; dy <= b.h; dy++) for (let dx = -1; dx <= b.w; dx++) {
      const corner = (dx === -1 || dx === b.w) && (dy === -1 || dy === b.h);
      const inside = dx >= 0 && dx < b.w && dy >= 0 && dy < b.h;
      if (corner || inside) continue;
      const x = b.x + dx, y = b.y + dy;
      if (this.inb(x, y) && this.path[this.idx(x, y)]) out.push(this.idx(x, y));
    }
    return out;
  }

  // Passable tiles adjacent to a building (for staff spawning / dino feeding)
  adjacentPassable(b, passFn) {
    const out = [];
    for (let dy = -1; dy <= b.h; dy++) for (let dx = -1; dx <= b.w; dx++) {
      const inside = dx >= 0 && dx < b.w && dy >= 0 && dy < b.h;
      if (inside) continue;
      const x = b.x + dx, y = b.y + dy;
      if (passFn.call(this, x, y)) out.push([x, y]);
    }
    return out;
  }

  invalidate() {
    this.dirtyTerrain = true;
    this.pathVersion++;
    this.flowCache.clear();
    this.regionsDirty = true;
    this.powerDirty = true;
  }

  // ---------- regions (paddocks) ----------
  computeRegions() {
    const W = this.W, H = this.H, N = W * H;
    this.region.fill(-1);
    const regions = [];
    const q = this._queue;
    for (let s = 0; s < N; s++) {
      if (this.region[s] !== -1) continue;
      const sx = s % W, sy = (s / W) | 0;
      if (!this.dinoPass(sx, sy)) continue;
      const id = regions.length;
      const reg = { id, size: 0, forest: 0, water: 0, path: 0, feeders: { herb: [], carn: [] }, tiles: null, cx: 0, cy: 0, minX: sx, maxX: sx, minY: sy, maxY: sy };
      let head = 0, tail = 0;
      q[tail++] = s; this.region[s] = id;
      const tiles = [];
      const adjB = new Set();
      while (head < tail) {
        const c = q[head++];
        tiles.push(c);
        const x = c % W, y = (c / W) | 0;
        reg.size++; reg.cx += x; reg.cy += y;
        if (x < reg.minX) reg.minX = x; if (x > reg.maxX) reg.maxX = x;
        if (y < reg.minY) reg.minY = y; if (y > reg.maxY) reg.maxY = y;
        if (this.terrain[c] === T_FOREST) reg.forest++;
        if (this.path[c]) reg.path++;
        for (const [dx, dy] of DIRS4) {
          const nx = x + dx, ny = y + dy;
          if (!this.inb(nx, ny)) continue;
          const j = this.idx(nx, ny);
          const t = this.terrain[j];
          if (t === T_WATER || t === T_DEEP) reg.water++;
          if (this.bld[j]) adjB.add(this.bld[j]);
          if (this.region[j] === -1 && this.dinoPass(nx, ny)) { this.region[j] = id; q[tail++] = j; }
        }
      }
      reg.cx /= reg.size; reg.cy /= reg.size;
      reg.tiles = tiles;
      for (const bid of adjB) {
        const b = this.buildings.get(bid);
        if (!b) continue;
        const d = BUILDINGS[b.type];
        if (d.feeds) reg.feeders[d.feeds].push(b);
      }
      reg.public = reg.path > 0;
      regions.push(reg);
    }
    this.regions = regions;
    this.regionsDirty = false;
  }

  regionAt(x, y) {
    if (!this.inb(x, y)) return null;
    const r = this.region[this.idx(x, y)];
    return r >= 0 ? this.regions[r] : null;
  }

  // ---------- power ----------
  computePower(game) {
    const W = this.W, H = this.H;
    this.covered.fill(0);
    this.fencePowered.fill(0);
    let supply = 0;
    const outage = this.power.outage > 0;
    const sources = [], pylons = [];
    for (const b of this.buildings.values()) {
      if (b.type === 'power' && !outage && b.offline <= 0 && b.hp > 0) { sources.push(b); supply += BUILDINGS.power.supply * clamp(b.hp / b.maxHp + 0.3, 0.3, 1); }
      if (b.type === 'backup' && outage && this.power.backupFuel > 0 && this.power.backupDelay <= 0) { sources.push(b); supply += BUILDINGS.backup.backup; }
      if (b.type === 'pylon') pylons.push(b);
    }
    const coverDisc = (cx, cy, rad) => {
      const r2 = rad * rad;
      for (let y = Math.max(0, Math.floor(cy - rad)); y <= Math.min(H - 1, Math.ceil(cy + rad)); y++)
        for (let x = Math.max(0, Math.floor(cx - rad)); x <= Math.min(W - 1, Math.ceil(cx + rad)); x++)
          if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) this.covered[this.idx(x, y)] = 1;
    };
    for (const s of sources) coverDisc(s.x + s.w / 2 - 0.5, s.y + s.h / 2 - 0.5, BUILDINGS[s.type].range);
    // Pylons chain from covered area
    const active = new Set();
    let changed = true;
    while (changed) {
      changed = false;
      for (const p of pylons) {
        if (active.has(p.id) || p.hp <= 0) continue;
        if (this.covered[this.idx(p.x, p.y)]) { active.add(p.id); coverDisc(p.x, p.y, BUILDINGS.pylon.range); changed = true; }
      }
    }
    // Demand: fences first (safety), then buildings
    const fenceTiles = [];
    let fenceDemand = 0;
    for (let i = 0; i < W * H; i++) if (this.fence[i] === F_ELECTRIC && this.covered[i]) { fenceTiles.push(i); fenceDemand += FENCE_DEF[F_ELECTRIC].power; }
    let bDemand = 0;
    const consumers = [];
    for (const b of this.buildings.values()) {
      b.powered = false;
      const need = BUILDINGS[b.type].power;
      if (!need) { b.powered = true; continue; }
      if (this.covered[this.idx(b.x, b.y)] || this.covered[this.idx(b.x + b.w - 1, b.y + b.h - 1)]) { consumers.push(b); bDemand += need; }
    }
    const demand = fenceDemand + bDemand;
    let left = supply;
    // Fences
    if (left >= fenceDemand) {
      for (const i of fenceTiles) this.fencePowered[i] = 1;
      left -= fenceDemand;
    } else if (fenceTiles.length) {
      // partial: power a deterministic subset (cluster sweep so gaps appear in runs)
      const n = Math.floor(left / FENCE_DEF[F_ELECTRIC].power);
      for (let k = 0; k < n; k++) this.fencePowered[fenceTiles[k]] = 1;
      left = 0;
    }
    consumers.sort((a, b) => (BUILDINGS[a.type].cat === 'staff' ? -1 : 0) - (BUILDINGS[b.type].cat === 'staff' ? -1 : 0));
    for (const b of consumers) {
      const need = BUILDINGS[b.type].power;
      if (left >= need) { b.powered = true; left -= need; }
    }
    this.power.supply = supply; this.power.demand = demand;
    this.power.ratio = demand > 0 ? Math.min(1, supply / demand) : 1;
    this.powerDirty = false;
    this.dirtyTerrain = true;
  }

  // ---------- guest flow fields over path tiles ----------
  // returns Int16Array distances (-1 = unreachable) toward a set of target path tiles
  flowField(key, targetsFn) {
    let f = this.flowCache.get(key);
    if (f) return f;
    const N = this.W * this.H;
    f = new Int16Array(N).fill(-1);
    const q = this._queue;
    let head = 0, tail = 0;
    for (const t of targetsFn()) { if (f[t] === -1) { f[t] = 0; q[tail++] = t; } }
    while (head < tail) {
      const c = q[head++];
      const x = c % this.W, y = (c / this.W) | 0;
      for (const [dx, dy] of DIRS4) {
        const nx = x + dx, ny = y + dy;
        if (!this.inb(nx, ny)) continue;
        const j = this.idx(nx, ny);
        if (f[j] !== -1 || !this.path[j]) continue;
        f[j] = f[c] + 1; q[tail++] = j;
      }
    }
    this.flowCache.set(key, f);
    return f;
  }

  // ---------- generic BFS for free-roaming agents ----------
  // Returns array of [x,y] from (excl) start to goal (incl), or null
  bfs(sx, sy, passFn, goalFn, maxNodes = 4000) {
    const W = this.W;
    const stamp = ++this._stamp;
    const seen = this._seen, prev = this._prev, q = this._queue;
    const s = this.idx(sx, sy);
    let head = 0, tail = 0;
    q[tail++] = s; seen[s] = stamp; prev[s] = -1;
    let found = -1;
    while (head < tail && head < maxNodes) {
      const c = q[head++];
      const x = c % W, y = (c / W) | 0;
      if (c !== s && goalFn(x, y)) { found = c; break; }
      for (let k = 0; k < 4; k++) {
        const nx = x + DIRS4[k][0], ny = y + DIRS4[k][1];
        if (!this.inb(nx, ny)) continue;
        const j = this.idx(nx, ny);
        if (seen[j] === stamp) continue;
        seen[j] = stamp;
        if (!passFn.call(this, nx, ny) && !goalFn(nx, ny)) continue;
        prev[j] = c; q[tail++] = j;
      }
    }
    if (found < 0) return null;
    const out = [];
    let c = found;
    while (c !== s && c >= 0) { out.push([c % W, (c / W) | 0]); c = prev[c]; }
    out.reverse();
    return out;
  }
}
