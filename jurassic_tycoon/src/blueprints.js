// ---------- Less busywork: paddock blueprints and one-click paddock upkeep ----------
'use strict';

const BlueprintMixin = {
  // Rock or water that closes a paddock where there is no fence: a copy built elsewhere needs fence there
  naturalWalls(reg) {
    const w = this.world, out = new Set(), inReg = new Set(reg.tiles);
    for (const t of reg.tiles) {
      const x = t % w.W, y = (t / w.W) | 0;
      for (const [dx, dy] of DIRS8) {
        const nx = x + dx, ny = y + dy;
        if (!w.inb(nx, ny)) continue;
        const j = w.idx(nx, ny);
        if (inReg.has(j) || w.fence[j] || w.bld[j]) continue;
        if (!w.dinoPass(nx, ny)) out.add(j);
      }
    }
    if (!out.size) return [];
    // keep only the outer walls: a pond inside the paddock is scenery, not a wall
    const x0 = reg.minX - 2, y0 = reg.minY - 2, x1 = reg.maxX + 2, y1 = reg.maxY + 2;
    const seen = new Set(), q = [];
    for (let x = x0; x <= x1; x++) for (const y of [y0, y1]) if (w.inb(x, y)) q.push(w.idx(x, y));
    for (let y = y0; y <= y1; y++) for (const x of [x0, x1]) if (w.inb(x, y)) q.push(w.idx(x, y));
    while (q.length) {
      const j = q.pop();
      if (seen.has(j) || inReg.has(j) || w.fence[j]) continue;
      seen.add(j);
      const x = j % w.W, y = (j / w.W) | 0;
      for (const [dx, dy] of DIRS4) { const nx = x + dx, ny = y + dy; if (nx >= x0 && nx <= x1 && ny >= y0 && ny <= y1 && w.inb(nx, ny)) q.push(w.idx(nx, ny)); }
    }
    return [...out].filter((j) => seen.has(j));
  },

  // Fence tiles that enclose a region (8-neighbourhood so corner posts count)
  paddockFences(reg) {
    const w = this.world, out = new Set();
    for (const t of reg.tiles) {
      const x = t % w.W, y = (t / w.W) | 0;
      for (const [dx, dy] of DIRS8) {
        const nx = x + dx, ny = y + dy;
        if (!w.inb(nx, ny)) continue;
        const j = w.idx(nx, ny);
        if (w.fence[j]) out.add(j);
      }
    }
    return [...out];
  },

  copyPaddock(reg) {
    const w = this.world;
    const fences = this.paddockFences(reg);
    if (!fences.length) return null;
    const walls = this.naturalWalls(reg);
    let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
    for (const j of fences.concat(walls)) { const x = j % w.W, y = (j / w.W) | 0; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    const bp = { w: x1 - x0 + 1, h: y1 - y0 + 1, fences: [], feeders: [], trees: [] };
    for (const j of fences) {
      const f = w.fence[j];
      bp.fences.push([j % w.W - x0, ((j / w.W) | 0) - y0, f === F_BROKEN ? (w.fenceOrig[j] || F_ELECTRIC) : f === F_GATE ? F_ELECTRIC : f]);
    }
    for (const j of walls) bp.fences.push([j % w.W - x0, ((j / w.W) | 0) - y0, F_ELECTRIC]);
    for (const t of reg.tiles) if (w.terrain[t] === T_FOREST) bp.trees.push([t % w.W - x0, ((t / w.W) | 0) - y0]);
    for (const b of reg.feeders.herb.concat(reg.feeders.carn)) bp.feeders.push([b.x - x0, b.y - y0, b.type]);
    return bp;
  },

  // What a stamp would build at (x, y), and what it costs
  stampPlan(bp, x, y) {
    const w = this.world;
    const plan = { fences: [], feeders: [], trees: [], cost: 0, blocked: 0 };
    for (const [dx, dy, f] of bp.fences) {
      const tx = x + dx, ty = y + dy;
      if (!w.inb(tx, ty)) { plan.blocked++; continue; }
      const i = w.idx(tx, ty), t = w.terrain[i];
      if (w.fence[i] === f) continue;
      if (!(t === T_SAND || t === T_GRASS || t === T_FOREST || t === T_BASALT) || w.bld[i] || w.path[i]) { plan.blocked++; continue; }
      plan.fences.push([tx, ty, f]); plan.cost += FENCE_DEF[f].cost + (t === T_FOREST ? 40 : 0);
    }
    for (const [dx, dy, type] of bp.feeders) {
      if (w.canPlaceBuilding(type, x + dx, y + dy)) { plan.feeders.push([x + dx, y + dy, type]); plan.cost += BUILDINGS[type].cost; }
    }
    for (const [dx, dy] of bp.trees) {
      const tx = x + dx, ty = y + dy;
      if (!w.inb(tx, ty)) continue;
      const i = w.idx(tx, ty);
      if (w.terrain[i] === T_GRASS && !w.path[i] && !w.bld[i] && !w.fence[i]) { plan.trees.push([tx, ty]); plan.cost += TOOL_INFO.trees.cost; }
    }
    return plan;
  },

  stamp(bp, x, y) {
    const plan = this.stampPlan(bp, x, y);
    if (!plan.fences.length && !plan.feeders.length) return { n: 0, plan };
    if (!this.canAfford(plan.cost)) return { n: 0, plan, money: true };
    let n = 0;
    this.startBatch();
    for (const [tx, ty, f] of plan.fences) if (this.placeFence(tx, ty, f)) n++;
    for (const [tx, ty] of plan.trees) this.plantTree(tx, ty);
    for (const [tx, ty, type] of plan.feeders) if (this.placeBuilding(type, tx, ty)) n++;
    this.endBatch('paddock blueprint');
    this.world.computeRegions(); this.world.computePower(this);
    return { n, plan };
  },

  // Plant trees until about `target` of the paddock is covered
  plantCover(reg, target = 0.3) {
    const w = this.world;
    const need = Math.ceil(reg.size * target) - reg.forest;
    if (need <= 0) return 0;
    const cands = reg.tiles.filter((t) => w.terrain[t] === T_GRASS && !w.path[t] && !w.bld[t] && !w.fence[t]);
    // grow clumps around the existing cover first: looks natural and keeps open ground for walking
    const score = (t) => { const x = t % w.W, y = (t / w.W) | 0; let s = 0; for (const [dx, dy] of DIRS8) { const j = w.idx(x + dx, y + dy); if (w.terrain[j] === T_FOREST) s += 2; if (w.fence[j]) s += 1; } return s + rand(); };
    cands.sort((a, b) => score(b) - score(a));
    let n = 0;
    this.startBatch();
    for (const t of cands.slice(0, need)) if (this.plantTree(t % w.W, (t / w.W) | 0)) n++;
    this.endBatch('tree planting');
    w.computeRegions();
    return n;
  },

  // Put a feeder of the given diet on a free tile inside the paddock (near its centre)
  addFeeder(reg, diet, inBatch = false) {
    const w = this.world, type = diet === 'carn' ? 'feeder_c' : 'feeder_h';
    const tiles = reg.tiles.slice().sort((a, b) => dist2(a % w.W, (a / w.W) | 0, reg.cx, reg.cy) - dist2(b % w.W, (b / w.W) | 0, reg.cx, reg.cy));
    for (const t of tiles) {
      const x = t % w.W, y = (t / w.W) | 0;
      if (this.dinos.some((d) => d.tx === x && d.ty === y)) continue;
      if (!w.canPlaceBuilding(type, x, y)) continue;
      if (!inBatch) this.startBatch();
      const b = this.placeBuilding(type, x, y);
      if (!inBatch) this.endBatch(BUILDINGS[type].name);
      if (b) { w.computeRegions(); return b; }
    }
    return null;
  },

  // Repair every damaged or breached fence: of one paddock, or of the whole park
  repairFences(reg = null) {
    const w = this.world;
    const tiles = reg ? this.paddockFences(reg) : [...w.fence.keys()].filter((i) => w.fence[i]);
    let n = 0, cost = 0, blocked = 0;
    for (const i of tiles) {
      const x = i % w.W, y = (i / w.W) | 0;
      if (!this.needsRepair(x, y)) continue;
      const f = w.fence[i] === F_BROKEN ? (w.fenceOrig[i] || F_ELECTRIC) : w.fence[i];
      const c = Math.round(FENCE_DEF[f].cost * (w.fence[i] === F_BROKEN ? 1.5 : 0.5));
      if (!this.canAfford(c)) break;
      if (w.fence[i] === F_BROKEN && !this.clearGap(x, y)) { blocked++; continue; }
      this.spend(c, 'repairs'); cost += c; n++;
      const wasBroken = w.fence[i] === F_BROKEN;
      w.fence[i] = f; w.fenceHp[i] = FENCE_DEF[f].hp;
      if (wasBroken) w.invalidate(x, y); else w.markDirty(x, y);
      this.sparks(x + 0.5, y + 0.5, '#f8e080', 4);
    }
    return { n, cost, blocked };
  },
  repairAllCost() {
    const w = this.world;
    let c = 0, n = 0;
    for (let i = 0; i < w.fence.length; i++) {
      if (!w.fence[i]) continue;
      const x = i % w.W, y = (i / w.W) | 0;
      if (!this.needsRepair(x, y)) continue;
      const f = w.fence[i] === F_BROKEN ? (w.fenceOrig[i] || F_ELECTRIC) : w.fence[i];
      c += Math.round(FENCE_DEF[f].cost * (w.fence[i] === F_BROKEN ? 1.5 : 0.5)); n++;
    }
    return { cost: c, n };
  },
};
for (const [k, v] of Object.entries(BlueprintMixin)) Game.prototype[k] = v;
