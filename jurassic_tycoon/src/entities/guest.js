'use strict';

// ---------------- Guests ----------------
class Guest {
  constructor(game, tile) {
    this.id = _eid++;
    this.kind = 'guest';
    this.game = game;
    const w = game.world;
    this.cx = tile % w.W; this.cy = (tile / w.W) | 0;
    this.x = this.cx + 0.5 + randf(-0.25, 0.25); this.y = this.cy + 0.5 + randf(-0.25, 0.25);
    this.nx = this.cx; this.ny = this.cy; this.prev = -1;
    this.ox = randf(-0.28, 0.28); this.oy = randf(-0.28, 0.28);
    this.kid = chance(0.22);
    this.sprite = pick(this.kid ? PEOPLE.kids : PEOPLE.guests);
    this.state = 'walk';
    this.hunger = randf(0, 40); this.toilet = randf(0, 30); this.shopUrge = randf(0, 60);
    this.happy = 60; this.fear = 0;
    this.stay = randf(10, 20); // hours in park
    this.goal = null; this.insideT = 0; this.hidden = false; this.dead = false;
    this.anim = rand(); this.facing = 1;
    this.seenSpecies = new Set();
    this.viewT = rand();
    this.speedMul = randf(0.85, 1.15);
    this.spent = 0;
    this.thought = null;
  }

  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }

  update(dt) {
    const g = this.game, w = g.world;
    this.anim += dt;
    const hrs = dt / g.secPerHour;
    if (this.state === 'ride') { if (this.inJeep) { this.x = this.inJeep.x; this.y = this.inJeep.y; } return; }
    if ((this.state === 'queue' || this.state === 'inside') && (this.scareT = (this.scareT || 0) - dt) <= 0) {
      // a shop counter or a ride queue is no hiding place: a predator close by flushes everyone out
      this.scareT = 1;
      if (g.threatNear(this.cx, this.cy, 5)) { this.leaveCover(); this.state = 'flee'; this.goal = null; this.fear = 100; this.thought = 'RUN!'; this.thoughtT = 2; return; }
    }
    if (this.state === 'queue') {
      this.insideT -= dt;
      if (this.insideT <= 0) { this.happy -= 8; this.hidden = false; this.state = 'walk'; }
      return;
    }
    if (this.state === 'inside' || this.state === 'shelter') {
      this.insideT -= dt;
      if (this.state === 'shelter') {
        if (!g.alarm && g.threatNear(this.cx, this.cy, 12) == null && this.insideT <= 0) this.exitBuilding();
      } else if (this.insideT <= 0) this.exitBuilding();
      return;
    }
    if (this.state === 'frozen') { this.frozenT -= dt; if (this.frozenT <= 0) this.state = 'flee'; return; }
    this.stay -= hrs;
    this.hunger += hrs * 5; this.toilet += hrs * 6; this.shopUrge += hrs * 3;

    // Threat check
    this.viewT -= dt;
    if (this.viewT <= 0) {
      this.viewT = 0.8;
      const threat = g.threatNear(this.x, this.y, 7);
      if ((threat || g.alarm || (g.sirenAlarm && g.inSirenRange(this.x, this.y))) && this.state !== 'flee' && this.state !== 'frozen') {
        this.goal = null; this.fear = 100;
        // a predator in plain sight: some people freeze before they run (an alarm gives warning, so nobody does)
        if (threat && chance(0.3)) { this.state = 'frozen'; this.frozenT = randf(1, 2.5); this.thought = 'AAAH!'; this.thoughtT = 2; g.soundAt('scream', this.x, this.y, 0.6); }
        else this.state = 'flee';
      }
      if (this.state !== 'flee') this.lookAtDinos();
      if (this.hunger > 85 || this.toilet > 85) this.happy -= 1.5;
      if (g.crowd) {
        // people per walkable tile nearby: a wider path or plaza really does relieve the crush
        let c = 0, room = 0;
        const w = g.world;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const x = this.cx + dx, y = this.cy + dy; if (!w.inb(x, y)) continue; const i = w.idx(x, y); c += g.crowd[i]; if (w.path[i]) room++; }
        const limit = 6 * Math.max(3, room);
        if (c > limit) {
          this.happy -= (c - limit) * 0.04;
          this.crowded = true;
          if (chance(0.03)) { this.thought = 'CROWDED'; this.thoughtT = 1.5; }
        } else this.crowded = false;
      }
      if (g.hour >= 20 || g.hour < 6) { if (!g.litNear(this.x, this.y)) this.happy -= 0.3; }
      this.happy = clamp(this.happy, 0, 100);
      if (this.happy < 12 && this.state === 'walk') this.state = 'leave';
      if (this.stay <= 0 && this.state === 'walk') this.state = 'leave';
    }
    if (this.thoughtT > 0) this.thoughtT -= dt; else this.thought = null;

    // Movement between tiles
    // a park is a long walk: guests stroll briskly so far-off attractions are reachable in a visit
    let speed = (this.state === 'flee' ? 2.1 : 1.3) * this.speedMul;
    // a panicking crowd jams: narrow, packed paths turn an evacuation into a crush
    if (this.state === 'flee' && g.crowd) { const n = g.crowd[g.world.idx(this.cx, this.cy)] || 0; if (n > 3) speed *= clamp(1 - (n - 3) * 0.08, 0.35, 1); }
    const tx = this.nx + 0.5 + this.ox, ty = this.ny + 0.5 + this.oy;
    const dx = tx - this.x, dy = ty - this.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    const s = speed * dt;
    if (d > s) {
      this.x += (dx / d) * s; this.y += (dy / d) * s;
      if (Math.abs(dx) > 0.01) this.facing = dx > 0 ? 1 : -1;
      this.moving = true;
      return;
    }
    this.x = tx; this.y = ty; this.moving = false;
    const here = w.idx(this.nx, this.ny);
    this.prev = w.idx(this.cx, this.cy);
    this.cx = this.nx; this.cy = this.ny;
    if (!w.path[here]) {
      // path removed beneath us: walk to nearest path or vanish
      this.game.removeGuest(this, false); return;
    }
    this.chooseNext(here);
  }

  lookAtDinos() {
    const g = this.game;
    let view = 6;
    const vp = g.nearBuilding(this.cx, this.cy, 'viewing', 1);
    if (vp) view = BUILDINGS.viewing.view;
    let joy = 0;
    for (const d of g.creatures()) {
      if (d.carried) continue;
      const dd = dist2(this.x, this.y, d.x, d.y);
      if (dd > view * view) continue;
      if (d.loose) continue;
      if (d.isPtera && dd > 64) continue;
      const fresh = !this.seenSpecies.has(d.species);
      if (fresh) this.seenSpecies.add(d.species);
      joy += d.sp.appeal * (fresh ? 1.2 : 0.12) * (d.sedatedT > 0 ? 0.3 : 1);
    }
    if (joy > 0) {
      this.happy = Math.min(100, this.happy + Math.min(joy, 12));
      if (joy > 3 && chance(0.15)) { this.thought = pick(['WOW!', 'COOL!', 'OOH!', '!!']); this.thoughtT = 1.6; }
    } else {
      this.happy -= 0.25;
    }
  }

  pickGoal() {
    const g = this.game;
    let type = null;
    if (this.state === 'leave') return 'gate';
    if (this.hunger > 60) type = 'restaurant';
    else if (this.toilet > 60) type = 'restroom';
    else if ((g.hour >= 19 || g.hour < 6) && this.stay > 4) type = 'hotel';
    else if (this.shopUrge > 70) type = 'shop';
    else if (chance(0.45)) type = pick(['tour', 'tour', 'gyro', 'gyro', 'arena', 'feedshow', 'viewing', 'viewing', 'visitor', 'shop', 'restaurant']);
    if (!type) return null;
    const cands = g.buildingsOfType(type).filter((b) => b.powered || !BUILDINGS[b.type].power);
    if (!cands.length) { if (type === 'restroom' || type === 'restaurant') this.happy -= 2; return null; }
    // a pressing need goes to the nearest one you can actually walk to; anything else can be anywhere
    if (type === 'restroom' || type === 'restaurant') {
      const here = g.world.idx(this.cx, this.cy);
      let best = null, bd = Infinity;
      for (const b of cands) { const d = g.buildingField(b)[here]; if (d >= 0 && d < bd) { bd = d; best = b; } }
      if (!best) { this.happy -= 2; return null; }
      return best;
    }
    return pick(cands);
  }

  chooseNext(here) {
    const g = this.game, w = g.world;
    let field = null;
    if (this.state === 'flee') {
      // people trip when they run for their lives
      if (chance(0.03)) { this.state = 'frozen'; this.frozenT = randf(0.7, 1.4); this.thought = 'OOF!'; this.thoughtT = 1; return; }
      field = g.shelterField();
      if (field[here] === -1 || !g.anyShelterSpace()) {
        if (g.gateLocked) field = null; // nowhere to run: panic in place
        else { field = g.gateField(); this.state = 'flee'; this.fleeGate = true; }
      }
    } else if (this.state === 'leave') {
      field = g.gateLocked ? null : g.gateField();
    } else {
      if (!this.goal || (this.goal !== 'gate' && !w.buildings.has(this.goal.id))) { const urgent = this.hunger > 60 || this.toilet > 60; this.goal = urgent || chance(0.25) ? this.pickGoal() : null; }
      if (this.goal && this.goal !== 'gate') field = g.buildingField(this.goal);
    }
    if (field) {
      const dHere = field[here];
      if (dHere === 0) { this.arriveAt(); return; }
      if (dHere > 0) {
        const opts = [];
        for (const [dx, dy] of DIRS4) {
          const nx = this.cx + dx, ny = this.cy + dy;
          if (!w.inb(nx, ny)) continue;
          const j = w.idx(nx, ny);
          if (field[j] >= 0 && field[j] < dHere) opts.push([nx, ny]);
        }
        if (opts.length) { const o = pick(opts); this.nx = o[0]; this.ny = o[1]; return; }
      } else if (this.goal && this.goal !== 'gate') this.goal = null;
    }
    // Random walk on path network, avoiding immediate backtrack
    const opts = [];
    for (const [dx, dy] of DIRS4) {
      const nx = this.cx + dx, ny = this.cy + dy;
      if (!w.inb(nx, ny)) continue;
      const j = w.idx(nx, ny);
      if (w.path[j] && j !== this.prev) opts.push([nx, ny]);
    }
    if (!opts.length && this.prev >= 0 && w.path[this.prev]) opts.push([this.prev % w.W, (this.prev / w.W) | 0]);
    if (opts.length > 1) {
      // At a junction: favour directions that lead toward species this guest hasn't seen yet
      const weight = (o) => {
        const px = this.cx + (o[0] - this.cx) * 5, py = this.cy + (o[1] - this.cy) * 5;
        let wgt = 1;
        for (const d of g.creatures()) {
          if (d.loose || d.carried || this.seenSpecies.has(d.species)) continue;
          if (dist2(px, py, d.x, d.y) < 64) wgt += 3;
        }
        // and away from packed crowds
        const c = g.crowd ? g.crowd[w.idx(o[0], o[1])] : 0;
        return wgt / (1 + c * 0.15);
      };
      const o = weightedPick(opts, weight) || pick(opts);
      this.nx = o[0]; this.ny = o[1];
    } else if (opts.length) { this.nx = opts[0][0]; this.ny = opts[0][1]; }
  }

  arriveAt() {
    const g = this.game, w = g.world;
    if (this.state === 'leave' || (this.state === 'flee' && this.fleeGate)) {
      if (this.state === 'flee') g.stats.fledToday++;
      g.removeGuest(this, true); return;
    }
    if (this.state === 'flee') {
      const b = g.nearShelterWithSpace(this.cx, this.cy);
      if (b) { b.inside++; this.inBuilding = b; this.state = 'shelter'; this.hidden = true; this.insideT = 6; return; }
      this.fleeGate = true; return;
    }
    const b = this.goal;
    this.goal = null;
    if (!b || b === 'gate' || !w.buildings.has(b.id)) return;
    const def = BUILDINGS[b.type];
    if (def.power && !b.powered) { this.happy -= 3; return; }
    if (b.type === 'viewing' || b.type === 'arena' || b.type === 'feedshow') { this.happy = Math.min(100, this.happy + 4); this.lookAtDinos(); return; }
    if (b.type === 'tour' || b.type === 'gyro') {
      if (!b.queue) b.queue = [];
      if (b.queue.length >= 16) { this.happy -= 3; return; }
      b.queue.push(this); this.state = 'queue'; this.hidden = true; this.inBuilding = null; this.insideT = 40;
      return;
    }
    // Enter
    this.inBuilding = b; b.inside++;
    this.hidden = true; this.state = 'inside';
    this.insideT = b.type === 'hotel' ? randf(18, 30) : randf(3, 7);
    const price = def.income || 0;
    if (price) { g.earn(price * g.priceMul, 'shops'); this.spent += price; b.visitors++; b.revenue = (b.revenue || 0) + price * g.priceMul; }
    if (b.type === 'restaurant') this.hunger = 0;
    if (b.type === 'restroom') this.toilet = 0;
    if (b.type === 'shop') this.shopUrge = 0;
    if (b.type === 'hotel') { this.stay += 10; this.hunger = 10; this.toilet = 10; }
    this.happy = Math.min(100, this.happy + (def.appeal || 3));
  }

  leaveCover() {
    if (this.inBuilding) this.inBuilding.inside = Math.max(0, this.inBuilding.inside - 1);
    if (this.state === 'queue') for (const b of this.game.world.buildings.values()) if (b.queue) { const k = b.queue.indexOf(this); if (k >= 0) b.queue.splice(k, 1); }
    this.inBuilding = null; this.hidden = false;
  }

  exitBuilding() {
    const b = this.inBuilding;
    if (b) b.inside = Math.max(0, b.inside - 1);
    this.inBuilding = null; this.hidden = false;
    if (this.state === 'shelter') this.fear = 0;
    this.state = this.stay <= 0 || this.happy < 15 ? 'leave' : 'walk';
    this.fleeGate = false;
  }
}
