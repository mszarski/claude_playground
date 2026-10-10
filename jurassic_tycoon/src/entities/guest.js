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
    this.stay = randf(8, 18); // hours in park
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
    this.stay -= hrs;
    this.hunger += hrs * 5; this.toilet += hrs * 6; this.shopUrge += hrs * 3;

    // Threat check
    this.viewT -= dt;
    if (this.viewT <= 0) {
      this.viewT = 0.8;
      const threat = g.threatNear(this.x, this.y, 7);
      if ((threat || g.alarm || (g.sirenAlarm && g.inSirenRange(this.x, this.y))) && this.state !== 'flee') {
        this.state = 'flee'; this.goal = null; this.fear = 100;
        if (threat && chance(0.3)) { this.thought = 'AAAH!'; this.thoughtT = 2; g.soundAt('scream', this.x, this.y, 0.6); }
      }
      if (this.state !== 'flee') this.lookAtDinos();
      if (this.hunger > 85 || this.toilet > 85) this.happy -= 1.5;
      if (g.hour >= 20 || g.hour < 6) { if (!g.litNear(this.x, this.y)) this.happy -= 0.3; }
      this.happy = clamp(this.happy, 0, 100);
      if (this.happy < 12 && this.state === 'walk') this.state = 'leave';
      if (this.stay <= 0 && this.state === 'walk') this.state = 'leave';
    }
    if (this.thoughtT > 0) this.thoughtT -= dt; else this.thought = null;

    // Movement between tiles
    const speed = (this.state === 'flee' ? 2.0 : 0.95) * this.speedMul;
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
    else if (chance(0.45)) type = pick(['tour', 'tour', 'viewing', 'viewing', 'visitor', 'shop', 'restaurant']);
    if (!type) return null;
    const cands = g.buildingsOfType(type).filter((b) => b.powered || !BUILDINGS[b.type].power);
    if (!cands.length) { if (type === 'restroom' || type === 'restaurant') this.happy -= 2; return null; }
    const b = pick(cands);
    return b;
  }

  chooseNext(here) {
    const g = this.game, w = g.world;
    let field = null;
    if (this.state === 'flee') {
      field = g.shelterField();
      if (field[here] === -1 || !g.anyShelterSpace()) {
        if (g.gateLocked) field = null; // nowhere to run: panic in place
        else { field = g.gateField(); this.state = 'flee'; this.fleeGate = true; }
      }
    } else if (this.state === 'leave') {
      field = g.gateLocked ? null : g.gateField();
    } else {
      if (!this.goal || (this.goal !== 'gate' && !w.buildings.has(this.goal.id))) { this.goal = chance(0.25) ? this.pickGoal() : null; }
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
    if (opts.length) { const o = pick(opts); this.nx = o[0]; this.ny = o[1]; }
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
    if (b.type === 'viewing') { this.happy = Math.min(100, this.happy + 4); this.lookAtDinos(); return; }
    if (b.type === 'tour') {
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

  exitBuilding() {
    const b = this.inBuilding;
    if (b) b.inside = Math.max(0, b.inside - 1);
    this.inBuilding = null; this.hidden = false;
    if (this.state === 'shelter') this.fear = 0;
    this.state = this.stay <= 0 || this.happy < 15 ? 'leave' : 'walk';
    this.fleeGate = false;
  }
}
