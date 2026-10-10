// ---------- Vehicles: tour jeeps and the ACU helicopter ----------
'use strict';

// ---------------- Tour jeeps ----------------
class Jeep {
  constructor(game, station) {
    this.id = _eid++;
    this.kind = 'jeep';
    this.game = game; this.station = station;
    const acc = game.world.trackAccess(station);
    const [hx, hy] = acc[0];
    this.home = { x: hx, y: hy };
    this.x = hx + 0.5; this.y = hy + 0.5; this.cx = hx; this.cy = hy; this.nx = hx; this.ny = hy; this.prev = -1;
    this.state = 'parked'; this.riders = []; this.trip = 0; this.facing = 1; this.dir = 0;
    this.waitT = 0; this.viewT = 0; this.wrecked = 0; this.dead = false; this.anim = 0;
  }
  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }
  remove() { for (const r of this.riders) this.unload(r); this.riders = []; this.dead = true; }
  unload(r) {
    const w = this.game.world;
    const acc = w.accessTiles(this.station);
    r.hidden = false; r.inJeep = null;
    if (acc.length) { const t = pick(acc); r.cx = r.nx = t % w.W; r.cy = r.ny = (t / w.W) | 0; r.x = r.cx + 0.5; r.y = r.cy + 0.5; }
    r.state = r.stay <= 0 ? 'leave' : 'walk';
  }
  update(dt) {
    const g = this.game, w = g.world;
    this.anim += dt;
    if (this.wrecked > 0) { this.wrecked -= dt; if (this.wrecked <= 0) this.dead = true; return; }
    if (!w.buildings.has(this.station.id)) { this.remove(); return; }
    const powered = this.station.powered && w.power.outage <= 0;
    if (this.state === 'parked') {
      this.moving = false;
      const q = this.station.queue || [];
      this.waitT += dt;
      if (q.length && powered && (q.length >= 3 || this.waitT > 5)) {
        const other = g.jeeps.find((j) => j !== this && j.station === this.station && j.state === 'parked' && j.id < this.id);
        if (other) return; // first jeep in line loads first
        this.riders = q.splice(0, 6);
        for (const r of this.riders) { r.state = 'ride'; r.inJeep = this; g.earn(BUILDINGS.tour.income * g.priceMul, 'shops'); r.spent += BUILDINGS.tour.income; this.station.visitors++; this.station.revenue = (this.station.revenue || 0) + BUILDINGS.tour.income; }
        this.state = 'tour'; this.trip = 0; this.tripLen = randi(26, 46); this.waitT = 0;
      }
      return;
    }
    if (!powered) { this.moving = false; return; } // electric jeeps stall during outages
    this.viewT -= dt;
    if (this.viewT <= 0) {
      this.viewT = 1;
      for (const r of this.riders) {
        let joy = 0;
        for (const d of g.dinos) {
          if (d.carried || d.loose) continue;
          if (dist2(this.x, this.y, d.x, d.y) > 64) continue;
          const fresh = !r.seenSpecies.has(d.species);
          if (fresh) r.seenSpecies.add(d.species);
          joy += d.sp.appeal * (fresh ? 1.5 : 0.2);
        }
        r.happy = Math.min(100, r.happy + Math.min(joy, 15) + 0.5);
      }
    }
    const sp = 2.3;
    const tx = this.nx + 0.5, ty = this.ny + 0.5;
    const dx = tx - this.x, dy = ty - this.y, d = Math.sqrt(dx * dx + dy * dy);
    if (d > sp * dt) {
      this.x += dx / d * sp * dt; this.y += dy / d * sp * dt; this.moving = true;
      if (Math.abs(dx) > 0.05) this.facing = dx > 0 ? 1 : -1;
      return;
    }
    this.x = tx; this.y = ty;
    this.prev = w.idx(this.cx, this.cy); this.cx = this.nx; this.cy = this.ny;
    const here = w.idx(this.cx, this.cy);
    if (!w.track[here]) { this.remove(); return; }
    this.trip++;
    if (this.state === 'return' || this.trip >= this.tripLen) {
      if (this.cx === this.home.x && this.cy === this.home.y) {
        for (const r of this.riders) { r.happy = Math.min(100, r.happy + 8); this.unload(r); }
        this.riders = []; this.state = 'parked'; this.moving = false; return;
      }
      if (this.state !== 'return' || !this.path || !this.path.length) {
        this.state = 'return';
        this.path = w.bfs(this.cx, this.cy, (x, y) => w.inb(x, y) && !!w.track[w.idx(x, y)], (x, y) => x === this.home.x && y === this.home.y, 4000);
        if (!this.path) { this.state = 'tour'; this.tripLen += 10; }
      }
      if (this.path && this.path.length) { const [nx, ny] = this.path.shift(); this.nx = nx; this.ny = ny; return; }
    }
    const opts = [];
    for (const [ddx, ddy] of DIRS4) {
      const nx = this.cx + ddx, ny = this.cy + ddy;
      if (!w.inb(nx, ny)) continue;
      const j = w.idx(nx, ny);
      if (w.track[j] && j !== this.prev) opts.push([nx, ny]);
    }
    if (!opts.length && this.prev >= 0 && w.track[this.prev]) opts.push([this.prev % w.W, (this.prev / w.W) | 0]);
    if (opts.length) { const o = pick(opts); this.nx = o[0]; this.ny = o[1]; }
  }
}

// ---------------- ACU Helicopter ----------------
class Helicopter {
  constructor(game, pad) {
    this.id = _eid++;
    this.kind = 'heli';
    this.game = game; this.pad = pad;
    this.x = pad.x + 1.5; this.y = pad.y + 1.5;
    this.alt = 0; this.state = 'parked'; this.cargo = null; this.target = null;
    this.rotor = 0; this.facing = 1; this.thinkT = 0;
  }
  update(dt) {
    const g = this.game, w = g.world;
    this.rotor += dt * (this.alt > 0 || this.state !== 'parked' ? 30 : 0);
    if (!w.buildings.has(this.pad.id)) {
      if (this.cargo) { this.dropAt(this.tx, this.ty); }
      this.dead = true; return;
    }
    const fly = (tx, ty, sp) => {
      const dx = tx - this.x, dy = ty - this.y, d = Math.sqrt(dx * dx + dy * dy);
      if (Math.abs(dx) > 0.05) this.facing = dx > 0 ? 1 : -1;
      if (d < sp * dt) { this.x = tx; this.y = ty; return true; }
      this.x += dx / d * sp * dt; this.y += dy / d * sp * dt; return false;
    };
    switch (this.state) {
      case 'parked':
        this.alt = Math.max(0, this.alt - dt * 2);
        this.thinkT -= dt;
        if (this.thinkT <= 0) {
          this.thinkT = 1;
          if (this.cargo) { const dest = g.findPaddockFor(this.cargo); if (dest) { this.dest = dest; this.state = 'deliver'; } break; }
          const claimed = new Set(g.helis.filter((h) => h !== this && h.target).map((h) => h.target.id));
          const t = g.creatures().find((d) => d.sedatedT > 0 && (d.loose || d.relocate) && !d.carried && !claimed.has(d.id));
          if (t) { this.target = t; this.state = 'fetch'; g.soundAt('heli', this.x, this.y, 0.8); }
        }
        break;
      case 'strike': {
        // ACU gunship run: fly over the target and dart it from the air
        this.alt = Math.min(2.5, this.alt + dt * 2);
        const t = this.target;
        if (!t || t.dead || t.carried || t.sedatedT > 0) { this.target = null; this.state = 'return'; break; }
        if (dist(this.x, this.y, t.x, t.y) > 2.5) { fly(t.x, t.y, 6.5); break; }
        fly(t.x + Math.cos(g.time * 3) * 1.5, t.y + Math.sin(g.time * 3) * 1.5, 4);
        this.shotT = (this.shotT || 0) - dt;
        if (this.shotT <= 0) {
          this.shotT = 0.9;
          g.darts.push({ x: this.x, y: this.y - this.alt, t, life: 1.5 });
          g.soundAt('dart', this.x, this.y, 0.8);
        }
        break;
      }
      case 'fetch': {
        this.alt = Math.min(2.5, this.alt + dt * 2);
        const t = this.target;
        if (!t || t.dead || t.sedatedT <= 0 || t.carried) { this.target = null; this.state = 'return'; break; }
        if (fly(t.x, t.y - 0.2, 5.5)) {
          t.carried = this; this.cargo = t; this.target = null;
          g.log(`ACU airlifting ${t.name} the ${t.sp.name}.`, 'info', t);
          g.spend(2500, 'ops');
          const dest = g.findPaddockFor(t);
          if (dest) { this.dest = dest; this.state = 'deliver'; }
          else { this.state = 'return'; g.log(`No safe paddock for ${t.name}! Holding at helipad.`, 'warn', this); }
        }
        break;
      }
      case 'deliver': {
        this.alt = 2.5;
        if (fly(this.dest.x + 0.5, this.dest.y + 0.5, 5)) {
          // verify destination still valid
          const reg = w.regionAt(this.dest.x, this.dest.y);
          if (reg && !reg.public) { this.dropAt(this.dest.x, this.dest.y); this.state = 'return'; }
          else { const d2 = g.findPaddockFor(this.cargo); if (d2) this.dest = d2; else this.state = 'return'; }
        }
        break;
      }
      case 'return':
        this.alt = Math.min(2.5, this.alt + dt * 2);
        if (fly(this.pad.x + 1.5, this.pad.y + 1.5, 5.5)) this.state = 'parked';
        break;
    }
    if (this.cargo) { this.cargo.x = this.x; this.cargo.y = this.y; }
  }
  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }
  dropAt(x, y) {
    const d = this.cargo; if (!d) return;
    const moved = d.relocate && !d.loose;
    d.carried = null; d.x = x + 0.5; d.y = y + 0.5; d.sedatedT = 4; d.loose = d.isPtera ? d.loose : false; d.relocate = false; d.orderSedate = false;
    d.home = { x, y }; d.stress = Math.max(0, d.stress - 40);
    this.cargo = null;
    this.game.log(moved ? `${d.name} the ${d.sp.name} has been moved to the new paddock.` : `${d.name} the ${d.sp.name} is back in containment.`, 'good', d);
    this.game.burst(x + 0.5, y + 0.5, '#e8d8a0', 10);
  }
}
