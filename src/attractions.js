// ---------- Attractions: gyrospheres, the raptor arena and feeding shows ----------
'use strict';

// A glass ball that rolls two guests around the paddock next to its station
class Gyro {
  constructor(game, station, tile) {
    this.id = _eid++;
    this.kind = 'gyro';
    this.game = game; this.station = station;
    const w = game.world;
    this.home = { x: tile % w.W, y: (tile / w.W) | 0 };
    this.x = this.home.x + 0.5; this.y = this.home.y + 0.5;
    this.region = w.region[tile];
    this.state = 'parked'; this.riders = []; this.path = null; this.pi = 0;
    this.trip = 0; this.facing = 1; this.anim = 0; this.moving = false; this.viewT = 0; this.waitT = 0;
    this.wrecked = 0; this.dead = false;
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
    this.anim += dt; this.moving = false;
    if (this.wrecked > 0) { this.wrecked -= dt; if (this.wrecked <= 0) this.dead = true; return; }
    if (!w.buildings.has(this.station.id)) { this.remove(); return; }
    const reg = w.regions[this.region];
    // the paddock was opened up or rebuilt: bring everyone home
    if (!reg || reg.public || w.region[w.idx(this.tx, this.ty)] !== this.region) { this.remove(); return; }
    const powered = this.station.powered && w.power.outage <= 0;
    if (this.state === 'parked') {
      const q = this.station.queue || [];
      this.waitT += dt;
      if (q.length && powered && (q.length >= 2 || this.waitT > 4)) {
        if (g.gyros.some((o) => o !== this && o.station === this.station && o.state === 'parked' && o.id < this.id)) return;
        this.riders = q.splice(0, 2);
        const price = BUILDINGS.gyro.income * g.priceMul;
        for (const r of this.riders) { r.state = 'ride'; r.inJeep = this; g.earn(price, 'shops'); r.spent += price; this.station.visitors++; this.station.revenue = (this.station.revenue || 0) + price; }
        this.state = 'roll'; this.trip = randf(22, 34); this.waitT = 0;
      }
      return;
    }
    if (!powered) return; // balls lock up without power
    this.trip -= dt;
    this.viewT -= dt;
    if (this.viewT <= 0) {
      this.viewT = 1;
      for (const r of this.riders) {
        let joy = 0;
        for (const d of g.dinos) {
          if (d.carried || d.loose || dist2(this.x, this.y, d.x, d.y) > 36) continue;
          const fresh = !r.seenSpecies.has(d.species);
          if (fresh) r.seenSpecies.add(d.species);
          joy += d.sp.appeal * (fresh ? 2 : 0.3) * (d.sp.diet === 'carn' ? 1.5 : 1); // up close is the whole point
        }
        r.happy = Math.min(100, r.happy + Math.min(joy, 18) + 0.5);
      }
    }
    const pass = (x, y) => w.dinoPass(x, y) && w.region[w.idx(x, y)] === this.region;
    if (this.state === 'roll' && this.trip <= 0) {
      this.state = 'home';
      this.path = w.bfs(this.tx, this.ty, pass, (x, y) => x === this.home.x && y === this.home.y, 6000); this.pi = 0;
      if (!this.path) { this.x = this.home.x + 0.5; this.y = this.home.y + 0.5; }
    }
    if (!this.path || this.pi >= this.path.length) {
      if (this.state === 'home') {
        for (const r of this.riders) { r.happy = Math.min(100, r.happy + 6); this.unload(r); }
        this.riders = []; this.state = 'parked'; this.path = null;
        return;
      }
      const t = reg.tiles[Math.floor(rand() * reg.tiles.length)];
      const tx = t % w.W, ty = (t / w.W) | 0;
      this.path = w.bfs(this.tx, this.ty, pass, (x, y) => x === tx && y === ty, 3000); this.pi = 0;
      return;
    }
    if (stepAlong(this, dt, 1.7, pass, w)) this.path = null;
    this.moving = true;
  }
}

const AttractionMixin = {
  // Paddock tiles a Gyrosphere Station can launch into (touching it, or just across its fence)
  gyroAccess(b) {
    const w = this.world, out = [];
    for (let y = b.y - 2; y <= b.y + b.h + 1; y++) for (let x = b.x - 2; x <= b.x + b.w + 1; x++) {
      if (x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h) continue;
      if (!w.inb(x, y) || !w.dinoPass(x, y)) continue;
      const reg = w.regions[w.region[w.idx(x, y)]];
      if (reg && !reg.public && reg.size < 1200) out.push(w.idx(x, y));
    }
    return out;
  },

  gyroAttacked(gy, dino) {
    if (gy.wrecked) return;
    gy.wrecked = 6;
    const n = gy.riders.length;
    for (const r of gy.riders) { r.hidden = false; r.x = gy.x; r.y = gy.y; this.humanKilled(r, null); }
    gy.riders = [];
    this.burst(gy.x, gy.y, '#c8e8f8', 18); this.burst(gy.x, gy.y, '#c82020', n * 4);
    this.shake = Math.max(this.shake, 3);
    this.soundAt('crash', gy.x, gy.y, 1);
    this.log(`${dino.name} the ${dino.sp.name} cracked open a gyrosphere!${n ? ' ' + n + ' rider' + (n > 1 ? 's' : '') + ' lost.' : ''}`, 'bad', { x: gy.x, y: gy.y }, true);
  },

  updateAttractions(dt) {
    const w = this.world;
    this.gyros = this.gyros || [];
    // gyrosphere stations keep two balls in their paddock
    for (const st of this.buildingsOfType('gyro')) {
      if (!st.queue) st.queue = [];
      st.queue = st.queue.filter((gu) => !gu.dead && gu.state === 'queue');
      st.gyroT = (st.gyroT || 0) - dt;
      if (st.gyroT > 0) continue;
      st.gyroT = 6;
      const mine = this.gyros.filter((gy) => gy.station === st && !gy.dead);
      const acc = this.gyroAccess(st);
      st.noPaddock = !acc.length;
      if (mine.length < 2 && acc.length) this.gyros.push(new Gyro(this, st, pick(acc)));
    }
    for (const gy of this.gyros) gy.update(dt);
    this.gyros = this.gyros.filter((gy) => !gy.dead);

    const day = this.hour >= 8 && this.hour < 20;
    // raptor arena: a show every 40-60s when the raptors are next door
    for (const b of this.buildingsOfType('arena')) {
      if (b.showing > 0) b.showing -= dt;
      b.showT = (b.showT === undefined ? randf(10, 20) : b.showT) - dt;
      if (b.showT > 0) continue;
      b.showT = randf(40, 60);
      const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
      const raptors = this.dinos.filter((d) => d.species === 'raptor' && !d.loose && !d.carried && d.sedatedT <= 0 && dist2(cx, cy, d.x, d.y) < 144);
      b.noRaptors = raptors.length < 2;
      if (!b.powered || !day || b.noRaptors) continue;
      const crowd = this.guests.filter((gu) => !gu.hidden && dist2(gu.x, gu.y, cx, cy) < 100);
      const pay = Math.min(crowd.length, 50) * 18 * this.priceMul;
      this.earn(pay, 'shops');
      b.revenue = (b.revenue || 0) + pay; b.visitors += crowd.length;
      for (const gu of crowd) { gu.happy = Math.min(100, gu.happy + 15); if (chance(0.25)) { gu.thought = pick(['CLEVER!', 'WOW!', 'EEK!']); gu.thoughtT = 2; } }
      if (pay) this.addFloat(cx, b.y - 0.5, '+' + fmtMoney(pay), '#f8d040');
      b.showing = 4;
      this.stats.shows = (this.stats.shows || 0) + 1;
      this.soundAt('screech', cx, cy, 1);
      const avgStress = raptors.reduce((a, d) => a + d.stress, 0) / raptors.length;
      for (const d of raptors) d.stress = Math.min(100, d.stress + 10);
      // every show is a gamble
      const risk = 0.04 + avgStress / 500 + (raptors.some((d) => d.unstable) ? 0.08 : 0) - (this.research.has('rangers') ? 0.02 : 0);
      if (chance(risk)) this.arenaIncident(b, raptors);
    }
    // feeding grandstand: noon and 6pm
    for (const b of this.buildingsOfType('feedshow')) {
      const slot = this.hour >= 18 ? 18 : this.hour >= 12 ? 12 : -1;
      const key = this.day * 100 + slot;
      if (slot < 0 || b.lastShow === key) continue;
      if (this.hour - slot > 1) { b.lastShow = key; continue; } // missed it (e.g. loaded mid-day)
      b.lastShow = key;
      if (!b.powered) continue;
      this.feedingShow(b);
    }
  },

  arenaIncident(b, raptors) {
    const r = pick(raptors);
    if (chance(0.6)) {
      this.spend(25000, 'fines');
      this.reputation = Math.max(0, this.reputation - 4);
      this.burst(b.x + b.w / 2, b.y + b.h / 2, '#c82020', 10);
      this.soundAt('scream', b.x + b.w / 2, b.y, 1);
      this.log(`A trainer was mauled during the raptor show! ${r.name} got too clever. −$25k`, 'bad', b, true);
      return;
    }
    // the raptors use the commotion to test their fence
    const reg = this.world.regionAt(r.tx, r.ty);
    if (!reg) return;
    const fences = this.paddockFences(reg).filter((j) => isSolidFence(this.world.fence[j]));
    if (!fences.length) return;
    const w = this.world;
    const j = fences.sort((a, c) => dist2(a % w.W, (a / w.W) | 0, r.x, r.y) - dist2(c % w.W, (c / w.W) | 0, r.x, r.y))[0];
    this.log('The raptor show went wrong: the pack is going for the fence!', 'bad', r, true);
    for (const d of raptors) { d.stress = 100; d.deterT = 0; }
    this.damageFence(j % w.W, (j / w.W) | 0, 9999, r);
  },

  feedingShow(b) {
    const w = this.world, cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    const cands = this.dinos.filter((d) => d.sp.diet === 'carn' && !d.loose && !d.carried && d.sedatedT <= 0 && dist2(cx, cy, d.x, d.y) < 144);
    b.noCarnivore = !cands.length;
    if (!cands.length) return;
    const d = cands.sort((a, c) => c.sp.appeal - a.sp.appeal)[0];
    // walk the star to the edge of its paddock nearest the stands
    const reg = w.regionAt(d.tx, d.ty);
    if (reg) {
      let best = null, bd = 1e9;
      for (const t of reg.tiles) { const x = t % w.W, y = (t / w.W) | 0, dd = dist2(x, y, cx, cy); if (dd < bd && w.dinoPass(x, y)) { bd = dd; best = [x, y]; } }
      if (best) { const p = w.bfs(d.tx, d.ty, w.dinoPass, (x, y) => x === best[0] && y === best[1], 3000); if (p) { d.path = p; d.pi = 0; d.state = 'walk'; } }
    }
    setTimeoutSim(this, 5, () => {
      if (d.dead || d.loose) return;
      d.hunger = Math.max(0, d.hunger - 60);
      this.burst(d.x + d.facing * 0.5, d.y, '#c83020', 12);
      this.soundAt('chomp', d.x, d.y, 1);
      this.soundAt(d.sp.size >= 3 ? 'roar' : 'screech', d.x, d.y, 1);
      const crowd = this.guests.filter((gu) => !gu.hidden && dist2(gu.x, gu.y, cx, cy) < 144);
      const pay = Math.min(crowd.length, 60) * 10 * this.priceMul;
      this.earn(pay, 'shops'); b.revenue = (b.revenue || 0) + pay; b.visitors += crowd.length;
      for (const gu of crowd) { gu.happy = Math.min(100, gu.happy + 20); if (chance(0.3)) { gu.thought = pick(['WHOA!', 'GROSS!', 'AGAIN!']); gu.thoughtT = 2; } }
      if (pay) this.addFloat(cx, b.y - 0.5, '+' + fmtMoney(pay), '#f8d040');
      this.log(`Feeding show: ${d.name} the ${d.sp.name} put on a show for ${crowd.length} guests.`, 'good', d);
    });
  },
};
for (const [k, v] of Object.entries(AttractionMixin)) Game.prototype[k] = v;
