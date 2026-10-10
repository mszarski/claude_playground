// ---------- Disasters & random events ----------
'use strict';

const DISASTERS = {
  storm: { name: 'Tropical Storm', icon: '⛈', desc: 'Lightning, power cuts, stressed dinosaurs.' },
  outage: { name: 'System Failure', icon: '⚡', desc: 'A disgruntled programmer shuts down the grid.' },
  quake: { name: 'Earthquake', icon: '〰', desc: 'Shakes fences and buildings apart.' },
  volcano: { name: 'Volcanic Eruption', icon: '🌋', desc: 'Lava pours from the mountain.' },
  disease: { name: 'Outbreak', icon: '☣', desc: 'A dinosaur falls ill. It may spread.' },
  rampage: { name: 'Rampage', icon: '‼', desc: 'A dinosaur goes berserk.' },
  raptors: { name: 'Clever Girls', icon: '◆', desc: 'Raptors coordinate a fence probe.' },
  inspection: { name: 'Safety Inspection', icon: '✔', desc: 'Inspector visit: fines or praise.' },
};

class Events {
  constructor(game) {
    this.game = game;
    this.storm = null;
    this.quake = 0;
    this.volcano = null;
    this.auto = true;
    this.nextDay = 4;
    this.lightning = 0; // flash timer
    this.bolts = [];
    this.history = [];
  }

  update(dt) {
    const g = this.game, w = g.world;
    // ----- Storm -----
    const s = this.storm;
    if (s) {
      s.t -= dt;
      if (s.phase === 'warn' && s.t <= 0) {
        s.phase = 'active'; s.t = s.duration;
        g.log('The storm has hit the island!', 'bad', null, true);
        g.sound('thunder');
      } else if (s.phase === 'active') {
        s.strikeT -= dt;
        if (s.strikeT <= 0) { s.strikeT = randf(1.8, 4.5); this.strike(); }
        s.flickerT -= dt;
        if (s.flickerT <= 0) {
          s.flickerT = randf(6, 12);
          if (chance(0.35)) {
            const plants = g.buildingsOfType('power');
            if (plants.length) {
              const p = pick(plants); p.offline = randf(10, 22); w.powerDirty = true;
              g.log('Storm surge knocked a Power Plant offline!', 'bad', p);
            }
          }
        }
        if (s.t <= 0) { this.storm = null; g.log('The storm has passed.', 'good'); }
      }
    }
    if (this.lightning > 0) this.lightning -= dt;
    for (const b of this.bolts) b.life -= dt;
    this.bolts = this.bolts.filter((b) => b.life > 0);

    // ----- Earthquake -----
    if (this.quake > 0) {
      this.quake -= dt;
      g.shake = Math.max(g.shake, 4);
      this.quakeTick = (this.quakeTick || 0) - dt;
      if (this.quakeTick <= 0) {
        this.quakeTick = 0.45;
        for (let k = 0; k < 14; k++) {
          const x = randi(1, w.W - 2), y = randi(1, w.H - 2);
          const j = w.idx(x, y);
          if (w.fence[j] === F_ELECTRIC || w.fence[j] === F_WALL) g.damageFence(x, y, randf(15, 45), null);
        }
        for (const b of w.buildings.values()) if (chance(0.06)) g.damageBuilding(b, randf(15, 50), null);
      }
      if (this.quake <= 0) g.log('The earthquake is over. Check your fences!', 'warn');
    }

    // ----- Power outage -----
    const pw = w.power;
    if (pw.outage > 0) {
      const before = pw.backupDelay > 0;
      pw.outage -= dt;
      if (pw.backupDelay > 0) pw.backupDelay -= dt;
      if (before && pw.backupDelay <= 0 && g.hasBuilding('backup')) { w.powerDirty = true; g.log('Backup generators online.', 'good'); }
      if (pw.backupDelay <= 0 && g.hasBuilding('backup') && pw.backupFuel > 0) {
        pw.backupFuel -= dt;
        if (pw.backupFuel <= 0) { w.powerDirty = true; g.log('Backup generators out of fuel!', 'bad'); }
      }
      if (pw.outage <= 0) { pw.outage = 0; w.powerDirty = true; g.log('Main grid restored. Fences are live.', 'good', null, true); g.sound('powerup'); }
    }

    // ----- Volcano -----
    const v = this.volcano;
    if (v) {
      v.t -= dt;
      if (v.phase === 'warn') {
        g.shake = Math.max(g.shake, 1);
        if (v.t <= 0) { v.phase = 'erupt'; v.t = 22; g.log('ERUPTION! Lava is flowing!', 'bad', w.volcano, true); g.sound('boom'); }
      } else if (v.phase === 'erupt') {
        g.shake = Math.max(g.shake, 2);
        v.spawnT -= dt;
        v.ashT -= dt;
        if (v.ashT <= 0) { v.ashT = 0.05; g.particles.push({ x: w.volcano.x + 0.5 + randf(-0.5, 0.5), y: w.volcano.y + randf(-0.5, 0.5), vx: randf(-1, 1), vy: randf(-4, -2), life: randf(1, 2.5), max: 2.5, color: chance(0.3) ? '#f86a20' : '#4a4040', size: chance(0.5) ? 2 : 1, grav: 1.5, z: true }); }
        if (v.spawnT <= 0) { v.spawnT = 0.18; this.spreadLava(); }
        if (v.t <= 0) { this.volcano = null; g.log('The eruption has subsided.', 'warn'); }
      }
    }
    // lava cooling
    this.lavaTick = (this.lavaTick || 0) - dt;
    if (this.lavaTick <= 0) {
      this.lavaTick = 1;
      let changed = false;
      for (let i = 0; i < w.W * w.H; i++) {
        if (w.terrain[i] !== T_LAVA) continue;
        w.lavaT[i] -= 1;
        if (w.lavaT[i] <= 0 && !this.volcano) { w.terrain[i] = T_BASALT; changed = true; w.markDirty(i % w.W, (i / w.W) | 0); }
      }
      if (changed) { w.regionsDirty = true; w.flowCache.clear(); w.pathVersion++; }
    }

    // ----- Disease spread -----
    this.diseaseTick = (this.diseaseTick || 0) - dt;
    if (this.diseaseTick <= 0) {
      this.diseaseTick = 5;
      const sick = g.dinos.filter((d) => d.sick > 0);
      if (sick.length && !g.hasBuilding('vet')) {
        for (const s of sick) for (const d of g.dinos) {
          if (d.sick > 0 || d.carried) continue;
          if (dist2(d.x, d.y, s.x, s.y) < 16 && chance(0.05)) { d.sick = 20; g.log(`${d.name} the ${d.sp.name} has caught the illness!`, 'warn', d); }
        }
      }
    }
  }

  onNewDay(day) {
    if (!this.auto) return;
    if (day < this.nextDay) return;
    const g = this.game;
    if (g.dinos.length === 0) { this.nextDay = day + 1; return; }
    const early = day < 8;
    const opts = [
      ['storm', 30], ['outage', early ? 4 : 14], ['quake', early ? 3 : 10], ['disease', early ? 5 : 14], ['rampage', early ? 4 : 12],
      ['raptors', g.dinos.some((d) => d.species === 'raptor') ? 14 : 0], ['inspection', 12],
      ['volcano', day > 12 ? 6 : 0],
    ];
    const ev = weightedPick(opts, (o) => o[1]);
    const m = this.evMul || 1;
    this.nextDay = day + Math.max(1, Math.round(randi(2, 4) / m));
    if (ev) this.trigger(ev[0]);
  }

  trigger(type) {
    const g = this.game, w = g.world;
    this.history.push({ type, day: g.day });
    switch (type) {
      case 'storm':
        if (this.storm) return;
        this.storm = { phase: 'warn', t: 18, duration: randf(40, 60), strikeT: 1, flickerT: 4 };
        g.log('Weather warning: a tropical storm will hit in a few hours. Consider evacuating guests.', 'warn', null, true);
        break;
      case 'outage':
        w.power.outage = randf(45, 65);
        w.power.backupDelay = 4;
        w.power.backupFuel = 50;
        w.powerDirty = true;
        g.log('SYSTEM FAILURE! The main grid is down. Electric fences are offline!', 'bad', null, true);
        g.sound('powerdown');
        break;
      case 'quake':
        this.quake = randf(6, 9);
        g.log('EARTHQUAKE!', 'bad', null, true);
        g.sound('boom');
        break;
      case 'volcano':
        if (this.volcano) return;
        this.volcano = { phase: 'warn', t: 15, spawnT: 0, ashT: 0, front: [] };
        g.log('Seismic alert: the volcano is about to erupt!', 'warn', w.volcano, true);
        break;
      case 'disease': {
        const cands = g.dinos.filter((d) => !d.sick && !d.carried);
        if (!cands.length) return;
        const d = pick(cands); d.sick = 30;
        g.log(`${d.name} the ${d.sp.name} has fallen ill!` + (g.hasBuilding('vet') ? ' The vets are on it.' : ' Build a Vet Clinic!'), 'warn', d, true);
        break;
      }
      case 'rampage': {
        const cands = g.dinos.filter((d) => !d.carried && d.sedatedT <= 0 && !d.loose && d.sp.strength >= 3);
        if (!cands.length) return;
        const d = pick(cands); d.stress = 100; d.deterT = 0; d.comfort = 0; d.rageT = 40;
        g.log(`${d.name} the ${d.sp.name} is in a rage and charging the fences!`, 'bad', d, true);
        break;
      }
      case 'raptors': {
        const rs = g.dinos.filter((d) => d.species === 'raptor' && !d.loose && !d.carried);
        for (const d of rs) { d.deterT = 0; d.stress = Math.max(d.stress, 75); d.state = 'idle'; d.stateT = 0; }
        if (rs.length) g.log('The raptors are testing the fences... systematically.', 'warn', rs[0], true);
        break;
      }
      case 'inspection': {
        const loose = g.dinos.filter((d) => d.loose).length;
        const broken = g.countBrokenFences();
        const recent = g.recentDeaths();
        if (loose || recent > 0 || broken > 3) {
          const fine = Math.min(120000, 25000 + loose * 10000 + recent * 15000 + broken * 1000);
          g.spend(fine, 'fines');
          g.reputation = Math.max(0, g.reputation - 6);
          g.log(`Safety inspector fined the park ${fmtMoney(fine)}!`, 'bad', null, true);
        } else {
          g.earn(15000, 'grants');
          g.reputation = Math.min(100, g.reputation + 5);
          g.log('Safety inspection passed with flying colors! +$15k grant.', 'good', null, true);
        }
        break;
      }
    }
  }

  strike() {
    const g = this.game, w = g.world;
    let tx, ty;
    const r = rand();
    const blds = Array.from(w.buildings.values());
    if (r < 0.35 && blds.length) { const b = pick(blds); tx = b.x + randi(0, b.w - 1); ty = b.y + randi(0, b.h - 1); }
    else if (r < 0.75) {
      // random fence tile
      for (let k = 0; k < 60; k++) {
        const x = randi(1, w.W - 2), y = randi(1, w.H - 2);
        const f = w.fence[w.idx(x, y)];
        if (f === F_ELECTRIC || f === F_WALL) { tx = x; ty = y; break; }
      }
    }
    if (tx === undefined) { tx = randi(2, w.W - 3); ty = randi(2, w.H - 3); }
    this.lightning = 0.25;
    this.bolts.push({ x: tx + 0.5, y: ty + 0.5, life: 0.35, seed: randi(0, 9999) });
    g.soundAt('thunder', tx, ty, 1);
    const j = w.idx(tx, ty);
    if (w.fence[j] === F_ELECTRIC || w.fence[j] === F_WALL) g.damageFence(tx, ty, randf(50, 110), null);
    const b = w.buildings.get(w.bld[j]);
    if (b) {
      g.damageBuilding(b, randf(40, 120), null);
      if (b.type === 'power') { b.offline = randf(12, 25); w.powerDirty = true; g.log('Lightning struck a Power Plant! It is offline.', 'bad', b); }
    }
    if (w.terrain[j] === T_FOREST && chance(0.3)) { w.terrain[j] = T_GRASS; w.invalidate(tx, ty); }
    g.sparks(tx + 0.5, ty + 0.5, '#f8f8a0', 10);
    for (const d of g.dinos) if (dist2(d.x, d.y, tx, ty) < 36) d.stress = Math.min(100, d.stress + 15);
  }

  spreadLava() {
    const g = this.game, w = g.world;
    const v = this.volcano;
    const vx = w.volcano.x, vy = w.volcano.y;
    // pick a random existing lava tile (or the crater) and flow outward
    let src;
    if (!v.front.length || chance(0.15)) src = [vx + randi(-1, 1), vy + randi(-1, 1)];
    else src = pick(v.front);
    const opts = [];
    for (const [dx, dy] of DIRS4) {
      const nx = src[0] + dx, ny = src[1] + dy;
      if (!w.inb(nx, ny)) continue;
      const t = w.terrain[w.idx(nx, ny)];
      if (t === T_DEEP || t === T_WATER || t === T_VOLCANO || t === T_LAVA) continue;
      // prefer flowing away from crater
      const out = dist(nx, ny, vx, vy) - dist(src[0], src[1], vx, vy);
      opts.push([nx, ny, out > 0 ? 3 : 0.4]);
    }
    if (!opts.length) return;
    const o = weightedPick(opts, (q) => q[2]);
    const [nx, ny] = o;
    if (dist(nx, ny, vx, vy) > 12) return;
    const j = w.idx(nx, ny);
    w.terrain[j] = T_LAVA; w.lavaT[j] = randf(35, 60);
    w.path[j] = 0;
    if (w.fence[j]) { w.fence[j] = F_NONE; w.fenceHp[j] = 0; }
    const b = w.buildings.get(w.bld[j]);
    if (b) g.damageBuilding(b, 9999, null);
    v.front.push([nx, ny]);
    if (v.front.length > 80) v.front.shift();
    // creatures caught in lava
    for (const d of g.dinos) if (!d.carried && d.tx === nx && d.ty === ny) g.killDino(d, 'lava');
    for (const h of g.humans()) if (!h.hidden && h.tx === nx && h.ty === ny) g.humanKilled(h, null);
    w.invalidate(nx, ny);
  }
}
