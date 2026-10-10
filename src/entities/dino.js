// ---------- Dinosaurs ----------
'use strict';

class Dino {
  constructor(game, species, x, y) {
    this.id = _eid++;
    this.kind = 'dino';
    this.game = game;
    this.species = species;
    this.sp = SPECIES[species];
    this.x = x + 0.5; this.y = y + 0.5;
    this.home = { x, y };
    this.hp = this.sp.hp;
    this.hunger = 25; this.stress = 15; this.comfort = 70;
    this.sedation = 0; this.sedatedT = 0;
    this.state = 'idle'; this.stateT = randf(1, 3);
    this.path = null; this.pi = 0;
    this.facing = chance(0.5) ? 1 : -1;
    this.anim = rand() * 10; this.moving = false;
    this.name = pick(DINO_NAMES);
    this.sick = 0; this.attackCD = 0; this.flash = 0; this.deterT = 0;
    this.loose = false; this.carried = null; this.target = null;
    this.thinkT = 0; this.age = 0; this.kills = 0;
    this.comfortParts = {};
    this.roarCD = randf(5, 20);
    this.lastBreach = 0;
    this.growth = 0.6; // hatchlings grow up over a couple of days
    this.ageDays = 0;
    this.lifespan = this.sp.life * randf(0.9, 1.15);
  }
  get elderly() { return this.ageDays > this.lifespan * 0.8; }
  get sellValue() { return Math.round(this.sp.cost * (this.growth < 1 ? 0.25 : this.elderly ? 0.15 : 0.35)); }
  get str() { return this.sp.strength * this.growth; }

  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }
  get camouflaged() { return !!(this.sp.camo && this.loose && this.state !== 'hunt' && this.state !== 'attack' && this.sedatedT <= 0); }
  get speed() {
    let s = this.sp.speed * 0.55;
    if (this.hp < this.sp.hp * 0.4) s *= 0.7;
    if (this.sick > 0) s *= 0.7;
    if (this.elderly) s *= 0.8;
    return s;
  }

  update(dt) {
    const g = this.game, w = g.world;
    this.anim += dt;
    this.moving = false;
    if (this.flash > 0) this.flash -= dt;
    if (this.carried) return;
    this.age += dt;
    this.ageDays += dt / (g.secPerHour * 24);
    if (this.ageDays > this.lifespan && !this.loose && this.sedatedT <= 0) { g.killDino(this, 'old age', true); return; }
    if (this.growth < 1) {
      this.growth = Math.min(1, this.growth + dt / 150);
      if (this.growth >= 1) g.log(`${this.name} the ${this.sp.name} is fully grown.`, 'info', this);
    }

    // Sedation
    if (this.sedatedT > 0) {
      this.sedatedT -= dt;
      // No helicopter? A ground crew trucks sedated escapees home (slow and pricey)
      if ((this.loose || this.relocate) && !g.hasBuilding('helipad')) {
        this.truckT = (this.truckT || 0) + dt;
        if (this.truckT > 18) {
          const dest = g.findPaddockFor(this);
          if (dest) {
            g.spend(12000, 'ops');
            g.burst(this.x, this.y, '#e8d8a0', 8);
            this.x = dest.x + 0.5; this.y = dest.y + 0.5; this.home = { x: dest.x, y: dest.y };
            this.loose = false; this.relocate = false; this.orderSedate = false; this.sedatedT = 3; this.truckT = 0;
            g.world.regionsDirty = true;
            g.log(`Ground crew trucked ${this.name} the ${this.sp.name} back to a paddock (-$12k). A helipad is faster.`, 'good', this);
          } else this.truckT = 10;
        }
      } else this.truckT = 0;
      if (this.sedatedT <= 0) {
        this.sedation = 0; this.stress = Math.max(0, this.stress - 30);
        this.state = 'idle'; this.stateT = 1;
      }
      return;
    }
    if (this.sedation > 0) this.sedation = Math.max(0, this.sedation - dt * 0.02);

    // Needs
    const hrate = this.sp.diet === 'carn' ? 1.0 : 0.85;
    this.hunger = Math.min(100, this.hunger + dt * hrate * (this.loose ? 1.4 : 1));
    if (this.hunger >= 100) this.hp -= dt * 0.8;
    if (this.sick > 0) {
      // Vets cure quickly; untreated illness slowly runs its course but costs health
      if (g.hasBuilding('vet')) { this.sick -= dt * 2.5; }
      else { this.sick -= dt * 0.12; this.hp -= dt * (this.hunger > 60 ? 0.3 : 0.14); }
      if (this.sick <= 0) { this.sick = 0; g.log(`${this.name} the ${this.sp.name} has recovered.`, 'good', this); }
    }
    if (this.hp <= 0) { g.killDino(this, this.hunger >= 100 ? 'starvation' : this.sick > 0 ? 'illness' : 'injuries'); return; }
    if (this.hp < this.sp.hp && this.hunger < 60 && this.sick <= 0) this.hp = Math.min(this.sp.hp, this.hp + dt * 0.5);

    // Stress drifts toward a target from comfort / hunger / weather
    let target = (100 - this.comfort) * 0.85;
    if (this.hunger > 55) target += (this.hunger - 55) * 1.1;
    if (g.events.storm && g.events.storm.phase === 'active') target += 18;
    if (g.events.quake > 0) target += 30;
    if (this.loose) target += 15;
    if (g.isNight && this.sp.clever) target += 12; // raptors prowl at night
    target = clamp(target, 0, 100);
    this.stress += (target - this.stress) * Math.min(1, dt * 0.08);
    if (this.deterT > 0) this.deterT -= dt;
    if (this.attackCD > 0) this.attackCD -= dt;

    // Stuck inside a wall/building (e.g. fence rebuilt on top) -> pop to nearest free tile
    if (!w.dinoPass(this.tx, this.ty)) {
      const p = w.bfs(this.tx, this.ty, () => true, (x, y) => w.dinoPass(x, y), 400);
      if (p && p.length) { const [nx, ny] = p[p.length - 1]; this.x = nx + 0.5; this.y = ny + 0.5; }
      this.path = null;
    }

    this.roarCD -= dt;
    if (this.roarCD <= 0) {
      this.roarCD = randf(15, 40);
      const call = this.sp.diet === 'carn' ? (this.sp.size >= 3 ? 'roar' : 'screech') : (this.sp.size >= 3 ? 'bellow' : 'honk');
      if (this.sp.size >= 2 || this.loose || chance(0.3)) g.soundAt(call, this.x, this.y, this.loose ? 1 : 0.6);
    }

    this.thinkT -= dt;
    switch (this.state) {
      case 'idle':
        this.stateT -= dt;
        if (this.stateT <= 0) this.decide();
        break;
      case 'walk':
      case 'tofeed':
      case 'tofence':
      case 'graze': {
        const sp = this.state === 'walk' ? this.speed * (this.loose && this.stress > 70 ? 1.3 : 1) : this.speed;
        const done = stepAlong(this, dt, sp, w.dinoPass, w);
        if (done) this.arrive();
        break;
      }
      case 'eat':
        this.stateT -= dt;
        if (this.stateT <= 0) { this.state = 'idle'; this.stateT = randf(1, 3); }
        break;
      case 'hunt':
        this.updateHunt(dt);
        break;
      case 'attack':
        this.updateFenceAttack(dt);
        break;
      default:
        this.state = 'idle';
    }
  }

  decide() {
    const g = this.game, w = g.world;
    const reg = w.regionAt(this.tx, this.ty);
    // Loose carnivores look for prey
    if (this.sp.diet === 'carn' && (this.loose ? this.hunger > 15 : this.hunger > 60)) {
      const prey = this.findPrey(this.loose ? 10 : 30);
      if (prey) { this.state = 'hunt'; this.target = prey; this.thinkT = 0; return; }
    }
    // Territorial giants fight rivals that share their paddock
    if (this.sp.territorial && this.stress > 40 && !this.loose) {
      const myReg = w.region[w.idx(this.tx, this.ty)];
      const rival = g.dinos.find((o) => o !== this && o.sp.territorial && !o.carried && o.sedatedT <= 0 && w.region[w.idx(o.tx, o.ty)] === myReg);
      if (rival && chance(0.5)) {
        this.state = 'hunt'; this.target = rival; this.thinkT = 0;
        if (!this.fightLogged) { this.fightLogged = true; g.log(`${this.name} the ${this.sp.name} is fighting ${rival.name} the ${rival.sp.name}! Separate them.`, 'bad', this, true); g.soundAt('roar', this.x, this.y, 1); }
        return;
      }
    }
    // Hungry: go to feeder
    if (this.hunger > 50 && reg) {
      const feeders = reg.feeders[this.sp.diet];
      if (feeders.length) {
        const f = pick(feeders);
        const p = w.bfs(this.tx, this.ty, w.dinoPass, (x, y) => {
          const b = w.buildingAt(x, y); return b && b.id === f.id;
        }, 3000);
        if (p && p.length) {
          p.pop(); // stop adjacent to the feeder
          this.path = p; this.pi = 0; this.state = 'tofeed'; this.feederTarget = f;
          if (!p.length) this.arrive();
          return;
        }
      }
      if (this.sp.diet === 'herb' && reg.forest > 0) {
        const p = w.bfs(this.tx, this.ty, w.dinoPass, (x, y) => w.terrain[w.idx(x, y)] === T_FOREST && w.dinoPass(x, y), 2500);
        if (p) { this.path = p; this.pi = 0; this.state = 'graze'; if (!p.length) this.arrive(); return; }
      }
    }
    // Fence testing
    const clever = this.sp.clever && chance(0.25);
    if (!this.loose && this.deterT <= 0 && (this.stress > 62 || clever)) {
      if (this.planFenceAttack(clever)) return;
    }
    // Loose: rampage on buildings if stressed and big
    if (this.loose && this.sp.strength >= 5 && this.stress > 60 && chance(0.4)) {
      const p = w.bfs(this.tx, this.ty, w.dinoPass, (x, y) => !!w.buildingAt(x, y), 900);
      if (p && p.length) { p.pop(); this.path = p; this.pi = 0; this.state = 'tofence'; this.attackBuilding = true; return; }
    }
    // Herbivores mostly sleep through the night
    if (g.isNight && this.sp.diet === 'herb' && !this.loose && this.hunger < 70 && chance(0.8)) { this.state = 'idle'; this.stateT = randf(6, 12); this.sleeping = true; return; }
    this.sleeping = false;
    // Wander
    if (chance(0.7)) {
      const rad = this.loose ? 10 : 7;
      for (let tries = 0; tries < 6; tries++) {
        const tx = this.tx + randi(-rad, rad), ty = this.ty + randi(-rad, rad);
        if (!w.dinoPass(tx, ty)) continue;
        if (!this.loose && w.region[w.idx(tx, ty)] !== w.region[w.idx(this.tx, this.ty)]) continue;
        // calm grazers stay near home even if a fence is down
        if (this.loose && this.sp.danger < 4 && this.stress < 55 && (dist2(tx, ty, this.home.x, this.home.y) > 36 || w.path[w.idx(tx, ty)])) continue;
        const p = w.bfs(this.tx, this.ty, w.dinoPass, (x, y) => x === tx && y === ty, 1500);
        if (p && p.length) { this.path = p; this.pi = 0; this.state = 'walk'; return; }
      }
    }
    this.state = 'idle'; this.stateT = randf(1.5, 4.5);
  }

  arrive() {
    const g = this.game, w = g.world;
    this.path = null;
    if (this.state === 'tofeed') {
      const f = this.feederTarget;
      if (f && w.buildings.has(f.id)) {
        const cost = this.sp.diet === 'carn' ? 400 : 150;
        g.spend(cost, 'food');
        this.hunger = Math.max(0, this.hunger - 80);
        this.state = 'eat'; this.stateT = 3;
        if (this.sp.diet === 'carn') { g.burst(this.x + this.facing * 0.5, this.y, '#c83020', 8); g.soundAt('chomp', this.x, this.y, 1); }
        else g.burst(this.x + this.facing * 0.5, this.y, '#e8c860', 6);
        return;
      }
    } else if (this.state === 'graze') {
      this.hunger = Math.max(0, this.hunger - 30);
      this.state = 'eat'; this.stateT = 4;
      return;
    } else if (this.state === 'tofence') {
      this.state = 'attack'; this.attackCD = 0.4; this.attackT = randf(8, 16);
      return;
    }
    this.state = 'idle'; this.stateT = randf(1, 4);
  }

  planFenceAttack(clever) {
    const w = this.game.world;
    const isFence = (x, y) => {
      if (!w.inb(x, y)) return false;
      const f = w.fence[w.idx(x, y)];
      return isSolidFence(f);
    };
    let goalFence = null;
    if (clever) {
      // Raptors look for the weakest point: unpowered or damaged fence along their paddock
      const reg = w.regionAt(this.tx, this.ty);
      if (reg) {
        let best = null, bestScore = -1;
        for (let k = 0; k < 60; k++) {
          const t = reg.tiles[Math.floor(rand() * reg.tiles.length)];
          const x = t % w.W, y = (t / w.W) | 0;
          for (const [dx, dy] of DIRS4) {
            if (!isFence(x + dx, y + dy)) continue;
            const j = w.idx(x + dx, y + dy);
            const score = (w.fence[j] === F_ELECTRIC && !w.fencePowered[j] ? 100 : 0) + (100 - w.fenceHp[j] / FENCE_DEF[w.fence[j]].hp * 100) + (w.fence[j] === F_WALL ? -50 : 0);
            if (score > bestScore) { bestScore = score; best = [x + dx, y + dy]; }
          }
        }
        if (best && bestScore > 20) goalFence = best;
        else if (!(this.stress > 62)) return false;
      }
    }
    const p = w.bfs(this.tx, this.ty, w.dinoPass, goalFence
      ? (x, y) => x === goalFence[0] && y === goalFence[1]
      : (x, y) => isFence(x, y), 2500);
    if (!p || !p.length) return false;
    p.pop();
    this.path = p; this.pi = 0; this.state = 'tofence'; this.attackBuilding = false;
    if (!p.length) this.arrive();
    return true;
  }

  updateFenceAttack(dt) {
    const g = this.game, w = g.world;
    this.attackT -= dt;
    if (this.attackT <= 0 || this.sedatedT > 0) { this.state = 'idle'; this.stateT = 2; this.attackBuilding = false; return; }
    if (this.attackCD > 0) return;
    this.attackCD = 1.1 + rand() * 0.5;
    // find adjacent target
    let tgt = null;
    for (const [dx, dy] of DIRS4) {
      const x = this.tx + dx, y = this.ty + dy;
      if (!w.inb(x, y)) continue;
      const j = w.idx(x, y);
      if (!this.attackBuilding && (isSolidFence(w.fence[j]))) { tgt = { x, y, j, fence: true }; break; }
      if (this.attackBuilding && w.bld[j]) { tgt = { x, y, j, b: w.buildings.get(w.bld[j]) }; break; }
    }
    if (!tgt) { this.state = 'idle'; this.stateT = 1; this.attackBuilding = false; return; }
    this.facing = tgt.x + 0.5 > this.x ? 1 : -1;
    this.lunge = 0.25;
    const str = this.str;
    if (tgt.fence) {
      const type = w.fence[tgt.j];
      let dmg;
      if (type === F_ELECTRIC && w.fencePowered[tgt.j]) {
        // ZAP
        dmg = str >= 8 && this.stress > 85 ? str * 0.8 : str * 0.25;
        this.flash = 0.25;
        this.hp -= 3;
        g.sparks(tgt.x + 0.5, tgt.y + 0.5);
        g.soundAt('zap', tgt.x, tgt.y, 1);
        if (!(str >= 8 && this.stress > 85)) {
          this.deterT = randf(20, 40) * (this.sp.clever ? 0.6 : 1);
          this.stress = Math.min(100, this.stress + 4);
          this.state = 'idle'; this.stateT = 2;
        }
      } else if (type === F_WALL || type === F_GATE) {
        dmg = str * 1.4;
        g.burst(tgt.x + 0.5, tgt.y + 0.5, '#9a9a90', 4);
        g.soundAt('thud', tgt.x, tgt.y, 0.6);
      } else {
        dmg = str * 4.5;
        g.burst(tgt.x + 0.5, tgt.y + 0.5, '#6a6a6a', 4);
        g.soundAt('thud', tgt.x, tgt.y, 0.6);
      }
      g.damageFence(tgt.x, tgt.y, dmg, this);
    } else if (tgt.b) {
      g.damageBuilding(tgt.b, str * 3, this);
      g.burst(tgt.x + 0.5, tgt.y + 0.5, '#8a6a4a', 5);
      g.soundAt('thud', tgt.x, tgt.y, 0.7);
    }
    g.shake = Math.max(g.shake, str >= 8 ? 2 : 0.5);
  }

  findPrey(range) {
    const g = this.game;
    let best = null, bd = range * range;
    if (this.loose) {
      for (const h of g.humans()) {
        if (h.hidden || h.dead) continue;
        const d = dist2(this.x, this.y, h.x, h.y);
        if (d < bd) { bd = d; best = h; }
      }
    }
    const angry = !this.loose && this.sp.size >= 3 && this.stress > 75;
    if ((this.loose || angry) && this.sp.size >= 2) {
      for (const j of g.jeeps) {
        if (j.wrecked || !j.riders.length) continue;
        if (angry && g.world.region[g.world.idx(j.tx, j.ty)] !== g.world.region[g.world.idx(this.tx, this.ty)]) continue;
        const d = dist2(this.x, this.y, j.x, j.y);
        if (d < bd) { bd = d; best = j; }
      }
    }
    // Contained carnivores (or loose ones) can also eat herbivores in their region
    const myReg = g.world.region[g.world.idx(this.tx, this.ty)];
    for (const d of g.dinos) {
      if (d === this || d.sp.diet !== 'herb' || d.carried || d.dead) continue;
      if (d.sp.size > this.sp.size + (this.sp.size >= 3 ? 0 : 0)) continue;
      if (!this.loose && g.world.region[g.world.idx(d.tx, d.ty)] !== myReg) continue;
      const dd = dist2(this.x, this.y, d.x, d.y);
      if (dd < bd) { bd = dd; best = d; }
    }
    return best;
  }

  updateHunt(dt) {
    const g = this.game, w = g.world;
    const t = this.target;
    if (!t || t.dead || t.hidden || t.carried || t.wrecked) { this.state = 'idle'; this.stateT = 1; this.target = null; return; }
    const d = dist(this.x, this.y, t.x, t.y);
    if (d > 16) { this.state = 'idle'; this.stateT = 1; this.target = null; return; }
    if (d < 0.75) {
      // Strike!
      if (this.attackCD > 0) return;
      this.attackCD = 1.2;
      this.lunge = 0.3;
      if (t.kind === 'jeep') {
        g.jeepAttacked(t, this); this.kills++; this.hunger = Math.max(0, this.hunger - 60);
        this.state = 'eat'; this.stateT = 6; this.target = null; return;
      }
      if (t.kind === 'dino') {
        t.hp -= this.str * (t.sp.territorial ? 2.5 : 12);
        t.flash = 0.2; t.stress = 100;
        if (t.sp.territorial && t.state !== 'hunt') { t.state = 'hunt'; t.target = this; }
        g.shake = Math.max(g.shake, t.sp.size >= 3 ? 2.5 : 0.5);
        g.burst(t.x, t.y, '#c83020', 6);
        g.soundAt('chomp', t.x, t.y, 1);
        if (t.hp <= 0) { g.killDino(t, t.sp.territorial ? `killed in a fight with ${this.name} the ${this.sp.name}` : `eaten by ${this.name} the ${this.sp.name}`); this.hunger = 0; this.kills++; this.state = 'eat'; this.stateT = 6; this.target = null; }
      } else {
        g.humanKilled(t, this);
        this.hunger = Math.max(0, this.hunger - 50); this.kills++;
        this.state = 'eat'; this.stateT = 5; this.target = null;
      }
      return;
    }
    // Chase: re-path periodically, otherwise move directly when close
    const sp = this.speed * (this.loose ? 1.55 : 1.3);
    if (d < 1.6) {
      const dx = t.x - this.x, dy = t.y - this.y;
      const nx = this.x + (dx / d) * sp * dt, ny = this.y + (dy / d) * sp * dt;
      if (w.dinoPass(Math.floor(nx), Math.floor(ny))) { this.x = nx; this.y = ny; this.moving = true; this.facing = dx > 0 ? 1 : -1; }
      return;
    }
    if (this.thinkT <= 0 || !this.path) {
      this.thinkT = 0.8;
      const ttx = Math.floor(t.x), tty = Math.floor(t.y);
      const p = w.bfs(this.tx, this.ty, w.dinoPass, (x, y) => Math.abs(x - ttx) <= 0 && Math.abs(y - tty) <= 0, 2500);
      if (!p) { this.state = 'idle'; this.stateT = 2; this.target = null; return; }
      this.path = p; this.pi = 0;
      this.allowGoalBlock = true;
    }
    stepAlong(this, dt, sp, null, w);
  }

  sedate() {
    this.sedatedT = 70; this.state = 'sedated'; this.path = null; this.target = null;
    this.game.log(`${this.name} the ${this.sp.name} has been sedated.`, 'good', this);
  }
}
