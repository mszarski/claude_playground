// ---------- Entities: dinosaurs, guests, staff, helicopter, particles ----------
'use strict';

const DINO_NAMES = ['Rexy', 'Blue', 'Delta', 'Echo', 'Charlie', 'Clever Girl', 'Big One', 'Spike', 'Cera', 'Littlefoot', 'Bumpy', 'Ducky', 'Petrie', 'Sarah', 'Roberta', 'Tiny', 'Nibbles', 'Chomper', 'Mabel', 'Gertie', 'Bob', 'Kevin', 'Pebbles', 'Doris', 'Hammond', 'Muldoon', 'Grant', 'Ellie', 'Ian', 'Dennis', 'Lex', 'Tim', 'Henry', 'Nedry Jr', 'Sprinkles', 'Mr. Teeth', 'Steve', 'Bubbles', 'Ziggy', 'Moxie'];

function stepAlong(e, dt, speed, passFn, world) {
  if (!e.path || e.pi >= e.path.length) return true;
  const [wx, wy] = e.path[e.pi];
  if (passFn && !passFn.call(world, wx, wy) && !e.allowGoalBlock) { e.path = null; return true; }
  const tx = wx + 0.5, ty = wy + 0.5;
  const dx = tx - e.x, dy = ty - e.y;
  const d = Math.sqrt(dx * dx + dy * dy);
  const s = speed * dt;
  if (Math.abs(dx) > 0.02) e.facing = dx > 0 ? 1 : -1;
  if (d <= s) { e.x = tx; e.y = ty; e.pi++; return e.pi >= e.path.length; }
  e.x += (dx / d) * s; e.y += (dy / d) * s;
  e.moving = true;
  return false;
}

let _eid = 1;

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
  }

  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }
  get camouflaged() { return !!(this.sp.camo && this.loose && this.state !== 'hunt' && this.state !== 'attack' && this.sedatedT <= 0); }
  get speed() {
    let s = this.sp.speed * 0.55;
    if (this.hp < this.sp.hp * 0.4) s *= 0.7;
    if (this.sick > 0) s *= 0.7;
    return s;
  }

  update(dt) {
    const g = this.game, w = g.world;
    this.anim += dt;
    this.moving = false;
    if (this.flash > 0) this.flash -= dt;
    if (this.carried) return;
    this.age += dt;

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
      return f === F_ELECTRIC || f === F_WALL;
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
      if (!this.attackBuilding && (w.fence[j] === F_ELECTRIC || w.fence[j] === F_WALL)) { tgt = { x, y, j, fence: true }; break; }
      if (this.attackBuilding && w.bld[j]) { tgt = { x, y, j, b: w.buildings.get(w.bld[j]) }; break; }
    }
    if (!tgt) { this.state = 'idle'; this.stateT = 1; this.attackBuilding = false; return; }
    this.facing = tgt.x + 0.5 > this.x ? 1 : -1;
    this.lunge = 0.25;
    const str = this.sp.strength;
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
      } else if (type === F_WALL) {
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
    if (this.loose && this.sp.size >= 2) {
      for (const j of g.jeeps) {
        if (j.wrecked || !j.riders.length) continue;
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
        t.hp -= this.sp.strength * (t.sp.territorial ? 2.5 : 12);
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
    this.sprite = pick(PEOPLE.guests);
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
      if (field[here] === -1 || !g.anyShelterSpace()) { field = g.gateField(); this.state = 'flee'; this.fleeGate = true; }
    } else if (this.state === 'leave') {
      field = g.gateField();
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

// ---------------- Staff: rangers & engineers ----------------
class Staff {
  constructor(game, role, home) {
    this.id = _eid++;
    this.kind = role;
    this.role = role;
    this.game = game;
    this.home = home;
    const w = game.world;
    const spots = w.adjacentPassable(home, w.humanPass);
    const s = spots.length ? pick(spots) : [home.x, home.y + home.h];
    this.x = s[0] + 0.5; this.y = s[1] + 0.5;
    this.state = 'idle'; this.stateT = rand() * 2;
    this.path = null; this.pi = 0;
    this.facing = 1; this.anim = rand(); this.moving = false;
    this.target = null; this.cd = 0; this.dead = false; this.hidden = false;
    this.thinkT = rand();
    this.sprite = role === 'ranger' ? PEOPLE.ranger : PEOPLE.worker;
  }
  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }

  update(dt) {
    const g = this.game, w = g.world;
    this.anim += dt; this.moving = false;
    if (this.cd > 0) this.cd -= dt;
    this.thinkT -= dt;
    if (!w.buildings.has(this.home.id)) { this.dead = true; return; }
    if (this.role === 'ranger') this.updateRanger(dt);
    else this.updateWorker(dt);
  }

  updateRanger(dt) {
    const g = this.game, w = g.world;
    if (this.thinkT <= 0) {
      this.thinkT = 1.0;
      // pick most dangerous loose dino (or ordered sedation)
      let best = null, bs = -1e9;
      for (const d of g.creatures()) {
        if (d.carried || d.sedatedT > 0) continue;
        if (!d.loose && !d.orderSedate) continue;
        if (d.camouflaged && dist2(this.x, this.y, d.x, d.y) > 9) continue;
        const s = d.sp.danger * 10 - dist(this.x, this.y, d.x, d.y);
        if (s > bs) { bs = s; best = d; }
      }
      if (best !== this.target) { this.target = best; this.path = null; }
      if (this.target) {
        const d = this.target;
        const dd = dist(this.x, this.y, d.x, d.y);
        if (dd > 4.2 && (!this.path || this.thinkCount++ % 3 === 0)) {
          const tx = d.x, ty = d.y;
          const p = w.bfs(this.tx, this.ty, w.humanPass, (x, y) => dist2(x + 0.5, y + 0.5, tx, ty) < 3.5 * 3.5 && w.humanPass(x, y), 4000);
          this.path = p; this.pi = 0;
          this.state = p ? 'move' : 'idle';
        }
      } else if (!this.path) {
        // patrol near home
        if (chance(0.3)) {
          const tx = this.home.x + randi(-5, 5), ty = this.home.y + randi(-5, 5);
          if (w.humanPass(tx, ty)) { this.path = w.bfs(this.tx, this.ty, w.humanPass, (x, y) => x === tx && y === ty, 600); this.pi = 0; }
        }
      }
      this.thinkCount = (this.thinkCount || 0) + 1;
    }
    if (this.target) {
      const d = this.target;
      const dd = dist(this.x, this.y, d.x, d.y);
      if (dd <= 4.5 && this.cd <= 0 && !d.carried && d.sedatedT <= 0) {
        this.cd = 1.6;
        this.facing = d.x > this.x ? 1 : -1;
        g.darts.push({ x: this.x, y: this.y - 0.3, t: d, life: 1.5 });
        g.soundAt('dart', this.x, this.y, 0.7);
        this.path = null;
        return;
      }
      if (dd < 2.2 && d.sp.danger >= 4 && !this.retreating) {
        // too close: back off
        const ax = this.x - d.x, ay = this.y - d.y, al = Math.hypot(ax, ay) || 1;
        const nx = this.x + ax / al * 2.1 * dt, ny = this.y + ay / al * 2.1 * dt;
        if (w.humanPass(Math.floor(nx), Math.floor(ny))) { this.x = nx; this.y = ny; this.moving = true; }
        return;
      }
      if (dd <= 3.8) { this.path = null; return; }
    }
    if (this.path) { if (stepAlong(this, dt, 1.9, w.humanPass, w)) this.path = null; }
  }

  updateWorker(dt) {
    const g = this.game, w = g.world;
    // Engineers are not heroes: drop everything and run from loose predators
    const threat = g.threatNear(this.x, this.y, 6);
    if (threat && this.state !== 'flee') {
      this.state = 'flee'; this.job = null;
      const hx = this.home.x, hy = this.home.y;
      this.path = w.bfs(this.tx, this.ty, w.humanPass, (x, y) => dist2(x, y, hx, hy) < 6, 5000);
      this.pi = 0;
    }
    if (this.state === 'flee') {
      if (!this.path || stepAlong(this, dt, 2.2, w.humanPass, w)) { this.path = null; if (!g.threatNear(this.x, this.y, 9)) this.state = 'idle'; }
      return;
    }
    if (this.state === 'repair') {
      const j = this.job;
      const fx = j % w.W, fy = (j / w.W) | 0;
      if (Math.abs(fx + 0.5 - this.x) + Math.abs(fy + 0.5 - this.y) > 1.6) { this.state = 'idle'; return; }
      this.facing = fx + 0.5 > this.x ? 1 : -1;
      this.workT = (this.workT || 0) + dt;
      if (this.workT > 0.5) { this.workT = 0; g.sparks(fx + 0.5, fy + 0.5, '#f8e080', 2); }
      if (this.jobType === 'fence') {
        const f = w.fence[j];
        if (f === F_BROKEN) {
          // rebuild if no dino standing there
          const blocked = g.dinos.some((d) => d.tx === fx && d.ty === fy && !d.carried);
          if (!blocked) {
            const orig = w.fenceOrig[j] || F_ELECTRIC;
            const cost = Math.round(FENCE_DEF[orig].cost * 0.6);
            if (g.money > -50000) {
              g.spend(cost, 'repairs');
              w.fence[j] = orig; w.fenceHp[j] = FENCE_DEF[orig].hp * 0.4;
              w.invalidate(fx, fy); g.onFenceRebuilt(fx, fy);
            }
          }
          this.state = 'idle';
        } else if (f === F_ELECTRIC || f === F_WALL) {
          const max = FENCE_DEF[f].hp;
          w.fenceHp[j] = Math.min(max, w.fenceHp[j] + dt * 22);
          g.spend(dt * 6, 'repairs');
          if (w.fenceHp[j] >= max) { this.state = 'idle'; w.markDirty(fx, fy); }
        } else this.state = 'idle';
      } else if (this.jobType === 'building') {
        const b = this.jobB;
        if (!b || !w.buildings.has(b.id)) { this.state = 'idle'; return; }
        b.hp = Math.min(b.maxHp, b.hp + dt * 30);
        g.spend(dt * 8, 'repairs');
        if (b.hp >= b.maxHp) this.state = 'idle';
      }
      return;
    }
    if (this.state === 'move') {
      if (stepAlong(this, dt, 1.7, w.humanPass, w)) {
        this.path = null;
        this.state = this.job != null ? 'repair' : 'idle';
        this.workT = 0;
      }
      return;
    }
    if (this.thinkT > 0) return;
    this.thinkT = 1.5 + rand();
    // Find nearest repair job reachable
    const claimed = new Set(g.staff.filter((s) => s !== this && s.role === 'worker' && s.job != null && s.state !== 'idle').map((s) => s.job));
    const needs = (x, y) => {
      if (!w.inb(x, y)) return false;
      const j = w.idx(x, y);
      if (claimed.has(j)) return false;
      if (g.threatNear(x + 0.5, y + 0.5, 8)) return false;
      const f = w.fence[j];
      if (f === F_BROKEN) return true;
      if ((f === F_ELECTRIC || f === F_WALL) && w.fenceHp[j] < FENCE_DEF[f].hp * 0.98) return true;
      const b = w.buildings.get(w.bld[j]);
      if (b && b.hp < b.maxHp * 0.95 && !claimed.has('b' + b.id)) return true;
      return false;
    };
    const p = w.bfs(this.tx, this.ty, w.humanPass, needs, 6000);
    if (p && p.length) {
      const goal = p.pop();
      const j = w.idx(goal[0], goal[1]);
      const b = w.buildings.get(w.bld[j]);
      if (b && !(w.fence[j])) { this.jobType = 'building'; this.jobB = b; this.job = 'b' + b.id; }
      else { this.jobType = 'fence'; this.job = j; }
      this.path = p; this.pi = 0; this.state = 'move';
      if (!p.length) this.state = 'repair';
    } else {
      this.job = null;
      // idle wander near home
      if (dist(this.x, this.y, this.home.x, this.home.y) > 4) {
        const hp = w.bfs(this.tx, this.ty, w.humanPass, (x, y) => dist2(x, y, this.home.x, this.home.y) < 9, 3000);
        if (hp) { this.path = hp; this.pi = 0; this.state = 'move'; }
      }
    }
  }
}

// ---------------- Pteranodons (aviary flyers) ----------------
class Ptera {
  constructor(game, x, y) {
    this.id = _eid++;
    this.kind = 'dino'; this.isPtera = true;
    this.game = game; this.species = 'ptera'; this.sp = SPECIES.ptera;
    this.x = x + 0.5; this.y = y + 0.5; this.alt = 1.5; this.home = { x, y };
    this.hp = this.sp.hp; this.hunger = 20; this.stress = 10; this.comfort = 80; this.comfortParts = {};
    this.sedation = 0; this.sedatedT = 0; this.loose = false; this.carried = null; this.dead = false;
    this.name = pick(DINO_NAMES); this.facing = 1; this.anim = rand() * 6; this.flash = 0; this.kills = 0;
    this.tgt = null; this.prey = null; this.cd = 0; this.sick = 0;
  }
  get tx() { return Math.floor(this.x); }
  get ty() { return Math.floor(this.y); }
  get speed() { return this.sp.speed * 0.6; }
  aviaryHere() {
    for (const a of this.game.buildingsOfType('aviary')) {
      if (this.x >= a.x && this.x < a.x + a.w && this.y >= a.y && this.y < a.y + a.h && a.hp >= a.maxHp * 0.35) return a;
    }
    return null;
  }
  sedate() {
    this.sedatedT = 70; this.prey = null; this.tgt = null;
    this.game.log(`${this.name} the Pteranodon has been sedated and dropped out of the sky.`, 'good', this);
  }
  update(dt) {
    const g = this.game, w = g.world;
    this.anim += dt;
    if (this.flash > 0) this.flash -= dt;
    if (this.carried) return;
    if (this.cd > 0) this.cd -= dt;
    if (this.sedatedT > 0) {
      this.sedatedT -= dt; this.alt = Math.max(0, this.alt - dt * 3);
      if (this.loose && !g.hasBuilding('helipad')) {
        this.truckT = (this.truckT || 0) + dt;
        if (this.truckT > 18) {
          const dest = g.findPaddockFor(this);
          if (dest) { g.spend(12000, 'ops'); this.x = dest.x + 0.5; this.y = dest.y + 0.5; this.sedatedT = 3; this.truckT = 0; g.log(`Ground crew returned ${this.name} the Pteranodon to the Aviary (-$12k).`, 'good', this); }
          else this.truckT = 10;
        }
      }
      if (this.sedatedT <= 0) this.sedation = 0;
      return;
    }
    const av = this.aviaryHere();
    const wasLoose = this.loose;
    this.loose = !av;
    if (this.loose && !wasLoose) {
      g.stats.escapes++; g.reputation = Math.max(0, g.reputation - 2);
      g.log(`CONTAINMENT BREACH! ${this.name} the Pteranodon has escaped the Aviary!`, 'bad', this, true);
      g.sound('alarm'); g.emit('breach', [this]);
    } else if (!this.loose && wasLoose) g.log(`${this.name} the Pteranodon is back in the Aviary.`, 'good', this);
    if (this.hp <= 0) { g.killDino(this, 'injuries'); return; }
    const fly = (tx, ty, sp) => {
      const dx = tx - this.x, dy = ty - this.y, d = Math.sqrt(dx * dx + dy * dy);
      if (Math.abs(dx) > 0.05) this.facing = dx > 0 ? 1 : -1;
      if (d < sp * dt) { this.x = tx; this.y = ty; return true; }
      this.x += dx / d * sp * dt; this.y += dy / d * sp * dt; return false;
    };
    if (!this.loose) {
      this.alt = 0.9 + Math.sin(this.anim * 0.9) * 0.3;
      if (!this.tgt || fly(this.tgt[0], this.tgt[1], this.speed * 0.6)) this.tgt = [av.x + 0.8 + rand() * (av.w - 1.6), av.y + 2 + rand() * (av.h - 2.8)];
      return;
    }
    // Loose: hunt people from the sky
    this.hunger = Math.min(100, this.hunger + dt * 0.8);
    if (!this.prey && this.cd <= 0) {
      let best = null, bd = 144;
      for (const h of g.humans()) { if (h.hidden || h.dead) continue; const d2 = dist2(this.x, this.y, h.x, h.y); if (d2 < bd) { bd = d2; best = h; } }
      this.prey = best;
    }
    const p = this.prey;
    if (p && (p.dead || p.hidden)) this.prey = null;
    if (this.prey) {
      const d = dist(this.x, this.y, p.x, p.y);
      this.alt = clamp(d * 0.4, 0.2, 2.5);
      if (fly(p.x, p.y, this.speed * 1.5) || d < 0.45) {
        g.humanKilled(p, this);
        this.kills++; this.hunger = 0; this.prey = null; this.cd = randf(18, 30); this.tgt = null;
        g.soundAt('screech', this.x, this.y, 1);
      }
      return;
    }
    this.alt = Math.min(3, this.alt + dt);
    if (!this.tgt || fly(this.tgt[0], this.tgt[1], this.speed)) {
      for (let k = 0; k < 10; k++) {
        const tx = this.x + randf(-12, 12), ty = this.y + randf(-12, 12);
        if (w.inb(Math.floor(tx), Math.floor(ty)) && w.isLand(w.terrain[w.idx(Math.floor(tx), Math.floor(ty))])) { this.tgt = [tx, ty]; break; }
      }
      if (!this.tgt) this.tgt = [w.W / 2, w.H / 2];
    }
    if (chance(dt * 0.05)) g.soundAt('screech', this.x, this.y, 0.7);
  }
}

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
