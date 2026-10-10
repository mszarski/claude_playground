'use strict';

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
