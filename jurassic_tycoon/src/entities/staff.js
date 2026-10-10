'use strict';

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
        } else if (isSolidFence(f)) {
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
      if ((isSolidFence(f)) && w.fenceHp[j] < FENCE_DEF[f].hp * 0.98) return true;
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
