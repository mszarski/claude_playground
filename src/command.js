// ---------- RTS-style commands for staff and helicopters (optional: everyone also works on their own) ----------
'use strict';

const CommandMixin = {
  // Give an order to a group of units. target: { dino } | { fence:{x,y} } | { building } | { x, y }
  commandUnits(units, target) {
    const w = this.world;
    let n = 0;
    const live = units.filter((u) => u && !u.dead);
    for (const u of live) {
      if (u.kind === 'heli') {
        if (target.dino && !target.dino.dead) {
          const d = target.dino;
          if (d.sedatedT > 0 && (d.loose || d.relocate) && !d.carried && !u.cargo) { u.target = d; u.state = 'fetch'; n++; }
          else if (d.sedatedT <= 0 && (u.state === 'parked' || u.state === 'return') && !u.cargo) { if (!this.acuStrike(d, u)) n++; }
        }
        continue;
      }
      if (u.kind === 'ranger') {
        if (target.dino) u.order = { type: 'hunt', d: target.dino };
        else u.order = { type: 'move', x: target.x, y: target.y };
      } else if (u.kind === 'worker') {
        if (target.fence && this.needsRepair(target.fence.x, target.fence.y)) u.order = { type: 'repair', x: target.fence.x, y: target.fence.y };
        else if (target.building && target.building.hp < target.building.maxHp) u.order = { type: 'repairB', b: target.building, x: target.building.x, y: target.building.y };
        else u.order = { type: 'move', x: target.x, y: target.y };
        u.state = 'idle'; u.path = null; u.thinkT = 0; u.job = null;
      } else continue;
      u.path = null; u.thinkT = 0; u.reached = false;
      n++;
    }
    if (target.dino && live.some((u) => u.kind === 'ranger')) target.dino.orderSedate = true;
    return n;
  },
  releaseUnits(units) {
    for (const u of units) if (u && u.order) { u.order = null; u.path = null; u.thinkT = 0; u.reached = false; }
  },
  needsRepair(x, y) {
    const w = this.world;
    if (!w.inb(x, y)) return false;
    const i = w.idx(x, y), f = w.fence[i];
    return f === F_BROKEN || (isSolidFence(f) && w.fenceHp[i] < FENCE_DEF[f].hp);
  },

  // One click for a breach: alarm on, every ranger on the nearest dangerous escapee, choppers scrambled
  respondToBreach() {
    const loose = this.creatures().filter((d) => d.loose && !d.carried && d.sedatedT <= 0);
    if (!loose.length) return 'Nothing is loose.';
    if (!this.alarm && loose.some((d) => d.sp.danger >= 4)) { this.alarm = true; this.log('EVACUATION ALARM: guests are heading to shelters.', 'warn', null, true); }
    const worst = loose.slice().sort((a, b) => b.sp.danger - a.sp.danger);
    let rangers = 0;
    for (const s of this.staff) {
      if (s.role !== 'ranger' || s.dead) continue;
      const d = worst.slice(0, 3).sort((a, b) => dist2(s.x, s.y, a.x, a.y) - dist2(s.x, s.y, b.x, b.y))[0];
      s.order = { type: 'hunt', d }; s.path = null; s.thinkT = 0; d.orderSedate = true; rangers++;
    }
    let helis = 0;
    for (const h of this.helis) {
      if (h.cargo || !(h.state === 'parked' || h.state === 'return')) continue;
      const d = worst[helis % worst.length];
      if (!this.acuStrike(d, h)) helis++;
    }
    this.log(`Breach response: ${rangers} ranger${rangers === 1 ? '' : 's'} and ${helis} helicopter${helis === 1 ? '' : 's'} dispatched.`, 'info', worst[0], true);
    return null;
  },
};
for (const [k, v] of Object.entries(CommandMixin)) Game.prototype[k] = v;

function orderText(u) {
  const o = u.order;
  if (!o) return 'Working on their own';
  if (o.type === 'hunt') return `Ordered: tranquilize ${o.d.name} the ${o.d.sp.name}`;
  if (o.type === 'repair' || o.type === 'repairB') return 'Ordered: repair';
  return u.reached ? 'Holding position' : 'Ordered: move';
}
