// Paddock blueprints, one-click tree cover / feeders / fence repair, auto feeder on hatch, undo of a stamp.
(() => {
  startGame({ seed: window.TEST_SEED || 1337, difficulty: 'calm', size: 'medium' });
  const g = window.GAME, w = g.world, s = g.start;
  const P = buildStarterPark(g, { staff: true });
  w.computeRegions();
  const out = {};
  const regAt = (x, y) => w.regionAt(x, y);
  let reg = regAt(P[0][0] + 3, P[0][1] + 6);
  for (const t of reg.tiles) if (w.terrain[t] === T_FOREST) w.terrain[t] = T_GRASS;
  w.invalidate(); w.computeRegions();
  reg = regAt(P[0][0] + 3, P[0][1] + 6);
  out.cover0 = Math.round(reg.forest / reg.size * 100);
  out.planted = g.plantCover(reg);
  const reg2 = regAt(P[0][0] + 3, P[0][1] + 6);
  out.cover1 = Math.round(reg2.forest / reg2.size * 100);
  // copy the paddock and stamp it into open land to the east
  const bp = g.copyPaddock(reg2);
  out.bp = [bp.w, bp.h, bp.fences.length, bp.trees.length, bp.feeders.length];
  T.clear(s.x - 30, s.y - 50, s.x + 30, s.y - 28);
  const money = g.money;
  const r = g.stamp(bp, s.x - 7, s.y - 46);
  out.stamped = r.n; out.stampCost = r.plan.cost; out.spent = money - g.money;
  const newReg = regAt(s.x - 7 + Math.floor(bp.w / 2), s.y - 46 + Math.floor(bp.h / 2));
  out.newPaddock = !!newReg && !newReg.public && newReg.size > 50;
  out.newFeeders = newReg ? newReg.feeders.herb.length + newReg.feeders.carn.length : 0;
  // undo the stamp gets the money back
  const u = g.undo(); w.computeRegions();
  out.undo = u && u.label; out.refunded = Math.round(g.money - (money - out.spent));
  // auto feeder: hatch a carnivore into the empty-carnivore paddock that has none
  const pr = regAt(P[2][0] + 6, P[2][1] + 6);
  for (const f of pr.feeders.carn.slice()) g.removeBuildingFx(f);
  w.computeRegions();
  g.hatch('dilo', P[2][0] + 6, P[2][1] + 6); w.computeRegions();
  out.autoFeeder = regAt(P[2][0] + 6, P[2][1] + 6).feeders.carn.length;
  // break three fences, repair them all in one go
  const fx = P[1][0], fy = P[1][1] + 4;
  for (let k = 0; k < 3; k++) { const i = w.idx(fx, fy + k); w.fence[i] = F_BROKEN; w.fenceHp[i] = 0; }
  w.invalidate();
  out.brokenBefore = g.countBrokenFences();
  out.repair = g.repairFences();
  out.brokenAfter = g.countBrokenFences();
  for (let i = 0; i < 30 * 5; i++) g.update(1 / 30);
  RENDER.draw(1 / 30, UIX);
  out.ok = out.planted > 0 && out.cover1 >= 28 && out.stamped > 20 && out.newPaddock && out.newFeeders >= 1 && out.undo === 'paddock blueprint' && Math.abs(out.refunded - out.spent) < 2 && out.autoFeeder === 1 && out.brokenAfter === 0;
  return out;
})()
