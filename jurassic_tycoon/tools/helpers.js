// Shared helpers injected before test scripts
window.T = {
  spot(type, cx, cy, maxR = 20) {
    const g = window.GAME, w = g.world, d = BUILDINGS[type];
    for (let r = 0; r < maxR; r++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
      if (w.canPlaceBuilding(type, cx + dx, cy + dy)) return { x: cx + dx, y: cy + dy };
    }
    return null;
  },
  place(type, cx, cy) { const s = this.spot(type, cx, cy); return s ? window.GAME.placeBuilding(type, s.x, s.y) : null; },
  rect(x0, y0, x1, y1, type = F_ELECTRIC) {
    const g = window.GAME; let n = 0;
    for (let x = x0; x <= x1; x++) { n += g.placeFence(x, y0, type); n += g.placeFence(x, y1, type); }
    for (let y = y0 + 1; y < y1; y++) { n += g.placeFence(x0, y, type); n += g.placeFence(x1, y, type); }
    return n;
  },
  path(x0, y0, x1, y1) { const g = window.GAME; for (let x = Math.min(x0, x1); x <= Math.max(x0, x1); x++) g.placePath(x, y0); for (let y = Math.min(y0, y1); y <= Math.max(y0, y1); y++) g.placePath(x1, y); },
  run(secs) { const g = window.GAME; for (let i = 0; i < secs * 30; i++) g.update(1 / 30); },
  // clear rect of terrain to grass (test only)
  clear(x0, y0, x1, y1) { const w = window.GAME.world; for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) { const i = w.idx(x, y); if (w.terrain[i] === T_FOREST || w.terrain[i] === T_ROCK || w.terrain[i] === T_SAND) w.terrain[i] = T_GRASS; } w.invalidate(); },
};
