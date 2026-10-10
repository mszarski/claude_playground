// Isla Sorna Cleanup with competent play: two walled holding paddocks + rangers.
(() => {
  const out = [];
  for (let rep = 0; rep < 3; rep++) {
    startGame({ scenario: 'sorna', difficulty: 'normal' }); const g = GAME, w = g.world, s = g.start;
    const flat = (x0, y0, x1, y1) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) { if (!w.inb(x, y)) continue; const i = w.idx(x, y); if (w.terrain[i] === T_ROCK || w.terrain[i] === T_SAND) w.terrain[i] = T_GRASS; } w.invalidate(); };
    flat(s.x - 22, s.y - 26, s.x + 22, s.y - 9);
    T.rect(s.x - 20, s.y - 24, s.x - 3, s.y - 11, F_WALL); T.rect(s.x + 3, s.y - 24, s.x + 18, s.y - 11, F_WALL);
    g.placeBuilding('feeder_h', s.x - 11, s.y - 13); g.placeBuilding('feeder_c', s.x + 10, s.y - 13);
    for (const [t, x, y] of [['ranger', s.x - 4, s.y - 6], ['ranger', s.x + 8, s.y - 5], ['maint', s.x - 10, s.y - 6]]) { const sp = T.spot(t, x, y); if (sp) g.placeBuilding(t, sp.x, sp.y); }
    w.computeRegions();
    const spent = 520000 - Math.round(g.money);
    let day = null;
    const daily = [];
    const L = []; g.on('log', e => { if (/broke|collapsed|BREACH|died/.test(e.msg)) L.push('D' + g.day + ' ' + e.msg.slice(0, 70)); });
    for (let i = 0; i < 11 * 60 && !g.scenarioDone; i++) { T.run(1); if (i % 60 === 0) daily.push(g.dinos.filter(d => !d.loose && !d.carried).length + '/' + g.dinos.length); }
    out.push({ daily: daily.join(' '), L: L.slice(0, 6), spent, done: g.scenarioDone, day: g.day, prog: SCENARIOS.sorna.progress(g), dinoDeaths: g.stats.dinoDeaths, staffDeaths: g.stats.staffDeaths, money: Math.round(g.money) });
  }
  return out;
})()
