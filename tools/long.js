// A scripted "competent player" over many days with random disasters on.
(() => {
  startGame({ difficulty: 'normal' });
  const g = window.GAME, w = g.world, s = g.start;
  T.clear(s.x - 26, s.y - 34, s.x + 26, s.y - 1);
  const out = [];
  const note = (m) => out.push(`D${g.day} $${Math.round(g.money / 1000)}k ★${g.stars}: ${m}`);
  T.place('power', s.x - 11, s.y - 4);
  T.place('hatchery', s.x + 6, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3); T.place('pylon', s.x, s.y - 18);
  T.path(s.x - 24, s.y - 9, s.x + 24, s.y - 9); T.path(s.x, s.y - 5, s.x, s.y - 9);
  for (let y = s.y - 32; y <= s.y - 9; y++) g.placePath(s.x, y);
  // paddock slots: [x0,y0,x1,y1]
  const slots = [[s.x - 14, s.y - 20, s.x - 2, s.y - 11], [s.x + 2, s.y - 20, s.x + 14, s.y - 11], [s.x - 14, s.y - 32, s.x - 2, s.y - 22], [s.x + 2, s.y - 32, s.x + 14, s.y - 22], [s.x - 26, s.y - 32, s.x - 16, s.y - 11], [s.x + 16, s.y - 32, s.x + 26, s.y - 11]];
  let used = 0;
  const openPaddock = (feed, wall) => {
    const [x0, y0, x1, y1] = slots[used++];
    T.rect(x0, y0, x1, y1, wall ? F_WALL : F_ELECTRIC);
    for (let y = y0 + 1; y < y0 + 4; y++) for (let x = x0 + 1; x < x0 + 5; x++) { const i = w.idx(x, y); if (w.terrain[i] === T_GRASS) w.terrain[i] = T_FOREST; }
    w.invalidate();
    g.placeBuilding(feed, Math.floor((x0 + x1) / 2), y1 - 1);
    w.computeRegions(); w.computePower(g);
    return [x0, y0, x1, y1];
  };
  const hatchIn = (p, sp, n = 1) => { for (let k = 0; k < n; k++) g.hatch(sp, randi(p[0] + 2, p[2] - 2), randi(p[1] + 2, p[3] - 3)); };
  const p1 = openPaddock('feeder_h');
  T.place('visitor', s.x + 3, s.y - 7); T.place('restaurant', s.x - 5, s.y - 7); T.place('restroom', s.x - 2, s.y - 7);
  T.place('viewing', s.x - 8, s.y - 8);
  hatchIn(p1, 'galli', 3); hatchIn(p1, 'para', 2);
  note('start');
  const plan = [
    { at: 150000, f: () => { T.place('ranger', s.x + 9, s.y - 7); T.place('maint', s.x - 13, s.y - 7); note('staff'); } },
    { at: 230000, f: () => { const p = openPaddock('feeder_h'); hatchIn(p, 'trike'); hatchIn(p, 'stego', 2); T.place('shop', s.x + 6, s.y - 7); note('paddock 2'); } },
    { at: 150000, f: () => { const p = openPaddock('feeder_c'); hatchIn(p, 'dilo', 1); T.place('viewing', s.x - 6, s.y - 21); T.place('vet', s.x + 13, s.y - 7); note('paddock 3 dilo'); } },
    { at: 130000, f: () => { T.place('helipad', s.x + 16, s.y - 6); T.place('backup', s.x - 18, s.y - 6); T.place('pylon', s.x + 14, s.y - 26); T.place('pylon', s.x - 14, s.y - 26); note('helipad+backup'); } },
    { at: 360000, f: () => { const p = openPaddock('feeder_c', true); hatchIn(p, 'raptor', 3); note('raptors'); }, stars: 2 },
    { at: 160000, f: () => { T.place('hotel', s.x - 20, s.y - 7); T.place('shelter', s.x + 20, s.y - 4); note('hotel'); } },
    { at: 300000, f: () => { const p = openPaddock('feeder_c', true); hatchIn(p, 'trex'); note('trex'); }, stars: 3 },
    { at: 260000, f: () => { hatchIn(slots[1], 'anky'); hatchIn(slots[0], 'brachio'); note('brachio+anky'); }, stars: 3 },
    { at: 350000, f: () => { const p = openPaddock('feeder_c', true); hatchIn(p, 'spino'); note('spino'); }, stars: 4 },
  ];
  const blog = []; g.on('log', (e) => { if (/BREACH|broke|died|destroyed|collapsed|trucked|airlift/i.test(e.msg)) blog.push('D' + g.day + ' ' + e.msg); });
  let pi = 0;
  const days = [];
  for (let d = 0; d < 40 && !g.gameOver; d++) {
    for (let k = 0; k < 4; k++) {
      T.run(15);
      // a sensible player sounds the alarm while predators are loose
      const threat = g.dinos.some((x) => x.loose && x.sp.danger >= 4 && x.sedatedT <= 0);
      if (threat !== g.alarm) g.alarm = threat;
    }
    while (pi < plan.length && g.money > plan[pi].at && g.stars >= (plan[pi].stars || 0)) { plan[pi].f(); pi++; }
    const h = g.history[g.history.length - 1];
    if (h) days.push(`D${h.day} $${Math.round(h.money / 1000)}k +${Math.round(h.income / 1000)}k -${Math.round(h.expense / 1000)}k g${h.guests} rep${Math.round(h.rep)} ★${h.stars} score${Math.round(g.score)} dinos${g.dinos.length}`);
  }
  const ev = g.events.history.map((e) => e.day + ':' + e.type).join(' ');
  return { blog: blog.slice(0, 40), days, out, ev, stats: g.stats, goal: g.goalIdx, over: g.gameOver, loose: g.looseCount };
})()
