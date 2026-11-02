(() => {
  document.getElementById('btnSandbox').click();
  const g = GAME, w = g.world, s = g.start;
  T.clear(s.x - 16, s.y - 26, s.x + 16, s.y - 1);
  T.place('power', s.x - 11, s.y - 4); T.place('hatchery', s.x + 7, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3);
  T.path(s.x - 12, s.y - 7, s.x + 12, s.y - 7); T.path(s.x, s.y - 5, s.x, s.y - 7);
  const av = g.placeBuilding('aviary', s.x - 2, s.y - 13);
  T.place('ranger', s.x + 6, s.y - 10);
  w.computeRegions(); w.computePower(g);
  const errs = [];
  for (let i = 0; i < 4; i++) { const e = g.canHatchAt('ptera', av.x + 2, av.y + 2); if (e) errs.push(e); g.hatch('ptera', av.x + 2, av.y + 2); }
  g.time = 9; T.run(15);
  const before = g.pteros.map(p => [p.loose, p.alt.toFixed(1)]);
  RENDER.cam.zoom = 3; UIX.centerOn(s.x, s.y - 10);
  for (let i = 0; i < 5; i++) RENDER.draw(1/30, UIX);
  if (window.STOP) return;
  // break the dome
  g.damageBuilding(av, av.maxHp * 0.7, null);
  T.run(30);
  const after = g.pteros.map(p => [p.loose, Math.round(p.sedatedT), p.kills]);
  return { av: !!av, errs, before, after, deaths: g.stats.deaths, loose: g.looseCount, log: g.logs.slice(-8).map(l => l.msg) };
})()
