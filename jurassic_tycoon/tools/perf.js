(() => {
  document.getElementById('btnSandbox').click();
  const g = window.GAME, w = g.world, s = g.start;
  T.clear(s.x - 22, s.y - 30, s.x + 22, s.y - 1);
  T.place('power', s.x - 11, s.y - 4); T.place('power', s.x + 11, s.y - 20);
  T.place('hatchery', s.x + 6, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3); T.place('pylon', s.x, s.y - 16);
  T.rect(s.x - 20, s.y - 28, s.x - 2, s.y - 10); T.rect(s.x + 2, s.y - 28, s.x + 20, s.y - 10);
  g.placeBuilding('feeder_h', s.x - 10, s.y - 12); g.placeBuilding('feeder_h', s.x + 10, s.y - 12);
  T.path(s.x - 21, s.y - 9, s.x + 21, s.y - 9); T.path(s.x, s.y - 5, s.x, s.y - 9);
  for (let y = s.y - 28; y <= s.y - 9; y++) g.placePath(s.x, y);
  T.place('visitor', s.x + 2, s.y - 7); T.place('restaurant', s.x - 5, s.y - 7); T.place('shop', s.x - 8, s.y - 7);
  T.place('restroom', s.x - 1, s.y - 7); T.place('hotel', s.x + 12, s.y - 7);
  w.computeRegions(); w.computePower(g);
  const sp = ['galli', 'para', 'trike', 'stego', 'anky', 'brachio'];
  for (let i = 0; i < 40; i++) { const left = i % 2 === 0; g.hatch(sp[i % sp.length], left ? s.x - 19 + (i % 15) : s.x + 3 + (i % 15), s.y - 27 + (i % 15)); }
  g.time = 9;
  T.run(10);
  // spawn lots of guests
  const gate = g.buildingsOfType('gate')[0];
  for (let i = 0; i < 400; i++) { const acc = w.accessTiles(gate); g.guests.push(new Guest(g, acc[0])); }
  const t0 = performance.now();
  T.run(20);
  const upd = (performance.now() - t0) / (20 * 30);
  const t1 = performance.now();
  for (let i = 0; i < 60; i++) RENDER.draw(1 / 60, UIX);
  const drw = (performance.now() - t1) / 60;
  const t2 = performance.now();
  for (let i = 0; i < 5; i++) RENDER.redrawTerrain();
  const ter = (performance.now() - t2) / 5;
  return { dinos: g.dinos.length, guests: g.guests.length, updateMs: upd.toFixed(2), drawMs: drw.toFixed(2), terrainMs: ter.toFixed(1) };
})()
