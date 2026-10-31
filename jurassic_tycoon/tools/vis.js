(() => {
  document.getElementById('btnSandbox').click();
  const g = window.GAME, w = g.world, s = g.start;
  T.clear(s.x - 18, s.y - 28, s.x + 18, s.y - 1);
  T.place('power', s.x - 11, s.y - 4);
  T.place('hatchery', s.x + 6, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3);
  const px0 = s.x - 7, py0 = s.y - 20, px1 = s.x + 7, py1 = s.y - 9;
  T.rect(px0, py0, px1, py1);
  T.rect(s.x + 9, s.y - 22, s.x + 16, s.y - 10, F_WALL);
  g.placeBuilding('feeder_c', s.x + 3, py1 - 2);
  T.path(s.x - 12, py1 + 1, s.x + 12, py1 + 1);
  T.path(s.x, s.y - 5, s.x, py1 + 1);
  T.place('visitor', s.x + 2, s.y - 7); T.place('restaurant', s.x - 5, s.y - 7); T.place('shop', s.x - 8, s.y - 7);
  for (let x = s.x - 12; x <= s.x + 12; x += 4) g.placeBuilding('lamp', x, py1 + 2);
  T.place('ranger', s.x + 10, s.y - 7);
  w.computeRegions(); w.computePower(g);
  g.hatch('raptor', s.x - 3, s.y - 15); g.hatch('raptor', s.x + 2, s.y - 13); g.hatch('raptor', s.x, s.y - 17);
  g.hatch('trex', s.x + 12, s.y - 16);
  g.time = 10; T.run(15);
  g.time = 22;
  T.run(1);
  RENDER.cam.zoom = 2; UIX.centerOn(s.x, s.y - 12);
  return { dinos: g.dinos.length };
})()
