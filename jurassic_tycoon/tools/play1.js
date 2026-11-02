(() => {
  startGame({ difficulty: 'normal' });
  const g = window.GAME, w = g.world, s = g.start;
  T.clear(s.x - 14, s.y - 26, s.x + 14, s.y - 1);
  const res = {};
  res.power = !!T.place('power', s.x - 11, s.y - 4);
  res.hatch = !!T.place('hatchery', s.x + 5, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3);
  const px0 = s.x - 7, py0 = s.y - 20, px1 = s.x + 7, py1 = s.y - 9;
  res.fences = T.rect(px0, py0, px1, py1);
  // a bit of jungle in the paddock
  for (let y = py0 + 1; y < py0 + 5; y++) for (let x = px0 + 1; x < px0 + 6; x++) w.terrain[w.idx(x, y)] = T_FOREST;
  res.feeder = !!g.placeBuilding('feeder_h', s.x + 3, py1 - 2);
  T.path(s.x - 8, py1 + 1, s.x + 8, py1 + 1);
  T.path(s.x, s.y - 5, s.x, py1 + 1);
  res.visitor = !!T.place('visitor', s.x + 2, s.y - 7);
  res.rest = !!T.place('restaurant', s.x - 4, s.y - 7);
  res.view = !!T.place('viewing', s.x - 3, py1 + 2);
  w.computeRegions(); w.computePower(g);
  res.hatchErr = g.canHatchAt('galli', s.x, s.y - 15);
  g.hatch('galli', s.x - 2, s.y - 15); g.hatch('galli', s.x + 1, s.y - 14); g.hatch('galli', s.x + 2, s.y - 16);
  g.hatch('para', s.x - 3, s.y - 12); g.hatch('dilo', s.x + 4, s.y - 17);
  g.time = 8;
  T.run(60);
  res.money = Math.round(g.money); res.dinos = g.dinos.map(d => [d.species, Math.round(d.hunger), Math.round(d.stress), Math.round(d.comfort), d.state, d.loose]);
  res.guests = g.guests.length; res.loose = g.looseCount;
  res.pw = w.power; res.day = g.day; res.hour = g.hour.toFixed(1); res.goal = g.goalIdx; res.stars = g.stars; res.att = g.attraction;
  UIX.centerOn(s.x, s.y - 11);
  RENDER.cam.zoom = 3; UIX.centerOn(s.x, s.y - 11);
  return res;
})()
