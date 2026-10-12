(() => {
  document.getElementById('btnSandbox').click();
  const g = window.GAME, w = g.world, s = g.start;
  T.clear(s.x - 20, s.y - 30, s.x + 20, s.y - 1);
  T.place('power', s.x - 11, s.y - 4);
  T.place('hatchery', s.x + 7, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3); T.place('pylon', s.x, s.y - 16);
  const px0 = s.x - 7, py0 = s.y - 22, px1 = s.x + 7, py1 = s.y - 11;
  T.rect(px0, py0, px1, py1);
  g.placeBuilding('feeder_h', s.x + 3, py1 - 2);
  T.path(s.x - 12, s.y - 6, s.x + 12, s.y - 6); T.path(s.x, s.y - 5, s.x, s.y - 6);
  const st = g.placeBuilding('tour', s.x - 2, s.y - 8);
  // loop track around paddock
  for (let x = px0 - 2; x <= px1 + 2; x++) { g.placeTrack(x, py0 - 2); g.placeTrack(x, py1 + 2); }
  for (let y = py0 - 2; y <= py1 + 2; y++) { g.placeTrack(px0 - 2, y); g.placeTrack(px1 + 2, y); }
  T.place('restaurant', s.x - 6, s.y - 8);
  w.computeRegions(); w.computePower(g);
  ['galli', 'galli', 'trike', 'brachio'].forEach((sp, i) => g.hatch(sp, s.x - 3 + i * 2, s.y - 16));
  g.time = 9;
  T.run(40);
  const states = {}; for (const gu of g.guests) { const k = gu.state + ':' + (gu.goal ? (gu.goal === 'gate' ? 'gate' : gu.goal.type) : '-'); states[k] = (states[k] || 0) + 1; }
  const info = { states, pw: st.powered, acc2: w.accessTiles(st).length, station: !!st, access: st && w.trackAccess(st).length, jeeps: g.jeeps.map(j => [j.state, j.riders.length]), queue: st.queue.length, rev: st.revenue || 0 };
  // now the rex
  const loose = new Dino(g, 'trex', px0 - 3, py1 + 3); loose.loose = true; g.dinos.push(loose); w.regionsDirty = true;
  T.run(25);
  info.after = { jeeps: g.jeeps.map(j => [j.state, j.riders.length, j.wrecked > 0]), deaths: g.stats.deaths };
  RENDER.cam.zoom = 2; UIX.centerOn(s.x, s.y - 14);
  return info;
})()
