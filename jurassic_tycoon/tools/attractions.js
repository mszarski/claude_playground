// Gyrospheres carry guests through a paddock, the raptor arena runs shows (and incidents), feeding shows happen at noon.
(() => {
  startGame({ seed: window.TEST_SEED || 1337, difficulty: 'calm', size: 'small' });
  const g = window.GAME, w = g.world, s = g.start;
  const P = buildStarterPark(g, { staff: true });
  g.sandbox = true; g.money = 5e6;
  // herbivores in P[0], raptors in P[1]
  spawnIn(g, 'para', P[0], 3); spawnIn(g, 'raptor', P[1], 3);
  w.computeRegions(); g.updateLoose();
  const out = {};
  // gyro station against the south fence of P[0], on the boulevard side
  const [x0, y0, x1, y1] = P[0];
  let st = null;
  for (let x = x0 + 1; x < x1 - 3 && !st; x++) st = g.placeBuilding('gyro', x, y1 + 1);
  out.gyroPlaced = !!st;
  out.gyroAccess = st ? g.gyroAccess(st).length : 0;
  // arena and grandstand near P[1]
  const [ax0, ay0, ax1, ay1] = P[1];
  // arena just north of the raptor paddock, raptors gathered at that end
  T.clear(ax0, ay0 - 5, ax1, ay0 - 1);
  let ar = null; for (let x = ax0 + 2; x < ax1 - 4 && !ar; x++) ar = g.placeBuilding('arena', x, ay0 - 4);
  for (const d of g.dinos.filter((d) => d.species === 'raptor')) { d.x = ax0 + 6.5 + rand() * 3; d.y = ay0 + 2.5; }
  if (ar) for (let y = ay0 - 5; y <= s.y - 9; y++) g.placePath(ax0 - 1, y);
  if (ar) for (let x = ax0 - 1; x <= ar.x; x++) g.placePath(x, ay0 - 5);
  if (ar) { T.clear(ar.x + 4, ar.y - 4, ar.x + 12, ar.y + 3); T.place('power', ar.x + 7, ar.y); }
  let fs = null; for (let x = ax1 - 3; x > ax0 && !fs; x--) fs = g.placeBuilding('feedshow', x, ay1 + 1);
  out.arena = !!ar; out.feed = !!fs;
  w.computeRegions(); w.computePower(g);
  // paths to all of them
  for (let x = x0 - 2; x <= ax1 + 2; x++) g.placePath(x, s.y - 9);
  w.computeRegions();
  // guests: stand some near the arena, queue some for the gyros
  const gate = g.buildingsOfType('gate')[0];
  for (let i = 0; i < 40; i++) g.guests.push(new Guest(g, w.accessTiles(gate)[0]));
  g.time = 24 * 2 + 11.5; g.lastDay = g.day;
  if (st) for (const gu of g.guests.slice(0, 6)) { gu.goal = st; gu.state = 'queue'; gu.hidden = true; gu.insideT = 99; st.queue.push(gu); }
  if (ar) for (const gu of g.guests.slice(10, 30)) { gu.x = ar.x + 2; gu.y = ar.y - 1; }
  if (ar) ar.showT = 0.1;
  w.computePower(g);
  const money0 = g.money;
  let rode = 0, shows = 0;
  for (let i = 0; i < 30 * 12; i++) {
    g.update(1 / 30);
    rode = Math.max(rode, g.gyros.filter((gy) => gy.riders.length && gy.state !== 'parked').length);
    if (ar && ar.showing > 3.9) shows++;
    if (ar && i % 30 === 0) for (const d of g.dinos.filter((d) => d.species === 'raptor')) { d.x = ax0 + 6.5; d.y = ay0 + 2.5; }
  }
  RENDER.draw(1 / 30, UIX);
  out.gyros = g.gyros.length; out.rodeAtOnce = rode;
  out.arenaDbg = ar ? { powered: ar.powered, noR: ar.noRaptors, showT: ar.showT, hour: g.hour } : null;
  out.arenaShows = shows; out.arenaRevenue = ar ? Math.round(ar.revenue || 0) : 0;
  out.feedRevenue = fs ? Math.round(fs.revenue || 0) : 0;
  out.feedNote = fs ? (fs.noCarnivore ? 'no carnivore' : 'ok') : 'none';
  // force an arena incident and check it resolves without errors
  if (ar) g.arenaIncident(ar, g.dinos.filter((d) => d.species === 'raptor'));
  for (let i = 0; i < 30 * 3; i++) g.update(1 / 30);
  out.earned = Math.round(g.money - money0);
  out.ok = out.gyroPlaced && out.gyroAccess > 0 && out.rodeAtOnce >= 1 && out.arenaShows >= 1 && out.arenaRevenue > 0;
  return out;
})()
