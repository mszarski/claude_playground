(() => {
  document.getElementById('btnSandbox').click();
  const g = window.GAME, w = g.world, s = g.start;
  T.clear(s.x - 18, s.y - 28, s.x + 18, s.y - 1);
  T.place('power', s.x - 11, s.y - 4);
  T.place('hatchery', s.x + 6, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3); T.place('pylon', s.x, s.y - 16);
  const px0 = s.x - 7, py0 = s.y - 22, px1 = s.x + 7, py1 = s.y - 9;
  T.rect(px0, py0, px1, py1);
  g.placeBuilding('feeder_c', s.x + 3, py1 - 2);
  T.path(s.x - 12, py1 + 1, s.x + 12, py1 + 1);
  T.path(s.x, s.y - 5, s.x, py1 + 1);
  T.place('visitor', s.x + 2, s.y - 7);
  T.place('restaurant', s.x - 5, s.y - 7);
  T.place('ranger', s.x + 10, py1 + 3);
  T.place('maint', s.x - 10, py1 + 3);
  T.place('helipad', s.x + 11, s.y - 7);
  T.place('shelter', s.x - 9, s.y - 7);
  w.computeRegions(); w.computePower(g);
  g.hatch('trex', s.x, s.y - 15);
  g.time = 9;
  T.run(25);
  const log = [];
  g.on('log', (e) => log.push(`[${g.day}d ${g.hour.toFixed(1)}h] ${e.msg}`));
  // knock out the grid and make rex angry
  g.events.trigger('outage');
  const rex = g.dinos[0]; rex.stress = 100; rex.rageT = 30;
  T.run(60);
  const snap = { guests: g.guests.length, deaths: g.stats.deaths, loose: g.looseCount, rex: rex ? [rex.state, rex.loose, Math.round(rex.sedatedT), !!rex.carried, rex.x.toFixed(1), rex.y.toFixed(1)] : null, staff: g.staff.map(s => [s.role, s.state, !!s.target]), heli: g.helis.map(h => h.state), money: Math.round(g.money) };
  T.run(60);
  snap.after = { guests: g.guests.length, deaths: g.stats.deaths, loose: g.looseCount, rex: [rex.state, rex.loose, Math.round(rex.sedatedT), !!rex.carried, rex.dead], heli: g.helis.map(h => h.state), broken: g.countBrokenFences() };
  snap.log = log.slice(-30);
  UIX.centerOn(rex.x, rex.y);
  return snap;
})()
