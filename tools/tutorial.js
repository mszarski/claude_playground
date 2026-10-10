// Walk through every tutorial step using the game API, checking each step advances.
(() => {
  document.getElementById('btnTut').click();
  const g = GAME, w = g.world, s = g.start, tut = UIX.tutorial;
  const log = [];
  const tick = (sec) => { for (let i = 0; i < sec * 30; i++) { g.update(1 / 30); tut.update(1 / 30); if (tut.done) return; } };
  const at = () => tut.done ? 'done' : tut.i;
  T.clear(s.x - 16, s.y - 24, s.x + 16, s.y - 1);
  document.getElementById('tutNext').click(); log.push('next->' + at());
  T.place('power', s.x - 7, s.y - 6); tick(1); log.push('power->' + at());
  T.place('pylon', s.x - 1, s.y - 9); T.place('pylon', s.x - 2, s.y - 15); T.rect(s.x - 6, s.y - 18, s.x + 4, s.y - 10); w.computeRegions(); w.computePower(g); tick(1); log.push('paddock->' + at());
  g.placeBuilding('feeder_h', s.x, s.y - 12); tick(1); log.push('feeder->' + at());
  T.place('hatchery', s.x + 5, s.y - 5); tick(1); log.push('hatchery->' + at());
  g.hatch('galli', s.x - 2, s.y - 14); g.hatch('galli', s.x + 1, s.y - 14); tick(1); log.push('hatch->' + at());
  T.place('visitor', s.x + 3, s.y - 8); T.path(s.x - 5, s.y - 7, s.x + 2, s.y - 7); T.path(s.x, s.y - 5, s.x, s.y - 7);
  g.time = 9; tick(15); log.push('guests(' + g.guests.length + ')->' + at());
  T.place('ranger', s.x - 12, s.y - 9); tick(1); log.push('ranger->' + at());
  log.push('loose=' + g.looseCount + ' broken=' + g.countBrokenFences());
  UIX.toggleAlarm(); tick(1); log.push('alarm->' + at());
  tick(60);
  // repair any broken fence
  for (let i = 0; i < w.fence.length; i++) if (w.fence[i] === F_BROKEN) g.emergencyRepair(i % w.W, (i / w.W) | 0);
  tick(20); log.push('contain->' + at() + ' loose=' + g.looseCount + ' broken=' + g.countBrokenFences());
  UIX.toggleAlarm(); tick(1); log.push('alarmoff->' + at());
  const nb = document.getElementById('tutNext'); if (nb) nb.click(); log.push('final->' + at());
  return log;
})()
