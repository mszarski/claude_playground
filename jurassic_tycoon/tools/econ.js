(() => {
  startGame({ difficulty: 'normal' });
  const g = window.GAME, w = g.world, s = g.start;
  g.events.auto = false;
  T.clear(s.x - 18, s.y - 28, s.x + 18, s.y - 1);
  const log = [];
  const act = (m) => log.push(`D${g.day} $${Math.round(g.money / 1000)}k: ${m}`);
  T.place('power', s.x - 11, s.y - 4);
  T.place('hatchery', s.x + 6, s.y - 3);
  T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3); T.place('pylon', s.x, s.y - 16);
  const px0 = s.x - 7, py0 = s.y - 20, px1 = s.x + 7, py1 = s.y - 9;
  T.rect(px0, py0, px1, py1);
  for (let y = py0 + 1; y < py0 + 5; y++) for (let x = px0 + 1; x < px0 + 6; x++) w.terrain[w.idx(x, y)] = T_FOREST;
  w.invalidate();
  g.placeBuilding('feeder_h', s.x + 3, py1 - 2);
  T.path(s.x - 8, py1 + 1, s.x + 8, py1 + 1);
  T.path(s.x, s.y - 5, s.x, py1 + 1);
  T.place('visitor', s.x + 2, s.y - 7);
  T.place('restaurant', s.x - 4, s.y - 7);
  T.place('restroom', s.x - 1, s.y - 7);
  T.place('viewing', s.x - 3, py1 + 2);
  w.computeRegions(); w.computePower(g);
  g.hatch('galli', s.x - 2, s.y - 15); g.hatch('galli', s.x + 1, s.y - 14); g.hatch('galli', s.x + 2, s.y - 16);
  g.hatch('para', s.x - 3, s.y - 12); g.hatch('para', s.x - 4, s.y - 13);
  act('initial build');
  const days = [];
  let built2 = false;
  for (let d = 0; d < 20; d++) {
    T.run(60);
    const h = g.history[g.history.length - 1];
    days.push(`D${h.day}: $${Math.round(h.money / 1000)}k in ${Math.round(h.income / 1000)}k out ${Math.round(h.expense / 1000)}k guests ${h.guests} rep ${Math.round(h.rep)} stars ${h.stars} att ${Math.round(g.attraction)} desired ${g.desiredGuests()} tk ${Math.round(h.ledger.tickets / 1000)}k shop ${Math.round(h.ledger.shops / 1000)}k food ${Math.round(h.ledger.food / 1000)}k up ${Math.round(h.ledger.upkeep / 1000)}k`);
    if (!built2 && g.money > 200000) {
      built2 = true;
      T.rect(s.x + 9, s.y - 26, s.x + 17, s.y - 10);
      g.placeBuilding('feeder_h', s.x + 13, s.y - 12);
      T.path(s.x + 8, py1 + 1, s.x + 17, py1 + 1);
      T.place('pylon', s.x + 12, s.y - 17);
      w.computeRegions(); w.computePower(g);
      g.hatch('trike', s.x + 13, s.y - 20);
      g.hatch('stego', s.x + 11, s.y - 22);
      T.place('shop', s.x + 10, s.y - 7);
      act('second paddock');
    }
  }
  const hap = g.happySamples ? Math.round(g.happySamples.reduce((a, b) => a + b, 0) / g.happySamples.length) : null;
  return { days, log, hap, dinos: g.dinos.map(d => d.species + ':' + Math.round(d.comfort) + '/' + Math.round(d.stress)), goals: g.goalIdx };
})()
