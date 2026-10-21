// Every InGen call and every choice: the card renders, the choice applies, and its contract resolves.
(() => {
  const out = [];
  const setup = () => {
    startGame({ seed: window.TEST_SEED || 1337, difficulty: 'normal', size: 'small' });
    const g = window.GAME;
    const P = buildStarterPark(g, { staff: true });
    spawnIn(g, 'para', P[0], 3); spawnIn(g, 'raptor', P[1], 2); spawnIn(g, 'galli', P[2], 2);
    g.placeBuilding('helipad', g.start.x + 12, g.start.y - 4);
    g.world.computeRegions(); g.updateLoose(); g.updateComfort();
    g.time = 24 * 4 + 9; g.lastDay = g.day; g.stars = 3; g.stats.born = 1;
    for (let i = 0; i < 30; i++) g.guests.push(new Guest(g, g.world.accessTiles(g.buildingsOfType('gate')[0])[0]));
    return g;
  };
  for (const [id, def] of Object.entries(DECISIONS)) {
    for (let ci = 0; ci < def.choices.length; ci++) {
      const g = setup();
      const params = def.prep(g);
      if (!params) { out.push(`${id}: skipped (not eligible)`); break; }
      g.pendingDecision = { id, params };
      UIX.decisionShown = null;
      UIX.updateHUD(1);
      const btn = document.querySelector(`[data-ch="${ci}"]`);
      if (!btn) throw new Error(`${id}: card did not render`);
      const money = g.money;
      btn.click();
      if (g.pendingDecision) throw new Error(`${id}: choice ${ci} did not resolve`);
      // run past every deadline
      for (let k = 0; k < 30; k++) g.update(1 / 30);
      const contracts = (g.contracts || []).length;
      g.time += 24 * 11; g.lastDay = g.day;
      for (let k = 0; k < 60; k++) g.update(1 / 30);
      if ((g.contracts || []).length) throw new Error(`${id}: contract never resolved`);
      out.push(`${id}#${ci}: Δ$${Math.round((g.money - money) / 1000)}k contracts:${contracts}`);
    }
  }
  // the schedule itself offers calls on its own
  const g = setup();
  g.nextDecisionDay = 0; g.pendingDecision = null;
  g.maybeOfferDecision();
  out.push('scheduled: ' + (g.pendingDecision ? g.pendingDecision.id : 'none'));
  if (!g.pendingDecision) throw new Error('no call was scheduled');
  // and a save keeps deals
  g.resolveDecision(0); g.addMod('ad_blitz', 3);
  const g2 = loadGame(serialize(g));
  if (!g2.mods.some((m) => m.id === 'ad_blitz')) throw new Error('mods lost on load');
  UIX.closeModal();
  return out;
})()
