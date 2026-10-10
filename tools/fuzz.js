// Random-action fuzzer: throws on exceptions or invalid state.
(() => {
  const results = [];
  for (let run = 0; run < 3; run++) {
    startGame({ difficulty: 'chaos', sandbox: true });
    let g = GAME; const w = g.world, s = g.start;
    const btypes = Object.keys(BUILDINGS).filter((b) => b !== 'gate');
    const sps = Object.keys(SPECIES);
    const dis = Object.keys(DISASTERS);
    let actions = 0;
    T.clear(s.x - 22, s.y - 32, s.x + 22, s.y - 1);
    T.place('power', s.x - 11, s.y - 4); T.place('hatchery', s.x + 7, s.y - 3); T.place('power', s.x + 10, s.y - 20);
    T.place('pylon', s.x - 3, s.y - 8); T.place('pylon', s.x + 3, s.y - 3); T.place('pylon', s.x, s.y - 18);
    T.rect(s.x - 18, s.y - 28, s.x - 4, s.y - 12); T.rect(s.x + 2, s.y - 28, s.x + 16, s.y - 12);
    T.path(s.x - 20, s.y - 10, s.x + 20, s.y - 10); T.path(s.x, s.y - 5, s.x, s.y - 10);
    T.place('visitor', s.x + 3, s.y - 8); T.place('ranger', s.x - 8, s.y - 8); T.place('maint', s.x + 9, s.y - 8); T.place('helipad', s.x - 16, s.y - 8);
    const av = T.place('aviary', s.x - 2, s.y - 34);
    w.computeRegions(); w.computePower(g);
    for (const sp of sps) { if (sp === 'ptera') { if (av) g.hatch('ptera', av.x + 2, av.y + 2); continue; } const left = chance(0.5); g.hatch(sp, left ? randi(s.x - 16, s.x - 6) : randi(s.x + 4, s.x + 14), randi(s.y - 26, s.y - 14)); }
    const check = () => {
      for (const d of g.creatures()) if (!isFinite(d.x) || !isFinite(d.y) || !isFinite(d.hp)) throw new Error('bad dino ' + d.species + ' ' + d.x + ',' + d.y);
      for (const gu of g.guests) if (!isFinite(gu.x) || !isFinite(gu.y)) throw new Error('bad guest state ' + gu.state);
      for (const st of g.staff) if (!isFinite(st.x) || !isFinite(st.y)) throw new Error('bad staff');
      for (const j of g.jeeps) if (!isFinite(j.x)) throw new Error('bad jeep');
      if (!isFinite(g.money)) throw new Error('bad money');
    };
    for (let step = 0; step < 400; step++) {
      const r = Math.random();
      const x = s.x + randi(-20, 20), y = s.y + randi(-30, 0);
      g = GAME;
      if (r < 0.25) g.placeBuilding(pick(btypes), x, y);
      else if (r < 0.4) { const x2 = x + randi(3, 12), y2 = y + randi(3, 10); T.rect(x, y, x2, y2, chance(0.7) ? F_ELECTRIC : F_WALL); }
      else if (r < 0.5) { for (let k = 0; k < 10; k++) g.placePath(x + k, y); }
      else if (r < 0.55) { for (let k = 0; k < 10; k++) g.placeTrack(x, y + k); }
      else if (r < 0.7) { w.regionsDirty || w.computeRegions(); const sp = pick(sps); g.hatch(sp, x, y); }
      else if (r < 0.78) g.demolishAt(x, y);
      else if (r < 0.82) g.events.trigger(pick(dis));
      else if (r < 0.84) { const d = pick(g.creatures()); if (d) { if (chance(0.5)) g.acuStrike(d); else { d.orderSedate = true; d.relocate = true; } } }
      else if (r < 0.86) g.emergencyRepair(x, y);
      else if (r < 0.87) {
        const G0 = GAME; const json = serialize(G0); const g2 = loadGame(json);
        const a = [G0.world.buildings.size, G0.creatures().length, Math.round(G0.money)], b = [g2.world.buildings.size, g2.creatures().length, Math.round(g2.money)];
        if (a.join() !== b.join()) throw new Error('save/load mismatch ' + a + ' vs ' + b);
        setActiveGame(g2);
      }
      else if (r < 0.88) g.alarm = !g.alarm;
      const G = GAME;
      for (let k = 0; k < 15; k++) G.update(1 / 30 * 4);
      actions++;
      if (step % 20 === 0) { RENDER.draw(1 / 30, UIX); UIX.renderInfo(true); }
      // in case of load, rebind
      if (GAME !== g) { /* continue on loaded game */ }
      try { check(); } catch (e) { throw new Error(`run ${run} step ${step}: ${e.message}`); }
      if (GAME.gameOver) break;
    }
    const G = GAME;
    results.push({ hatched: G.stats.hatched, escapes: G.stats.escapes, deaths: G.stats.deaths, dinoDeaths: G.stats.dinoDeaths, day: G.day, dinos: G.creatures().length, guests: G.guests.length, blds: G.world.buildings.size, money: Math.round(G.money), loose: G.looseCount, actions });
  }
  return results;
})()
