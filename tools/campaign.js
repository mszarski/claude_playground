// Campaign: every chapter boots with sane progress text, a forced win unlocks the next chapter and its perk applies.
(() => {
  try { localStorage.removeItem('jt_campaign'); } catch (e) { /* ignore */ }
  const out = [];
  for (const id of CAMPAIGN_ORDER) {
    startChapter(id);
    const g = window.GAME, sc = SCENARIOS[id];
    UIX.closeModal();
    if (g.mapSize !== sc.size) throw new Error(`${id}: wrong island size ${g.mapSize}`);
    for (let i = 0; i < 30 * 10; i++) g.update(1 / 30);
    if (g.scenarioDone) throw new Error(`${id}: ended immediately (${g.scenarioDone})`);
    const prog = sc.progress(g);
    if (!prog || /undefined|NaN/.test(prog)) throw new Error(`${id}: bad progress "${prog}"`);
    out.push(`${id} ${g.mapSize} $${Math.round(g.money / 1000)}k research:${[...g.research.done].join(',') || '-'} | ${prog}`);
    // win it: the end screen offers the next chapter
    g.emit('scenario', 'win');
    const next = document.querySelector('[data-m="next"]');
    const nextId = CAMPAIGN_ORDER[CAMPAIGN_ORDER.indexOf(id) + 1];
    if (nextId && !next) throw new Error(`${id}: no next-chapter button`);
    if (nextId && !campaignUnlocked(nextId)) throw new Error(`${nextId} not unlocked`);
    UIX.closeModal();
    // and a save keeps the chapter going
    const g2 = loadGame(serialize(g));
    if (g2.scenario !== id) throw new Error(`${id}: lost on load`);
  }
  // perks from all four earlier chapters reach the finale
  startChapter('c5');
  UIX.closeModal();
  const g = window.GAME;
  if (!['rangers', 'fence2', 'darts', 'acu'].every((r) => g.research.has(r))) throw new Error('perks missing in c5');
  if (g.money < 900000) throw new Error('investor perk missing');
  showCampaignMenu();
  const locked = document.querySelectorAll('#scenMenu [data-camp][disabled]').length;
  document.getElementById('scenBack').click();
  out.push('locked after all wins: ' + locked);
  try { localStorage.removeItem('jt_campaign'); } catch (e) { /* ignore */ }
  return out;
})()
