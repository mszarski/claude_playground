// RTS controls with the real mouse: box-select, right-click orders, the RESPOND button, engineer repair orders.
// Usage: node tools/rts.js [screenshotDir]
const { chromium } = require('./pw');
const path = require('path'), fs = require('fs');
const SP = process.argv[2] || require('os').tmpdir();
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
  const errs = []; p.on('pageerror', (e) => errs.push(e.message));
  await p.goto('file://' + path.resolve(__dirname, '..', 'index.html')); await p.waitForTimeout(300);
  await p.evaluate(fs.readFileSync(path.join(__dirname, 'helpers.js'), 'utf8'));
  const res = {};
  // a park with rangers, engineers and a loose raptor
  await p.evaluate(() => {
    startGame({ seed: 1337, difficulty: 'calm', size: 'small' });
    const g = GAME, s = g.start;
    const P = buildStarterPark(g, { staff: true });
    g.placeBuilding('ranger', s.x - 14, s.y - 7);
    g.world.computeRegions();
    spawnIn(g, 'raptor', P[1], 1);
    const d = g.dinos[0];
    d.x = s.x + 0.5; d.y = s.y - 9.5; // out on the boulevard
    g.updateLoose(); d.stress = 20;
    g.events.auto = false;
    UIX.r.cam.zoom = 2; UIX.centerOn(s.x, s.y - 12);
    for (let i = 0; i < 30; i++) g.update(1 / 30);
    window.RAP = d;
  });
  const toScreen = (wx, wy) => p.evaluate(([x, y]) => { const c = RENDER.cam; return [(x * TILE - c.x) * c.zoom, (y * TILE - c.y) * c.zoom]; }, [wx, wy]);
  // shift-drag a box over the whole view
  await p.keyboard.down('Shift');
  await p.mouse.move(80, 80); await p.mouse.down(); await p.mouse.move(700, 500); await p.mouse.move(1300, 840); await p.mouse.up();
  await p.keyboard.up('Shift');
  res.boxSelected = await p.evaluate(() => UIX.squad.map((u) => u.kind));
  // keep just the rangers and right-click the raptor
  await p.evaluate(() => UIX.selectSquad(UIX.squad.filter((u) => u.kind === 'ranger')));
  const [rx, ry] = await p.evaluate(() => [RAP.x, RAP.y - 0.3]);
  const [sx, sy] = await toScreen(rx, ry);
  await p.mouse.click(sx, sy, { button: 'right' });
  res.orders = await p.evaluate(() => UIX.squad.map((u) => u.order && u.order.type));
  await p.screenshot({ path: path.join(SP, 'rts-order.png') });
  res.sedated = await p.evaluate(() => { for (let i = 0; i < 30 * 40 && RAP.sedatedT <= 0; i++) GAME.update(1 / 30); return RAP.sedatedT > 0; });
  res.ordersAfter = await p.evaluate(() => { for (let i = 0; i < 30 * 2; i++) GAME.update(1 / 30); return GAME.staff.filter((s) => s.role === 'ranger').map((s) => s.order ? s.order.type : null); });
  // RESPOND: a second escape, one click
  res.respond = await p.evaluate(() => {
    const g = GAME, s = g.start;
    const d = g.dinos.find((x) => x !== RAP) || (spawnIn(g, 'dilo', [s.x - 7, s.y - 26, s.x + 7, s.y - 12], 1), g.dinos[g.dinos.length - 1]);
    d.x = s.x + 6.5; d.y = s.y - 9.5; g.updateLoose(); UIX.updateHUD(1);
    const btn = document.querySelector('[data-act="respond"]');
    if (!btn) return 'no button';
    btn.click();
    return { alarm: g.alarm, hunting: g.staff.filter((r) => r.order && r.order.type === 'hunt').length };
  });
  // engineer repair order on a broken fence far from them
  res.repair = await p.evaluate(() => {
    const g = GAME, w = g.world, s = g.start;
    g.alarm = false;
    for (const d of g.dinos) { d.sedatedT = 999; }
    const fx = s.x + 22, fy = s.y - 20;
    w.fence[w.idx(fx, fy)] = F_BROKEN; w.fenceHp[w.idx(fx, fy)] = 0; w.invalidate(fx, fy);
    const eng = g.staff.filter((x) => x.role === 'worker');
    UIX.selectSquad([eng[0]]);
    g.commandUnits(UIX.squad, { fence: { x: fx, y: fy }, x: fx, y: fy });
    for (let i = 0; i < 30 * 60 && w.fence[w.idx(fx, fy)] === F_BROKEN; i++) g.update(1 / 30);
    return { fixed: w.fence[w.idx(fx, fy)] !== F_BROKEN, order: eng[0].order ? eng[0].order.type : null };
  });
  res.errors = errs;
  console.log(JSON.stringify(res, null, 1));
  await b.close();
  const ok = res.boxSelected.includes('ranger') && res.orders.every((o) => o === 'hunt') && res.sedated && res.respond.alarm && res.repair.fixed && !errs.length;
  process.exit(ok ? 0 : 1);
})();
