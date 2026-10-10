// Drive the real UI with mouse events.
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const path = require('path');
const SP = process.argv[2];
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message + '\n' + e.stack));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });
  await page.goto('file://' + path.resolve(__dirname, '../index.html'));
  await page.waitForTimeout(500);
  await page.click('#btnNew');
  await page.waitForTimeout(300);
  // helper: tile -> screen
  const toScreen = async (tx, ty) => page.evaluate(([tx, ty]) => { const c = RENDER.cam; return [(tx * 16 + 8 - c.x) * c.zoom, (ty * 16 + 8 - c.y) * c.zoom]; }, [tx, ty]);
  const start = await page.evaluate(() => { const g = GAME, s = g.start; T = null; return s; });
  // clear test area for determinism
  await page.evaluate(() => { const w = GAME.world, s = GAME.start; for (let y = s.y - 16; y < s.y; y++) for (let x = s.x - 12; x < s.x + 12; x++) { const i = w.idx(x, y); if (w.terrain[i] !== T_WATER && w.terrain[i] !== T_DEEP) w.terrain[i] = T_GRASS; } w.invalidate(); RENDER.cam.zoom = 2; UIX.centerOn(s.x, s.y - 8); });
  // Build > Paddock
  await page.click('.cat[data-cat="build"]');
  await page.waitForTimeout(100);
  await page.screenshot({ path: SP + '/ui_flyout.png' });
  await page.click('#flyout .tool:nth-of-type(2)'); // fence? list order: path, fence, paddock...
  const toolName = await page.evaluate(() => UIX.tool);
  await page.keyboard.press('r');
  const tool2 = await page.evaluate(() => UIX.tool);
  let [ax, ay] = await toScreen(start.x - 6, start.y - 15);
  let [bx, by] = await toScreen(start.x + 4, start.y - 8);
  await page.mouse.move(ax, ay); await page.mouse.down(); await page.mouse.move((ax + bx) / 2, (ay + by) / 2, { steps: 4 }); await page.mouse.move(bx, by, { steps: 4 });
  await page.screenshot({ path: SP + '/ui_drag.png' });
  await page.mouse.up();
  const fences = await page.evaluate(() => { let n = 0; for (const f of GAME.world.fence) if (f) n++; return n; });
  // Power plant via flyout
  await page.click('.cat[data-cat="infra"]');
  await page.click('#flyout .tool:nth-of-type(1)');
  [ax, ay] = await toScreen(start.x - 9, start.y - 4);
  await page.mouse.move(ax, ay); await page.mouse.click(ax, ay);
  // hatchery
  await page.click('.cat[data-cat="dino"]');
  await page.click('#flyout .tool:nth-of-type(1)');
  [ax, ay] = await toScreen(start.x + 7, start.y - 4);
  await page.mouse.click(ax, ay);
  // feeder
  await page.click('#flyout .tool:nth-of-type(2)');
  [ax, ay] = await toScreen(start.x - 1, start.y - 12);
  await page.mouse.click(ax, ay);
  // species picker -> galli
  await page.keyboard.press('h');
  await page.waitForTimeout(100);
  await page.screenshot({ path: SP + '/ui_species.png' });
  await page.click('.card[data-sp="galli"]');
  [ax, ay] = await toScreen(start.x - 3, start.y - 11);
  await page.mouse.click(ax, ay);
  await page.waitForTimeout(100);
  // inspect a building
  await page.keyboard.press('q');
  const bld = await page.evaluate(() => { const b = GAME.buildingsOfType('hatchery')[0]; return b ? [b.x, b.y] : null; });
  if (bld) { [ax, ay] = await toScreen(bld[0] + 1, bld[1] + 1); await page.mouse.click(ax, ay); }
  await page.waitForTimeout(600);
  await page.screenshot({ path: SP + '/ui_inspect.png' });
  const state = await page.evaluate(() => ({ tool: UIX.tool, sel: UIX.selected && (UIX.selected.type || UIX.selected.kind), money: GAME.money, eggs: GAME.eggs.length, dinos: GAME.dinos.length, blds: [...GAME.world.buildings.values()].map(b => b.type), goal: GAME.goalIdx }));
  // wheel zoom
  await page.mouse.move(640, 400); await page.mouse.wheel(0, -200); await page.waitForTimeout(100);
  const zoom = await page.evaluate(() => RENDER.cam.zoom);
  // finance modal and disasters modal
  await page.click('#finBtn'); await page.waitForTimeout(100); await page.screenshot({ path: SP + '/ui_fin.png' });
  await page.keyboard.press('Escape');
  await page.click('#disBtn'); await page.waitForTimeout(100); await page.click('[data-dis="storm"]');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: SP + '/ui_after.png' });
  console.log(JSON.stringify({ toolName, tool2, fences, state, zoom }));
  if (errors.length) console.log(errors.join('\n'));
  await browser.close();
})();
