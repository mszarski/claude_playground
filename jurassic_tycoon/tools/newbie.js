// Walk through the first goals the way a new player would, using only the UI.
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const path = require('path');
const SP = process.argv[2];
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  await page.goto('file://' + path.resolve(__dirname, '../index.html'));
  await page.waitForTimeout(400);
  await page.click('#btnNew'); await page.click('[data-diff=normal]');
  await page.waitForTimeout(300);
  const toScreen = async (tx, ty) => page.evaluate(([tx, ty]) => { const c = RENDER.cam; return [(tx * 16 + 8 - c.x) * c.zoom, (ty * 16 + 8 - c.y) * c.zoom]; }, [tx, ty]);
  // find open grass areas programmatically so clicks are valid, like a player would eyeball
  const plan = await page.evaluate(() => {
    const g = GAME, w = g.world, s = g.start;
    RENDER.cam.zoom = 2; UIX.centerOn(s.x, s.y - 9);
    const free = (x0, y0, ww, hh) => { for (let y = y0; y < y0 + hh; y++) for (let x = x0; x < x0 + ww; x++) { if (!w.canBuildAt(x, y, false) || w.path[w.idx(x, y)]) return false; } return true; };
    const find = (ww, hh, near) => { for (let r = 0; r < 25; r++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) { const x = near[0] + dx, y = near[1] + dy; if (free(x, y, ww, hh)) return [x, y]; } return null; };
    return { s, hatch: find(3, 2, [s.x + 6, s.y - 3]), power: find(3, 3, [s.x - 7, s.y - 8]), paddock: find(11, 8, [s.x - 5, s.y - 18]) };
  });
  const shot = async (n) => page.screenshot({ path: `${SP}/nb_${n}.png` });
  // Goal 1: hatchery via Show me
  await page.click('#goalBtn');
  let [x, y] = await toScreen(plan.hatch[0] + 1, plan.hatch[1]);
  await page.mouse.click(x, y); await page.waitForTimeout(1500);
  // Goal 2: power
  await page.click('#goalBtn');
  [x, y] = await toScreen(plan.power[0] + 1, plan.power[1] + 1);
  await page.mouse.move(x, y); await page.waitForTimeout(100); await shot('power_preview');
  await page.mouse.click(x, y); await page.waitForTimeout(1500);
  // Goal 3: paddock
  await page.click('#goalBtn');
  const p = plan.paddock;
  let [ax, ay] = await toScreen(p[0], p[1]); let [bx, by] = await toScreen(p[0] + 10, p[1] + 7);
  await page.mouse.move(ax, ay); await page.mouse.down(); await page.mouse.move(bx, by, { steps: 6 }); await shot('paddock_drag'); await page.mouse.up();
  await page.waitForTimeout(300);
  // feeder
  await page.click('.cat[data-cat="dino"]'); await page.click('#flyout .tool:nth-of-type(2)');
  [x, y] = await toScreen(p[0] + 5, p[1] + 4); await page.mouse.click(x, y);
  await page.waitForTimeout(1500);
  await shot('after_paddock');
  const st = await page.evaluate(() => ({ goal: GAME.goalIdx, money: GAME.money, pw: GAME.world.power, msgs: GAME.logs.slice(-5).map((l) => l.msg) }));
  console.log(JSON.stringify(st));
  if (errors.length) console.log(errors.join('\n'));
  await browser.close();
})();
