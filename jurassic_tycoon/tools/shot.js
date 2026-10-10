// Usage: node tools/shot.js <url-or-file> <out.png> [w] [h] [waitMs] [evalScript]
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const [, , target, out, w = 1400, h = 1000, wait = 500, evalJs] = process.argv;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: +w, height: +h } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message + '\n' + e.stack));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push('CONSOLE ' + m.type() + ': ' + m.text()); });
  const url = target.startsWith('http') ? target : 'file://' + require('path').resolve(target);
  await page.goto(url);
  await page.waitForTimeout(+wait);
  if (evalJs) { const r = await page.evaluate(evalJs); if (r !== undefined) console.log('EVAL:', JSON.stringify(r)); }
  await page.waitForTimeout(200);
  await page.screenshot({ path: out });
  if (errors.length) console.log(errors.join('\n'));
  await browser.close();
})();
