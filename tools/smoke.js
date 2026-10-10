// Headless smoke test: boot the game, build a park via the API, run the sim, screenshot.
const path = require('path');
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const out = process.argv[2] || 'shot.png';
  const script = process.argv[3] ? require('fs').readFileSync(path.resolve(__dirname, 'helpers.js'), 'utf8') + '\n' + require('fs').readFileSync(process.argv[3], 'utf8') : '';
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message + '\n' + e.stack));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); else if (m.type() === 'log') console.log('LOG:', m.text()); });
  await page.goto('file://' + path.resolve(__dirname, '../index.html'));
  await page.waitForTimeout(600);
  if (script) {
    const r = await page.evaluate(script);
    if (r !== undefined) console.log('RESULT:', JSON.stringify(r, null, 1));
  }
  await page.waitForTimeout(400);
  await page.screenshot({ path: out });
  if (errors.length) console.log(errors.join('\n'));
  await browser.close();
})();
