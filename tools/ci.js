// CI test suite: boots every mode, runs the tutorial, fuzzes, and checks save/load.
// Usage: node tools/ci.js   (exit code 1 on any failure)
const { chromium } = require('./pw');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const helpers = fs.readFileSync(path.join(__dirname, 'helpers.js'), 'utf8');
const SEED = 1337;

const TESTS = [
  {
    name: 'boot all modes',
    run: `(() => {
      const modes = [{ difficulty: 'calm' }, { difficulty: 'easy' }, { difficulty: 'normal' }, { difficulty: 'chaos' }, { sandbox: true }, { tutorial: true, difficulty: 'easy' }]
        .concat(Object.keys(SCENARIOS).map((k) => ({ scenario: k, difficulty: 'normal' })));
      const out = [];
      for (const m of modes) {
        startGame(Object.assign({ seed: ${SEED} }, m));
        for (let i = 0; i < 30 * 20; i++) GAME.update(1 / 30);
        RENDER.draw(1 / 30, UIX);
        out.push(JSON.stringify(m) + ' day ' + GAME.day);
      }
      if (UIX.tutorial) UIX.tutorial.finish();
      return out;
    })()`,
    check: (r) => Array.isArray(r) && r.length >= 12,
  },
  { name: 'tutorial completes', file: 'tutorial.js', check: (r) => r[r.length - 1] === 'final->done' },
  { name: 'InGen calls: every choice', file: 'decisions.js', check: (r) => Array.isArray(r) && r.some((x) => x.startsWith('scheduled: ')) },
  { name: 'paddock blueprints and quick actions', file: 'busywork.js', check: (r) => r && r.ok },
  { name: 'attractions: gyrospheres, arena, feeding', file: 'attractions.js', check: (r) => r && r.ok },
  { name: 'fuzz + save/load round trip', file: 'fuzz.js', pre: 'window.FUZZ_RUNS = 1;', check: (r) => Array.isArray(r) && r.length >= 1 },
  {
    name: 'modals and panels render',
    run: `(() => {
      startGame({ seed: ${SEED}, difficulty: 'normal' });
      for (const f of ['showFinances', 'showDisasters', 'showMenu', 'showHelp', 'showSpeciesPicker', 'showRoster', 'showResearch', 'showAudio']) { UIX[f](); UIX.closeModal(); }
      return 'ok';
    })()`,
    check: (r) => r === 'ok',
  },
];

(async () => {
  const browser = await chromium.launch();
  let failed = 0;
  for (const t of TESTS) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => { if (m.type() === 'error' && !/fonts\.g/.test(m.text())) errors.push(m.text()); });
    await page.goto('file://' + path.join(ROOT, 'index.html'));
    await page.waitForTimeout(300);
    await page.evaluate(`window.TEST_SEED = ${SEED};`);
    if (t.pre) await page.evaluate(t.pre);
    const t0 = Date.now();
    let result, err = null;
    try {
      const code = helpers + '\n' + (t.file ? fs.readFileSync(path.join(__dirname, t.file), 'utf8') : t.run);
      result = await page.evaluate(code);
    } catch (e) { err = e.message.split('\n')[0]; }
    const ok = !err && errors.length === 0 && t.check(result);
    if (!ok) failed++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${t.name}  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    if (!ok) console.log('   ', err || errors.join('\n    ') || JSON.stringify(result).slice(0, 400));
    await page.close();
  }
  await browser.close();
  console.log(failed ? `${failed} test(s) failed` : 'All tests passed');
  process.exit(failed ? 1 : 0);
})();
