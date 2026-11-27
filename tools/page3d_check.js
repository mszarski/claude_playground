#!/usr/bin/env node
// Load 3-D scene pages (python/gpusonar/viz/page3d.py) in headless Chromium and fail on any console error or page
// error; optionally screenshot each (docs/tasks/VISUALIZATION.md, package V4). Opt-in: needs Node and the Playwright
// package with its Chromium (require('playwright')); nothing in the suite needs it (tests/python/test_scene_3d.py
// skips without it).
//
//   node tools/page3d_check.js [--shots DIR] [--scale 1] [--width 1280 --height 800] [--timeout 60000] PAGE.html|DIR ...
//
// A page passes when window.__ready becomes true and nothing was logged as an error. Software WebGL: Chromium's
// SwiftShader (--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader), so no GPU is needed.
// Prints one JSON line per page ({page, ok, errors, info, ms}) and exits 1 when any page failed.
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function args(argv) {
  const o = { shots: null, scale: 1, width: 1280, height: 800, timeout: 60000, pages: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--shots') o.shots = argv[++i];
    else if (a === '--scale') o.scale = +argv[++i];
    else if (a === '--width') o.width = +argv[++i];
    else if (a === '--height') o.height = +argv[++i];
    else if (a === '--timeout') o.timeout = +argv[++i];
    else if (fs.existsSync(a) && fs.statSync(a).isDirectory())
      o.pages.push(...fs.readdirSync(a).filter(f => f.endsWith('.html') && f !== 'index.html').sort().map(f => path.join(a, f)));
    else o.pages.push(a);
  }
  return o;
}

(async () => {
  const o = args(process.argv.slice(2));
  if (o.shots) fs.mkdirSync(o.shots, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  let failed = 0;
  for (const p of o.pages) {
    const t0 = Date.now();
    const ctx = await browser.newContext({ viewport: { width: o.width, height: o.height }, deviceScaleFactor: o.scale });
    const page = await ctx.newPage();
    const errors = [];
    page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
    page.on('pageerror', e => errors.push('pageerror: ' + (e && e.message)));
    page.on('requestfailed', r => errors.push('request failed: ' + r.url() + ' ' + ((r.failure() || {}).errorText || '')));
    let info = null;
    try {
      await page.goto('file://' + path.resolve(p), { waitUntil: 'load', timeout: o.timeout });
      await page.waitForFunction(() => window.__ready === true, null, { timeout: o.timeout });
      await page.waitForTimeout(200);
      info = await page.evaluate(() => window.__page || null);
      if (o.shots) await page.screenshot({ path: path.join(o.shots, path.basename(p).replace(/\.html$/, '') + '.png') });
    } catch (e) {
      errors.push('load: ' + (e && e.message ? e.message.split('\n')[0] : String(e)));
    }
    await ctx.close();
    const ok = errors.length === 0;
    if (!ok) failed++;
    console.log(JSON.stringify({ page: p, ok, errors, info, ms: Date.now() - t0 }));
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
