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
// Prints one JSON line per page ({page, ok, errors, info, chain, ms}) and exits 1 when any page failed.
// A page with the live chain's view (V6, viz/chain.py) also fails when, at any of its pings, the detections or tracks the
// chain layer drew differ in number from the report line's (`chain`: per ping {k, drawn_detections, report_detections,
// drawn_tracks, report_tracks, ellipses, gates, map}); `--chain-shot-ping K` screenshots it at ping K (default the last),
// `--zoom-chain` first zooms the view onto the chain's detections and tracks.
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function args(argv) {
  const o = { shots: null, scale: 1, width: 1280, height: 800, timeout: 60000, pages: [], chainPing: null, zoomChain: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--shots') o.shots = argv[++i];
    else if (a === '--scale') o.scale = +argv[++i];
    else if (a === '--width') o.width = +argv[++i];
    else if (a === '--height') o.height = +argv[++i];
    else if (a === '--timeout') o.timeout = +argv[++i];
    else if (a === '--chain-shot-ping') o.chainPing = +argv[++i];
    else if (a === '--zoom-chain') o.zoomChain = true;
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
    let info = null, chain = null;
    try {
      await page.goto('file://' + path.resolve(p), { waitUntil: 'load', timeout: o.timeout });
      await page.waitForFunction(() => window.__ready === true, null, { timeout: o.timeout });
      await page.waitForTimeout(200);
      info = await page.evaluate(() => window.__page || null);
      chain = await page.evaluate(() => {
        if (!window.__chain) return null;
        const out = []; for (let k = 0; k < window.__chain.pings; k++) out.push(window.__chain.set(k));
        return out; });
      if (chain) {
        for (const c of chain) {
          if (c.drawn_detections !== c.report_detections) errors.push(`chain ping ${c.k}: ${c.drawn_detections} detections drawn, ${c.report_detections} in the report line`);
          if (c.drawn_tracks !== c.report_tracks) errors.push(`chain ping ${c.k}: ${c.drawn_tracks} tracks drawn, ${c.report_tracks} in the report line`);
        }
        await page.evaluate(([k, z]) => { window.__chain.set(k == null ? window.__chain.pings - 1 : k); if (z && window.__chainZoom) window.__chainZoom(); },
                            [o.chainPing, o.zoomChain]);
        await page.waitForTimeout(200);
      }
      if (o.shots) await page.screenshot({ path: path.join(o.shots, path.basename(p).replace(/\.html$/, '') + '.png') });
    } catch (e) {
      errors.push('load: ' + (e && e.message ? e.message.split('\n')[0] : String(e)));
    }
    await ctx.close();
    const ok = errors.length === 0;
    if (!ok) failed++;
    console.log(JSON.stringify({ page: p, ok, errors, info, chain, ms: Date.now() - t0 }));
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
