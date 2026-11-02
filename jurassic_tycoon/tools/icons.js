// Render the home-screen icons from the game's own T. rex sprite: node tools/icons.js
const { chromium } = require('./pw');
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.join(root, 'index.html'));
  await p.waitForTimeout(300);
  const out = await p.evaluate(() => {
    const spr = DINO_SPRITES.trex.right[0];
    const make = (size, fill) => {
      const c = document.createElement('canvas'); c.width = c.height = size;
      const x = c.getContext('2d'); x.imageSmoothingEnabled = false;
      // banded pixel sunset, like the title screen, over a jungle floor
      const bands = ['#3a1a2a', '#6a2420', '#a03a18', '#d0601a', '#f0a030', '#f8d040'];
      const bh = Math.ceil((size * 0.72) / bands.length);
      bands.forEach((col, i) => { x.fillStyle = col; x.fillRect(0, i * bh, size, bh); });
      x.fillStyle = '#1c2a1a'; x.fillRect(0, size * 0.72, size, size * 0.28);
      const s = Math.floor((size * fill) / Math.max(spr.width, spr.height));
      const w = spr.width * s, h = spr.height * s;
      const ox = Math.round((size - w) / 2), oy = Math.round(size * 0.78 - h);
      x.fillStyle = 'rgba(0,0,0,0.35)'; x.fillRect(ox + s * 3, Math.round(size * 0.78) - s, w - s * 6, s * 2);
      x.drawImage(spr, ox, oy, w, h);
      return c.toDataURL('image/png').split(',')[1];
    };
    // 'any' icons fill more of the square; maskable ones keep the dino inside the 80% safe zone
    return { i192: make(192, 0.8), i512: make(512, 0.8), apple: make(180, 0.8), mask: make(512, 0.56) };
  });
  const dir = path.join(root, 'icons');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'icon-192.png'), Buffer.from(out.i192, 'base64'));
  fs.writeFileSync(path.join(dir, 'icon-512.png'), Buffer.from(out.i512, 'base64'));
  fs.writeFileSync(path.join(dir, 'apple-touch-icon.png'), Buffer.from(out.apple, 'base64'));
  fs.writeFileSync(path.join(dir, 'icon-maskable-512.png'), Buffer.from(out.mask, 'base64'));
  console.log('icons written to', dir);
  await b.close();
})();
