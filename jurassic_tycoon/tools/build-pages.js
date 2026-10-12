// Build the installable offline web app for GitHub Pages: node tools/build-pages.js [outDir]
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const { inline } = require('./bundle');
const root = path.resolve(__dirname, '..');
const out = process.argv[2] || path.join(root, 'dist', 'pages');
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(path.join(out, 'icons'), { recursive: true });
fs.mkdirSync(path.join(out, 'fonts'), { recursive: true });
const html = inline();
fs.writeFileSync(path.join(out, 'index.html'), html);
for (const f of ['manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon.png', 'fonts/OFL-PressStart2P.txt', 'fonts/OFL-VT323.txt']) {
  fs.copyFileSync(path.join(root, f), path.join(out, f));
}
// a new cache version whenever the game changes, so installed copies update
const version = crypto.createHash('sha1').update(html).update(fs.readFileSync(path.join(root, 'manifest.webmanifest'))).digest('hex').slice(0, 10);
fs.writeFileSync(path.join(out, 'sw.js'), fs.readFileSync(path.join(root, 'sw.js'), 'utf8').replace("const VERSION = 'dev';", `const VERSION = '${version}';`));
fs.writeFileSync(path.join(out, '.nojekyll'), '');
console.log(out, 'version', version, (html.length / 1024).toFixed(0) + 'KB');
