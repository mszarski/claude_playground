// Bundle index.html + src/*.js + fonts into one self-contained page.
//   node tools/bundle.js [out]   -> page body for publishing as a Claude artifact
//   require('./bundle').inline() -> full standalone HTML document (used by build-pages.js)
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');

function inline() {
  let html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  html = html.replace(/<script src="(src\/[^"]+)"><\/script>/g, (m, f) => '<script>\n' + fs.readFileSync(path.join(root, f), 'utf8').replace(/<\/script/g, '<\\/script') + '\n</script>');
  html = html.replace(/url\('(fonts\/[^']+\.woff2)'\)/g, (m, f) => `url(data:font/woff2;base64,${fs.readFileSync(path.join(root, f)).toString('base64')})`);
  return html;
}

function artifact() {
  // The artifact host provides doctype/html/head/body and can't serve the app's extra files.
  let html = inline();
  html = html.replace(/<!doctype html>\s*/i, '').replace(/<\/?html[^>]*>\s*/g, '').replace(/<\/?head>\s*/g, '').replace(/<\/?body>\s*/g, '')
    .replace(/<meta charset[^>]*>\s*/, '').replace(/<meta name="viewport"[^>]*>\s*/, '')
    .replace(/<!-- installable web app[^>]*-->\s*/, '').replace(/<(link|meta)[^>]*data-pwa>\s*/g, '');
  // the title must lead the file
  html = html.replace(/(<title>[^<]*<\/title>)\s*/, '');
  return '<title>Jurassic Tycoon</title>\n' + html;
}

if (require.main === module) {
  const out = process.argv[2] || path.join(root, 'dist', 'jurassic-tycoon.html');
  const html = artifact();
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, html);
  console.log(out, (html.length / 1024).toFixed(0) + 'KB');
}

module.exports = { inline, artifact };
