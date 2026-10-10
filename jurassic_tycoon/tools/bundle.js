// Bundle index.html + src/*.js into one self-contained page body for publishing as an artifact.
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const out = process.argv[2] || path.join(root, 'dist', 'jurassic-tycoon.html');
let html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
html = html.replace(/<script src="(src\/[^"]+)"><\/script>/g, (m, f) => '<script>\n' + fs.readFileSync(path.join(root, f), 'utf8').replace(/<\/script/g, '<\\/script') + '\n</script>');
// The artifact host provides doctype/html/head/body; keep our content in order.
html = html.replace(/<!doctype html>\s*/i, '').replace(/<\/?html[^>]*>\s*/g, '').replace(/<\/?head>\s*/g, '').replace(/<\/?body>\s*/g, '')
  .replace(/<meta charset[^>]*>\s*/, '').replace(/<meta name="viewport"[^>]*>\s*/, '');
// the title must lead the file
html = html.replace(/(<title>[^<]*<\/title>)\s*/, '');
html = '<title>Jurassic Tycoon</title>\n' + html;
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, html);
console.log(out, (html.length / 1024).toFixed(0) + 'KB');
