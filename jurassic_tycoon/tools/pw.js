// Resolve Playwright from the project (CI) or this machine's global install.
module.exports = (() => {
  try { return require('playwright'); } catch (e) { return require('/opt/node22/lib/node_modules/playwright'); }
})();
