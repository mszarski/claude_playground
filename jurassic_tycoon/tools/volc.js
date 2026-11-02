(() => {
  document.getElementById('btnSandbox').click();
  const g = window.GAME, w = g.world;
  g.time = 12;
  g.events.trigger('volcano');
  T.run(26);
  RENDER.cam.zoom = 3; UIX.centerOn(w.volcano.x, w.volcano.y + 2);
  let lava = 0; for (const t of w.terrain) if (t === T_LAVA) lava++;
  return { lava, v: w.volcano, phase: g.events.volcano && g.events.volcano.phase };
})()
