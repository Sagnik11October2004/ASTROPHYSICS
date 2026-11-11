'use strict';

(function () {
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  const menu = document.getElementById('menu-toggle');
  const nav = document.getElementById('main-nav');
  const progress = document.getElementById('reading-progress');
  const year = document.getElementById('current-year');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (year) year.textContent = new Date().getFullYear().toString();

  try {
    if (localStorage.getItem('portfolio-theme') === 'light') root.dataset.theme = 'light';
  } catch (e) { /* Storage can be disabled; the default theme still works. */ }

  function updateThemeLabel() {
    if (toggle) toggle.setAttribute('aria-label', root.dataset.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }
  updateThemeLabel();
  if (toggle) toggle.addEventListener('click', function () {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    updateThemeLabel();
    try { localStorage.setItem('portfolio-theme', root.dataset.theme); } catch (e) {}
  });

  function closeMenu() {
    if (!nav || !menu) return;
    nav.classList.remove('open');
    menu.setAttribute('aria-expanded', 'false');
  }
  if (menu && nav) {
    menu.addEventListener('click', function () {
      const isOpen = nav.classList.toggle('open');
      menu.setAttribute('aria-expanded', String(isOpen));
    });
    nav.querySelectorAll('a').forEach(function (anchor) { anchor.addEventListener('click', closeMenu); });
    window.addEventListener('keydown', function (event) { if (event.key === 'Escape') closeMenu(); });
  }

  let scrollScheduled = false;
  function updateScrollProgress() {
    const full = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.width = (full > 0 ? Math.max(0,Math.min(100,100 * window.scrollY / full)) : 0) + '%';
    scrollScheduled = false;
  }
  window.addEventListener('scroll', function () {
    if (!scrollScheduled) {
      scrollScheduled = true;
      window.requestAnimationFrame(updateScrollProgress);
    }
  }, { passive: true });
  window.addEventListener('resize', updateScrollProgress);
  updateScrollProgress();

  if ('IntersectionObserver' in window && !reducedMotion.matches) {
    root.classList.add('js-motion');
    const appear = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) { entry.target.classList.add('visible'); observer.unobserve(entry.target); }
      });
    }, { threshold: .09, rootMargin: '0px 0px -28px 0px' });
    document.querySelectorAll('.reveal').forEach(function (node) { appear.observe(node); });
  }

  // Educational log-log density model. This does not represent a measured profile.
  const slider = document.getElementById('slope-slider');
  const slopeOutput = document.getElementById('slope-value');
  const ratioOutput = document.getElementById('density-ratio');
  const chart = document.getElementById('density-chart');
  const presets = document.querySelectorAll('.preset-button');
  const NS = 'http://www.w3.org/2000/svg';

  function svgNode(tag, attributes, text) {
    const element = document.createElementNS(NS, tag);
    Object.keys(attributes || {}).forEach(function (key) { element.setAttribute(key, String(attributes[key])); });
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function drawChart(gamma) {
    if (!chart) return;
    while (chart.firstChild) chart.removeChild(chart.firstChild);
    const box = { x: 73, y: 27, w: 594, h: 329 };
    const minX = -2, maxX = 1, minY = -2.5, maxY = 5.1;
    function X(logRadius) { return box.x + (logRadius - minX) / (maxX - minX) * box.w; }
    function Y(logDensity) { return box.y + (maxY - logDensity) / (maxY - minY) * box.h; }
    const defs = svgNode('defs');
    const grad = svgNode('linearGradient', { id: 'cusp-fill', x1: '0', y1: '0', x2: '0', y2: '1' });
    grad.appendChild(svgNode('stop', { offset: '0%', 'stop-color': 'var(--accent)', 'stop-opacity': '.18' }));
    grad.appendChild(svgNode('stop', { offset: '100%', 'stop-color': 'var(--accent)', 'stop-opacity': '0' }));
    defs.appendChild(grad);
    const clip = svgNode('clipPath', { id: 'graph-clip' });
    clip.appendChild(svgNode('rect', { x: box.x, y: box.y, width: box.w, height: box.h }));
    defs.appendChild(clip);
    chart.appendChild(defs);
    const xTicks = [{ v: -2, l: '0.01' }, { v: -1, l: '0.1' }, { v: 0, l: '1' }, { v: 1, l: '10' }];
    const yTicks = [{ v: -2, l: '10⁻²' }, { v: 0, l: '1' }, { v: 2, l: '10²' }, { v: 4, l: '10⁴' }];
    xTicks.forEach(function (tick) {
      chart.appendChild(svgNode('line', { class: 'grid-line', x1: X(tick.v), x2: X(tick.v), y1: box.y, y2: box.y + box.h }));
      chart.appendChild(svgNode('text', { class: 'axis-label', x: X(tick.v), y: box.y + box.h + 27, 'text-anchor': 'middle' }, tick.l));
    });
    yTicks.forEach(function (tick) {
      chart.appendChild(svgNode('line', { class: 'grid-line', x1: box.x, x2: box.x + box.w, y1: Y(tick.v), y2: Y(tick.v) }));
      chart.appendChild(svgNode('text', { class: 'axis-label', x: box.x - 14, y: Y(tick.v) + 5, 'text-anchor': 'end' }, tick.l));
    });
    chart.appendChild(svgNode('line', { class: 'axis-line', x1: box.x, x2: box.x + box.w, y1: box.y + box.h, y2: box.y + box.h }));
    chart.appendChild(svgNode('line', { class: 'axis-line', x1: box.x, x2: box.x, y1: box.y, y2: box.y + box.h }));
    chart.appendChild(svgNode('text', { class: 'axis-title', x: box.x + box.w / 2, y: 417, 'text-anchor': 'middle' }, 'Radius r (pc)'));
    chart.appendChild(svgNode('text', { class: 'axis-title', x: 17, y: box.y + box.h / 2, transform: 'rotate(-90 17 ' + (box.y + box.h / 2) + ')', 'text-anchor': 'middle' }, 'Normalized density ρ(r) / ρ(1 pc)'));
    function lineData(slope) {
      let d = '';
      for (let i = 0; i <= 120; i++) {
        const r = minX + (maxX - minX) * i / 120;
        d += (i === 0 ? 'M ' : ' L ') + X(r).toFixed(2) + ' ' + Y(-slope * r).toFixed(2);
      }
      return d;
    }
    const plot = svgNode('g', { 'clip-path': 'url(#graph-clip)' });
    const currentPath = lineData(gamma);
    const area = currentPath + ' L ' + X(maxX) + ' ' + (box.y + box.h) + ' L ' + X(minX) + ' ' + (box.y + box.h) + ' Z';
    plot.appendChild(svgNode('path', { d: area, fill: 'url(#cusp-fill)' }));
    plot.appendChild(svgNode('path', { d: lineData(1), class: 'reference-line' }));
    plot.appendChild(svgNode('path', { d: currentPath, class: 'current-line' }));
    plot.appendChild(svgNode('line', { class: 'guide-line', x1: X(-1), x2: X(-1), y1: Y(gamma), y2: box.y + box.h }));
    plot.appendChild(svgNode('circle', { class: 'model-dot', cx: X(-1), cy: Y(gamma), r: 7 }));
    chart.appendChild(plot);
  }
  function updateSlope(value) {
    const gamma = Math.max(.5, Math.min(2.5, Number(value) || 1.8));
    if (slider) slider.value = gamma.toFixed(2);
    if (slopeOutput) slopeOutput.textContent = gamma.toFixed(2);
    if (ratioOutput) ratioOutput.textContent = Math.pow(10, gamma).toFixed(1) + '×';
    presets.forEach(function (button) {
      button.classList.toggle('active', Math.abs(Number(button.dataset.gamma) - gamma) < .001);
    });
    drawChart(gamma);
  }
  if (slider) {
    slider.addEventListener('input', function () { updateSlope(slider.value); });
    presets.forEach(function (button) {
      button.addEventListener('click', function () { updateSlope(button.dataset.gamma); });
    });
    updateSlope(slider.value);
  }

  // Dependency-free black hole artwork drawn with Canvas 2D.
  // The animation is artistic and is not a ray-traced GR model.
  const canvas = document.getElementById('blackhole-canvas');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) return;
  let width = 600, height = 600, visible = true, lastFrame = 0, handle = 0;
  function randomGenerator(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a * 1664525 + 1013904223) >>> 0;
      return a / 4294967296;
    };
  }
  const rnd = randomGenerator(7112026);
  const stars = Array.from({ length: 145 }, function () {
    return { x: rnd(), y: rnd(), r: rnd() * 1.25 + .24, glow: rnd(), phase: rnd() * 6.283 };
  });
  const dust = Array.from({ length: 245 }, function () {
    return { a: rnd() * Math.PI * 2, orbit: .48 + rnd() * .57, size: .5 + rnd() * 1.65, hue: rnd(), speed: .18 + rnd() * .8 };
  });
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    width = rect.width; height = rect.height;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawFrame(performance.now());
  }
  function ellipsePoint(angle, radius, cx, cy, squash, tilt) {
    const px = radius * Math.cos(angle), py = radius * squash * Math.sin(angle);
    return { x: cx + px * Math.cos(tilt) - py * Math.sin(tilt), y: cy + px * Math.sin(tilt) + py * Math.cos(tilt) };
  }
  function ringStroke(radius, thickness, color, cx, cy, squash, tilt, a0, a1, glow) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(tilt);
    ctx.scale(1,squash);
    ctx.beginPath();
    ctx.arc(0, 0, radius, a0, a1);
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness / squash;
    ctx.lineCap = 'round';
    ctx.shadowBlur = glow;
    ctx.shadowColor = color;
    ctx.stroke();
    ctx.restore();
  }
  function paintDisk(front, cx, cy, s, time) {
    const squash = .30, tilt = -.18;
    // Ellipse arc parameter angle: upper half = back, lower half = front.
    const a0 = front ? .04 : Math.PI + .04;
    const a1 = front ? Math.PI - .04 : Math.PI * 2 - .04;
    for (let k = 0; k < 13; k++) {
      const radius = s * (.245 + k * .0225);
      const alpha = (front ? .34 : .2) * (1 - k / 23);
      const hue = k < 5 ? '255,194,122' : '116,197,211';
      ringStroke(radius, 1.35 + (k < 6 ? 1.1 : 0), 'rgba(' + hue + ',' + alpha + ')', cx, cy, squash, tilt, a0, a1, k < 5 ? 9 : 3);
    }
    ctx.save();
    dust.forEach(function (d) {
      const angle = d.a + time * d.speed;
      const isFront = Math.sin(angle) >= 0;
      if (front !== isFront) return;
      const radius = s * d.orbit * .47;
      const p = ellipsePoint(angle, radius, cx, cy, squash, tilt);
      const color = d.hue > .47 ? '255,181,107' : '110,207,226';
      const alpha = (front ? .28 : .18) + (d.hue * .3);
      ctx.beginPath();
      ctx.fillStyle = 'rgba(' + color + ',' + alpha.toFixed(3) + ')';
      ctx.shadowColor = 'rgba(' + color + ',.6)';
      ctx.shadowBlur = d.size * 3;
      ctx.arc(p.x,p.y,d.size * Math.max(.52,s/600),0,Math.PI*2);
      ctx.fill();
    });
    ctx.restore();
  }
  function drawFrame(timestamp) {
    const s = Math.min(width,height), cx = width * .5, cy = height * .51;
    const t = timestamp * .00018;
    ctx.clearRect(0,0,width,height);
    stars.forEach(function (star) {
      const a = .16 + star.glow * .37 + .07 * Math.sin(t * 7 + star.phase);
      ctx.beginPath();
      ctx.fillStyle = 'rgba(183,224,239,' + Math.max(.05,a).toFixed(3) + ')';
      ctx.arc(star.x * width, star.y * height,star.r * Math.max(.55,s/600),0,Math.PI*2);
      ctx.fill();
    });
    // An ambient halo keeps the image legible without any downloaded assets.
    let halo = ctx.createRadialGradient(cx,cy,s*.055,cx,cy,s*.43);
    halo.addColorStop(0,'rgba(255,145,64,.23)');
    halo.addColorStop(.3,'rgba(236,120,65,.12)');
    halo.addColorStop(.60,'rgba(69,120,155,.065)');
    halo.addColorStop(1,'rgba(69,120,155,0)');
    ctx.fillStyle = halo; ctx.fillRect(0,0,width,height);
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    paintDisk(false,cx,cy,s,t);
    ctx.restore();
    let shadow = ctx.createRadialGradient(cx-s*.018,cy-s*.02,s*.035,cx,cy,s*.154);
    shadow.addColorStop(0,'#000003');
    shadow.addColorStop(.78,'#000004');
    shadow.addColorStop(.92,'#090909');
    shadow.addColorStop(1,'rgba(0,0,0,0)');
    ctx.beginPath(); ctx.fillStyle = shadow; ctx.arc(cx,cy,s*.155,0,Math.PI*2); ctx.fill();
    // Bright, asymmetrical photon-ring-inspired edge.
    ctx.save(); ctx.translate(cx,cy); ctx.rotate(-.34);
    ctx.beginPath(); ctx.ellipse(0,0,s*.148,s*.147,0,0,Math.PI*2);
    ctx.strokeStyle = 'rgba(255,189,116,.52)'; ctx.lineWidth = s*.007;
    ctx.shadowColor = '#ffab67'; ctx.shadowBlur = s*.034; ctx.stroke();
    ctx.beginPath(); ctx.arc(0,0,s*.149,.10,2.55);
    ctx.strokeStyle = 'rgba(255,216,149,.92)'; ctx.lineWidth = s*.010;
    ctx.shadowBlur = s*.026; ctx.stroke();
    ctx.restore();
    // Re-draw foreground material so it visibly passes in front of the shadow.
    ctx.save(); ctx.globalCompositeOperation = 'screen';
    paintDisk(true,cx,cy,s,t);
    ctx.restore();
    // A diffuse reflected halo and sparse orbital highlights.
    ringStroke(s*.265,1.2,'rgba(255,170,96,.24)',cx,cy,.48,-.18,Math.PI*.12,Math.PI*.85,8);
    ringStroke(s*.37,1,'rgba(118,199,218,.13)',cx,cy,.33,-.18,Math.PI*1.1,Math.PI*1.87,4);
  }
  function animate(now) {
    handle = 0;
    if (!visible || document.hidden || reducedMotion.matches) return;
    if (now - lastFrame >= 33) {
      drawFrame(now);
      lastFrame = now;
    }
    handle = window.requestAnimationFrame(animate);
  }
  function syncAnimation() {
    window.cancelAnimationFrame(handle);
    handle = 0;
    if (visible && !document.hidden && !reducedMotion.matches) handle = window.requestAnimationFrame(animate);
    else drawFrame(0);
  }
  if ('IntersectionObserver' in window) {
    const canvasObserver = new IntersectionObserver(function (entries) {
      visible = entries[0] && entries[0].isIntersecting;
      syncAnimation();
    }, { threshold: .01 });
    canvasObserver.observe(canvas);
  }
  window.addEventListener('resize', resizeCanvas, { passive: true });
  document.addEventListener('visibilitychange',syncAnimation);
  if (reducedMotion.addEventListener) reducedMotion.addEventListener('change',syncAnimation);
  resizeCanvas();
  syncAnimation();
})();
