// Lab 6 page: start each demo the first time its panel is both the active tab and on screen.
const started = new WeakSet();
const failures = {};

async function start(panel) {
  if (started.has(panel)) return;
  started.add(panel);
  const name = panel.dataset.demo;
  try {
    const mod = await import(`./demos/${name}.js`);
    await mod.init(panel);
    panel.dataset.ready = '1';
  } catch (e) {
    console.error(e);
    failures[name] = String((e && e.message) || e);
    panel.dataset.error = failures[name];
    const veil = panel.querySelector('.loading');
    if (veil) veil.textContent = '3D could not start in this browser';
  }
}

const io = new IntersectionObserver((entries) => {
  entries.forEach((en) => { if (en.isIntersecting && en.target.classList.contains('active')) start(en.target); });
}, { rootMargin: '300px' });

document.querySelectorAll('.feature-panel[data-demo]').forEach((panel) => {
  io.observe(panel);
  panel.addEventListener('panel:show', () => start(panel));
});

// The main thread: pointing at a card lights the matching terms of the formula.
document.querySelectorAll('.about-card.pick').forEach((card) => {
  const light = (on) => {
    document.querySelectorAll('.thread .term').forEach((t) => t.classList.toggle('lit', on && t.classList.contains(card.dataset.term)));
  };
  card.addEventListener('mouseenter', () => light(true));
  card.addEventListener('focus', () => light(true));
  card.addEventListener('mouseleave', () => light(false));
  card.addEventListener('blur', () => light(false));
});

// Deep links: #d12 opens that demo's tab and scrolls to it (used by the notebook's "Try it" links and the summary table).
function openFromHash() {
  const id = location.hash.slice(1);
  if (!/^d\d+[a-z]?$/.test(id)) return;
  const panel = document.querySelector(`.feature-panel[data-feature="${id}"]`);
  if (!panel) return;
  const tab = document.querySelector(`.feature-tab[data-feature="${id}"]`);
  if (tab) tab.click();
  panel.scrollIntoView({ behavior: 'auto', block: 'start' });
}
window.addEventListener('hashchange', openFromHash);
window.addEventListener('load', openFromHash);

// For tests: numbers about the page frame (the demos expose their own panel._state()).
window._state = () => {
  const panels = [...document.querySelectorAll('.feature-panel')];
  return {
    panels: panels.map((p) => p.dataset.feature),
    active: panels.filter((p) => p.classList.contains('active')).map((p) => p.dataset.feature),
    demos: panels.filter((p) => p.dataset.demo).map((p) => p.dataset.demo),
    ready: panels.filter((p) => p.dataset.ready).map((p) => p.dataset.feature),
    failed: failures,
    films: document.querySelectorAll('.film video').length,
    gallery: document.querySelectorAll('.style-grid video').length,
    katex: document.body.dataset.katex === '1' ? document.querySelectorAll('.katex').length : 0,
    katexErrors: document.querySelectorAll('.katex-error').length,
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  };
};

// For tests: HTTP status of every video and poster on the page (HEAD requests).
window._checkMedia = async () => {
  const urls = new Set();
  document.querySelectorAll('video').forEach((v) => { if (v.getAttribute('src')) urls.add(v.getAttribute('src')); if (v.getAttribute('poster')) urls.add(v.getAttribute('poster')); });
  const out = {};
  await Promise.all([...urls].map(async (u) => {
    try { const r = await fetch(u, { method: 'HEAD' }); out[u] = r.status; } catch (e) { out[u] = String(e); }
  }));
  const bad = Object.entries(out).filter(([, s]) => s !== 200);
  return { checked: urls.size, bad };
};
