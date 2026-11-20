// Lab 5 page: start each demo the first time its panel is both the active tab and on screen.
// A demo module js/demos/dXX.js exports `async function init(panel)`; see build/web/README_DEV.md.
const started = new WeakSet();

// assets/site.js sends "live:on" / "live:off" only to [data-live] elements that exist when it loads, but a viewer marks
// its canvas data-live later (MjViewer.autoRun), so without this every started 3D view kept rendering off screen.
const liveIO = new IntersectionObserver((entries) => {
  entries.forEach((en) => en.target.dispatchEvent(new CustomEvent(en.isIntersecting ? 'live:on' : 'live:off')));
}, { threshold: 0.2 });

async function start(panel) {
  if (started.has(panel)) return;
  started.add(panel);
  try {
    const mod = await import(`./demos/${panel.dataset.demo}.js`);
    await mod.init(panel);
    panel.querySelectorAll('[data-live]').forEach((el) => liveIO.observe(el));
    panel.dataset.ready = '1';
  } catch (e) {
    console.error(e);
    panel.dataset.error = String(e && e.message || e);
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

// The main thread: hovering a card lights the matching terms of the two formulas.
document.querySelectorAll('.about-card.pick').forEach((card) => {
  const terms = card.dataset.term.split(/\s+/);
  const light = () => {
    document.querySelectorAll('.thread .term').forEach((t) => t.classList.toggle('lit', terms.some((c) => t.classList.contains(c))));
  };
  const dark = () => document.querySelectorAll('.thread .term').forEach((t) => t.classList.remove('lit'));
  card.addEventListener('mouseenter', light);
  card.addEventListener('focus', light);
  card.addEventListener('mouseleave', dark);
  card.addEventListener('blur', dark);
});

// Deep links: #d12 opens that demo's tab and scrolls to it (used by the notebook and the gallery).
function openFromHash() {
  const id = location.hash.slice(1);
  if (!/^d\d+[a-z]?$/.test(id)) return;
  const panel = document.querySelector(`.feature-panel[data-feature="${id}"]`);
  if (!panel) return;
  const tab = document.querySelector(`.feature-tab[data-feature="${id}"]`);
  if (tab) tab.click();
  panel.closest('section').scrollIntoView({ behavior: 'auto', block: 'start' });
}
window.addEventListener('hashchange', openFromHash);
window.addEventListener('load', openFromHash);

// Numbers for the headless tests (build/web/pw.py: ["eval", "window._state()"]).
window._state = () => ({
  panels: [...document.querySelectorAll('.feature-panel')].map((p) => ({
    id: p.dataset.feature,
    active: p.classList.contains('active'),
    demo: !!p.dataset.demo,
    ready: !!p.dataset.ready,
    error: p.dataset.error || null,
    clip: (p.querySelector('video') || { getAttribute: () => null }).getAttribute('src'),
  })),
  tabs: document.querySelectorAll('.feature-tab').length,
  videos: document.querySelectorAll('video').length,
  gallery: document.querySelectorAll('#films .style-item').length,
  katex: document.querySelectorAll('.katex').length,
  katexErrors: document.querySelectorAll('.katex-error').length,
  viewport: window.innerWidth,
  pageWidth: document.documentElement.scrollWidth,   // > viewport means a horizontal scroll
  heroPlaying: (() => { const h = document.querySelector('.hero-video'); return !!h && !h.paused; })(),
});
