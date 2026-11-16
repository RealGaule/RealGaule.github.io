// Lab 3 page: start each demo the first time its panel is both the active tab and on screen.
const started = new WeakSet();

async function start(panel) {
  if (started.has(panel)) return;
  started.add(panel);
  try {
    const mod = await import(`./demos/${panel.dataset.demo}.js`);
    await mod.init(panel);
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

// The main thread: clicking a card lights the matching term of the formula and scrolls to its part.
document.querySelectorAll('.about-card.pick').forEach((card) => {
  const light = () => {
    document.querySelectorAll('.thread .term').forEach((t) => t.classList.toggle('lit', t.classList.contains(card.dataset.term)));
  };
  card.addEventListener('mouseenter', light);
  card.addEventListener('focus', light);
});

// Deep links: #d12 opens that demo's tab and scrolls to it (used by the notebook's "Try it" links).
function openFromHash() {
  const id = location.hash.slice(1);
  if (!/^d\d+[a-z]?$/.test(id)) return;
  const tab = document.querySelector(`.feature-tab[data-feature="${id}"]`);
  const panel = document.querySelector(`.feature-panel[data-feature="${id}"]`);
  if (!tab || !panel) return;
  tab.click();
  panel.closest('section').scrollIntoView({ behavior: 'auto', block: 'start' });
}
window.addEventListener('hashchange', openFromHash);
window.addEventListener('load', openFromHash);
