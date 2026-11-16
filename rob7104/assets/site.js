// Shared behaviour for the ROB7104 lab pages.
// Tabs: each .feature-tabs group switches the .feature-panel elements of its own <section>.
// A panel may hold videos and/or live demos; a demo is told when it becomes visible or hidden
// through the "panel:show" / "panel:hide" events dispatched on the panel.

document.querySelectorAll('.feature-tabs').forEach((group) => {
  const section = group.closest('section') || document;
  const tabs = group.querySelectorAll('.feature-tab');
  const panels = section.querySelectorAll('.feature-panel');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      if (tab.classList.contains('active')) return;
      tabs.forEach((t) => {
        t.classList.toggle('active', t === tab);
        t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
      });
      panels.forEach((panel) => {
        const on = panel.dataset.feature === tab.dataset.feature;
        const was = panel.classList.contains('active');
        panel.classList.toggle('active', on);
        panel.querySelectorAll('video').forEach((video) => {
          if (on) { video.currentTime = 0; video.play().catch(() => {}); } else video.pause();
        });
        if (on && !was) panel.dispatchEvent(new CustomEvent('panel:show'));
        if (!on && was) panel.dispatchEvent(new CustomEvent('panel:hide'));
      });
    });
  });
});

// Videos play only while on screen. Elements with [data-live] (canvases of live demos)
// get "live:on" / "live:off" events so they can stop their animation loop when off screen.
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const el = entry.target;
      if (el.tagName === 'VIDEO') {
        if (entry.isIntersecting) el.play().catch(() => {}); else el.pause();
      } else {
        el.dispatchEvent(new CustomEvent(entry.isIntersecting ? 'live:on' : 'live:off'));
      }
    });
  }, { threshold: 0.2 });
  document.querySelectorAll('video, [data-live]').forEach((el) => observer.observe(el));

  const reveal = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) { entry.target.classList.add('is-visible'); reveal.unobserve(entry.target); }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach((el) => reveal.observe(el));
} else {
  document.querySelectorAll('video').forEach((video) => video.play().catch(() => {}));
  document.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-visible'));
}

// Sliders: mirror the value into the <output> that follows, formatted by data-fmt (digits) and data-unit.
document.querySelectorAll('.controls input[type="range"]').forEach((input) => {
  const out = input.parentElement.querySelector('output');
  if (!out) return;
  const show = () => {
    const digits = Number(input.dataset.fmt || 2);
    out.textContent = Number(input.value).toFixed(digits) + (input.dataset.unit || '');
  };
  input.addEventListener('input', show);
  show();
});
