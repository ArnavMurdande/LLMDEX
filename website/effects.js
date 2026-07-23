(() => {
  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  );
  const buttonSelector = "button, a.btn";
  const visibleButtons = new Set();

  function installSpecularButtons() {
    const observer =
      "IntersectionObserver" in window
        ? new IntersectionObserver(
            (entries) => {
              entries.forEach((entry) => {
                if (entry.isIntersecting) visibleButtons.add(entry.target);
                else visibleButtons.delete(entry.target);
              });
            },
            { rootMargin: "260px" },
          )
        : null;

    const enhance = (root = document) => {
      const candidates = [];
      if (root.matches?.(buttonSelector)) candidates.push(root);
      root.querySelectorAll?.(buttonSelector).forEach((button) => {
        candidates.push(button);
      });
      candidates.forEach((button) => {
        if (button.dataset.specularReady === "true") return;
        button.dataset.specularReady = "true";
        button.classList.add("specular-button");
        if (observer) observer.observe(button);
        else visibleButtons.add(button);
      });
    };

    enhance();
    new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) enhance(node);
        });
        mutation.removedNodes.forEach((node) => {
          if (node.nodeType !== Node.ELEMENT_NODE) return;
          if (visibleButtons.has(node)) visibleButtons.delete(node);
          node.querySelectorAll?.(buttonSelector).forEach((button) => {
            visibleButtons.delete(button);
          });
        });
      });
    }).observe(document.body, { childList: true, subtree: true });

    let pointer = null;
    let frame = 0;
    const render = () => {
      frame = 0;
      if (!pointer || reducedMotion.matches) return;
      visibleButtons.forEach((button) => {
        if (!button.isConnected) {
          visibleButtons.delete(button);
          observer?.unobserve(button);
          return;
        }
        if (button.hidden) return;
        const rect = button.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const closestX = Math.max(rect.left, Math.min(pointer.x, rect.right));
        const closestY = Math.max(rect.top, Math.min(pointer.y, rect.bottom));
        const distance = Math.hypot(
          pointer.x - closestX,
          pointer.y - closestY,
        );
        const raw = Math.max(0, 1 - distance / 250);
        const opacity = raw * raw * (3 - 2 * raw);
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const angle =
          (Math.atan2(pointer.y - centerY, pointer.x - centerX) * 180) /
            Math.PI +
          90;
        button.style.setProperty("--specular-angle", `${angle}deg`);
        button.style.setProperty("--specular-opacity", opacity.toFixed(3));
      });
    };

    window.addEventListener(
      "pointermove",
      (event) => {
        pointer = { x: event.clientX, y: event.clientY };
        if (!frame) frame = window.requestAnimationFrame(render);
      },
      { passive: true },
    );
    document.documentElement.classList.add("specular-effects-ready");
  }

  const start = () => installSpecularButtons();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
