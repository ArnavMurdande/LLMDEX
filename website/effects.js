(() => {
  const REDUCED_MOTION = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  );
  const BUTTON_SELECTOR = "button, a.btn";
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
      if (root.matches?.(BUTTON_SELECTOR)) candidates.push(root);
      root.querySelectorAll?.(BUTTON_SELECTOR).forEach((button) =>
        candidates.push(button),
      );
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
      });
    }).observe(document.body, { childList: true, subtree: true });

    let pointer = null;
    let frame = 0;
    const render = () => {
      frame = 0;
      if (!pointer || REDUCED_MOTION.matches) return;
      visibleButtons.forEach((button) => {
        if (!button.isConnected || button.hidden) return;
        const rect = button.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const closestX = Math.max(
          rect.left,
          Math.min(pointer.x, rect.right),
        );
        const closestY = Math.max(
          rect.top,
          Math.min(pointer.y, rect.bottom),
        );
        const distance = Math.hypot(
          pointer.x - closestX,
          pointer.y - closestY,
        );
        const proximity = 250;
        const raw = Math.max(0, 1 - distance / proximity);
        const opacity = raw * raw * (3 - 2 * raw);
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const angle =
          (Math.atan2(pointer.y - centerY, pointer.x - centerX) * 180) /
            Math.PI +
          90;
        button.style.setProperty("--specular-angle", `${angle}deg`);
        button.style.setProperty(
          "--specular-opacity",
          opacity.toFixed(3),
        );
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

  function installPixelBlast() {
    const canvas = document.createElement("canvas");
    canvas.className = "pixel-blast-background";
    canvas.setAttribute("aria-hidden", "true");
    document.body.prepend(canvas);
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;

    const ripples = [];
    let width = 1;
    let height = 1;
    let dpr = 1;
    let pointer = { x: -1000, y: -1000, active: false };
    let animationFrame = 0;
    let lastFrame = 0;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = Math.max(1, window.innerWidth);
      height = Math.max(1, window.innerHeight);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const color = () =>
      document.documentElement.dataset.theme === "light"
        ? [37, 99, 235]
        : [96, 165, 250];

    const noise = (x, y, time) => {
      const value =
        Math.sin(x * 0.071 + time * 0.18) +
        Math.sin(y * 0.083 - time * 0.14) +
        Math.sin((x + y) * 0.037 + time * 0.11);
      return value / 3;
    };

    const draw = (timeMs = 0) => {
      animationFrame = window.requestAnimationFrame(draw);
      if (document.hidden) return;
      if (timeMs - lastFrame < 32 && !REDUCED_MOTION.matches) return;
      lastFrame = timeMs;
      const time = REDUCED_MOTION.matches ? 0 : timeMs / 1000;
      context.clearRect(0, 0, width, height);
      const [red, green, blue] = color();
      const spacing = 22;
      const edge = Math.max(120, Math.min(width, height) * 0.16);

      ripples.forEach((ripple) => {
        ripple.age = (timeMs - ripple.startedAt) / 1000;
      });
      while (ripples.length && ripples[0].age > 2.8) ripples.shift();

      for (let y = spacing / 2; y < height; y += spacing) {
        for (let x = spacing / 2; x < width; x += spacing) {
          const field = noise(x, y, time * 0.75);
          let intensity = Math.max(0, (field - 0.05) * 0.34);
          const edgeFade = Math.min(
            1,
            x / edge,
            y / edge,
            (width - x) / edge,
            (height - y) / edge,
          );
          intensity *= Math.max(0, edgeFade);

          if (pointer.active && !REDUCED_MOTION.matches) {
            const distance = Math.hypot(x - pointer.x, y - pointer.y);
            if (distance < 180) {
              const liquid = 1 - distance / 180;
              intensity +=
                liquid *
                (0.08 + 0.06 * Math.sin(time * 5 + distance * 0.07));
            }
          }

          ripples.forEach((ripple) => {
            const distance = Math.hypot(x - ripple.x, y - ripple.y);
            const ring = ripple.age * 220;
            const delta = Math.abs(distance - ring);
            if (delta < 30) {
              intensity +=
                (1 - delta / 30) *
                Math.max(0, 1 - ripple.age / 2.8) *
                0.34;
            }
          });

          if (intensity < 0.035) continue;
          const jitter =
            1 + 0.22 * Math.sin(x * 12.9898 + y * 78.233 + time);
          const radius = Math.max(0.65, 1.7 * jitter);
          context.beginPath();
          context.arc(x, y, radius, 0, Math.PI * 2);
          context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${Math.min(
            0.42,
            intensity,
          ).toFixed(3)})`;
          context.fill();
        }
      }
    };

    window.addEventListener("resize", resize, { passive: true });
    window.addEventListener(
      "pointermove",
      (event) => {
        pointer = { x: event.clientX, y: event.clientY, active: true };
      },
      { passive: true },
    );
    window.addEventListener("pointerleave", () => {
      pointer.active = false;
    });
    window.addEventListener(
      "pointerdown",
      (event) => {
        ripples.push({
          x: event.clientX,
          y: event.clientY,
          startedAt: performance.now(),
          age: 0,
        });
        if (ripples.length > 10) ripples.shift();
      },
      { passive: true },
    );
    resize();
    animationFrame = window.requestAnimationFrame(draw);
    document.documentElement.classList.add("pixel-blast-ready");

    window.addEventListener("pagehide", () => {
      window.cancelAnimationFrame(animationFrame);
    });
  }

  const start = () => {
    installPixelBlast();
    installSpecularButtons();
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
