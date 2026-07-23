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

<<<<<<< HEAD
  function installPixelBlast() {
    const canvas = document.createElement("canvas");
    canvas.className = "pixel-blast-background";
    canvas.setAttribute("aria-hidden", "true");
    document.body.prepend(canvas);
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;

    const config = Object.freeze({
      variant: "square",
      pixelSize: 3,
      color: [56, 189, 248],
      patternScale: 4,
      patternDensity: 0.8,
      pixelSizeJitter: 0.8,
      enableRipples: true,
      rippleSpeed: 0.4,
      rippleThickness: 0.12,
      rippleIntensityScale: 1.5,
      liquid: false,
      speed: 0.6,
      edgeFade: 0.12,
    });
    const particles = [];
    const ripples = [];
    let width = 1;
    let height = 1;
    let dpr = 1;
    let animationFrame = 0;
    let lastFrame = 0;

    const makeParticle = () => {
      const angle = Math.random() * Math.PI * 2;
      const velocity = 12 + Math.random() * 16;
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: Math.cos(angle) * velocity,
        vy: Math.sin(angle) * velocity,
        phase: Math.random() * Math.PI * 2,
        jitter: Math.random() * 2 - 1,
        alpha: 0.3 + Math.random() * 0.34,
      };
    };

    const seedParticles = () => {
      particles.length = 0;
      const cellSize =
        config.pixelSize * config.patternScale * 4;
      const count = Math.max(
        44,
        Math.round(
          (width * height * config.patternDensity) /
            (cellSize * cellSize),
        ),
      );
      for (let index = 0; index < count; index += 1) {
        particles.push(makeParticle());
      }
    };

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = Math.max(1, window.innerWidth);
      height = Math.max(1, window.innerHeight);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      seedParticles();
    };

    const draw = (timeMs = 0) => {
      animationFrame = window.requestAnimationFrame(draw);
      if (document.hidden) return;
      if (timeMs - lastFrame < 24 && !REDUCED_MOTION.matches) return;
      const deltaSeconds = lastFrame
        ? Math.min(0.05, (timeMs - lastFrame) / 1000)
        : 0;
      lastFrame = timeMs;
      const time = REDUCED_MOTION.matches ? 0 : timeMs / 1000;
      context.clearRect(0, 0, width, height);
      const [red, green, blue] = config.color;
      const fadeDistance =
        Math.max(36, Math.min(width, height) * config.edgeFade);

      ripples.forEach((ripple) => {
        ripple.age = (timeMs - ripple.startedAt) / 1000;
      });
      while (ripples.length && ripples[0].age > 2.4) ripples.shift();

      particles.forEach((particle) => {
        if (!REDUCED_MOTION.matches) {
          const flowX = Math.sin(time * 0.75 + particle.phase) * 3.4;
          const flowY = Math.cos(time * 0.62 + particle.phase) * 3.4;
          particle.x +=
            (particle.vx + flowX) * config.speed * deltaSeconds;
          particle.y +=
            (particle.vy + flowY) * config.speed * deltaSeconds;
        }

        if (particle.x < -6) particle.x = width + 6;
        if (particle.x > width + 6) particle.x = -6;
        if (particle.y < -6) particle.y = height + 6;
        if (particle.y > height + 6) particle.y = -6;

        let drawX = particle.x;
        let drawY = particle.y;
        let intensity = particle.alpha;
        if (config.enableRipples) {
          ripples.forEach((ripple) => {
            const dx = particle.x - ripple.x;
            const dy = particle.y - ripple.y;
            const distance = Math.hypot(dx, dy);
            const radius = ripple.age * config.rippleSpeed * 260;
            const thickness = config.rippleThickness * 240;
            const distanceFromRing = Math.abs(distance - radius);
            if (distanceFromRing >= thickness) return;
            const strength =
              (1 - distanceFromRing / thickness) *
              Math.max(0, 1 - ripple.age / 2.4) *
              config.rippleIntensityScale;
            const normalX = distance ? dx / distance : 0;
            const normalY = distance ? dy / distance : 0;
            drawX += normalX * strength * 7;
            drawY += normalY * strength * 7;
            intensity += strength * 0.2;
          });
        }

        const edgeFade = Math.max(
          0,
          Math.min(
            1,
            drawX / fadeDistance,
            drawY / fadeDistance,
            (width - drawX) / fadeDistance,
            (height - drawY) / fadeDistance,
          ),
        );
        intensity *= edgeFade;
        if (intensity <= 0.015) return;

        const size = Math.max(
          1,
          config.pixelSize *
            (1 + particle.jitter * config.pixelSizeJitter * 0.5),
        );
        context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${Math.min(
          0.78,
          intensity,
        ).toFixed(3)})`;
        context.fillRect(
          Math.round(drawX - size / 2),
          Math.round(drawY - size / 2),
          size,
          size,
        );
      });
    };

    window.addEventListener("resize", resize, { passive: true });
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
=======
  const start = () => installSpecularButtons();
>>>>>>> codex/llmstats-consensus
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
