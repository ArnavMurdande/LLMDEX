document.addEventListener("DOMContentLoaded", () => {
  // We will initialize bento cards when GSAP is available

  let isSpotlightCreated = false;

  function initBentoCards() {
    if (typeof gsap === "undefined") return;

    // Disable heavy animations on mobile
    const isMobile = window.innerWidth <= 768;
    if (isMobile) return;

    // Find all cards we want to apply the bento effect to
    const cards = document.querySelectorAll(
      ".metric-card, .chart-card, .method-card, .sentiment-card, .advisor-chatbot, .sentiment-explainer",
    );

    cards.forEach((card) => {
      if (card.dataset.bentoInitialized === "true") return;
      card.dataset.bentoInitialized = "true";

      card.classList.add("bento-card");

 

      // Click Effect (Ripple)
      card.addEventListener("click", (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const maxDistance = Math.max(
          Math.hypot(x, y),
          Math.hypot(x - rect.width, y),
          Math.hypot(x, y - rect.height),
          Math.hypot(x - rect.width, y - rect.height),
        );

        const ripple = document.createElement("div");
        ripple.style.cssText = `
                    position: absolute;
                    width: ${maxDistance * 2}px;
                    height: ${maxDistance * 2}px;
                    border-radius: 50%;
                    background: radial-gradient(circle, rgba(132, 0, 255, 0.4) 0%, rgba(132, 0, 255, 0.1) 40%, transparent 70%);
                    left: ${x - maxDistance}px;
                    top: ${y - maxDistance}px;
                    pointer-events: none;
                    z-index: 10;
                `;

        // insert as first child to be behind content if possible, or append
        card.appendChild(ripple);

        gsap.fromTo(
          ripple,
          {
            scale: 0,
            opacity: 1,
          },
          {
            scale: 1,
            opacity: 0,
            duration: 0.7,
            ease: "power2.out",
            onComplete: () => ripple.remove(),
          },
        );
      });
    });

    setupGlobalSpotlight();
  }

  function setupGlobalSpotlight() {
    if (isSpotlightCreated) return;
    isSpotlightCreated = true;

    const glowColor = "132, 0, 255";
    const spotlightRadius = 350;

    const spotlight = document.createElement("div");
    spotlight.id = "bento-spotlight";
    spotlight.className = "global-spotlight";
    spotlight.style.cssText = `
            position: fixed;
            width: 800px;
            height: 800px;
            border-radius: 50%;
            pointer-events: none;
            background: radial-gradient(circle,
                rgba(${glowColor}, 0.15) 0%,
                rgba(${glowColor}, 0.08) 15%,
                rgba(${glowColor}, 0.04) 25%,
                rgba(${glowColor}, 0.02) 40%,
                rgba(${glowColor}, 0.01) 65%,
                transparent 70%
            );
            z-index: 200;
            opacity: 0;
            transform: translate(-50%, -50%);
            mix-blend-mode: screen;
        `;
    document.body.appendChild(spotlight);

    document.addEventListener("mousemove", (e) => {
      if (window.innerWidth <= 768) return;

      const cards = document.querySelectorAll(".bento-card");
      let minDistance = Infinity;

      cards.forEach((card) => {
        const cardRect = card.getBoundingClientRect();
        const centerX = cardRect.left + cardRect.width / 2;
        const centerY = cardRect.top + cardRect.height / 2;
        const distance =
          Math.hypot(e.clientX - centerX, e.clientY - centerY) -
          Math.max(cardRect.width, cardRect.height) / 2;
        const effectiveDistance = Math.max(0, distance);

        minDistance = Math.min(minDistance, effectiveDistance);

        const proximity = spotlightRadius * 0.5;
        const fadeDistance = spotlightRadius * 0.85;

        let glowIntensity = 0;
        if (effectiveDistance <= proximity) {
          glowIntensity = 1;
        } else if (effectiveDistance <= fadeDistance) {
          glowIntensity =
            (fadeDistance - effectiveDistance) / (fadeDistance - proximity);
        }

        const relativeX =
          ((e.clientX - cardRect.left) / (cardRect.width || 1)) * 100;
        const relativeY =
          ((e.clientY - cardRect.top) / (cardRect.height || 1)) * 100;

        card.style.setProperty("--glow-x", `${relativeX}%`);
        card.style.setProperty("--glow-y", `${relativeY}%`);
        card.style.setProperty("--glow-intensity", glowIntensity.toString());
        card.style.setProperty("--glow-radius", `${spotlightRadius}px`);
      });

      gsap.to(spotlight, {
        left: e.clientX,
        top: e.clientY,
        duration: 0.1,
        ease: "power2.out",
      });

      const proximity = spotlightRadius * 0.5;
      const fadeDistance = spotlightRadius * 0.85;

      const targetOpacity =
        minDistance <= proximity
          ? 0.8
          : minDistance <= fadeDistance
            ? ((fadeDistance - minDistance) / (fadeDistance - proximity)) * 0.8
            : 0;

      gsap.to(spotlight, {
        opacity: targetOpacity,
        duration: targetOpacity > 0 ? 0.2 : 0.5,
        ease: "power2.out",
      });
    });

    document.addEventListener("mouseleave", () => {
      gsap.to(spotlight, { opacity: 0, duration: 0.3, ease: "power2.out" });
    });
  }

  // Try to init immediately and set up an observer for dynamically loaded cards
  setTimeout(initBentoCards, 300);

  const observer = new MutationObserver(() => {
    initBentoCards();
  });

  observer.observe(document.body, { childList: true, subtree: true });
});
