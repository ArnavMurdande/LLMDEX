import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import PixelBlast from "./PixelBlast";
import "./PixelBlast.css";

function PixelBlastBackground() {
  const [light, setLight] = useState(
    document.documentElement.dataset.theme === "light",
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setLight(document.documentElement.dataset.theme === "light");
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  return (
    <PixelBlast
      variant="square"
      pixelSize={4}
      color={light ? "#64748b" : "#52667a"}
      patternScale={2}
      patternDensity={2.0}
      pixelSizeJitter={0.5}
      enableRipples
      rippleSpeed={0.4}
      rippleThickness={0.12}
      rippleIntensityScale={1.5}
      liquid={false}
      speed={0}
      edgeFade={0}
      transparent
    />
  );
}

const mount = document.getElementById("pixel-blast-root");
if (mount) {
  createRoot(mount).render(<PixelBlastBackground />);
}
