const COLORS = ["#67e8f9", "#ffb454", "#6ee7a1", "#ff7f8d", "#9b8cff"];

export function createParticles(target) {
  let container = document.querySelector(".particles-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "particles-container";
    document.body.appendChild(container);
  }

  const rect = target.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;

  for (let index = 0; index < 18; index += 1) {
    const particle = document.createElement("div");
    particle.className = "particle";
    particle.style.left = `${centerX}px`;
    particle.style.top = `${centerY}px`;
    particle.style.background = COLORS[index % COLORS.length];
    particle.style.setProperty(
      "--burst-transform",
      `translate(${(Math.random() - 0.5) * 150}px, ${(Math.random() - 0.5) * 150}px)`
    );
    container.appendChild(particle);
    window.setTimeout(() => particle.remove(), 1900);
  }
}

export function triggerPiCelebration() {
  const overlay = document.getElementById("piSymbol");
  const confettiContainer = document.getElementById("confettiContainer");

  if (overlay) {
    overlay.classList.add("show");
  }

  for (let index = 0; index < 48; index += 1) {
    const confetti = document.createElement("div");
    confetti.className = "confetti";
    confetti.style.left = `${Math.random() * 100}%`;
    confetti.style.background = COLORS[index % COLORS.length];
    confetti.style.animationDelay = `${Math.random() * 0.4}s`;
    if (confettiContainer) {
      confettiContainer.appendChild(confetti);
    }
    window.setTimeout(() => confetti.remove(), 3200);
  }

  window.setTimeout(() => {
    if (overlay) {
      overlay.classList.remove("show");
    }
  }, 2800);
}
