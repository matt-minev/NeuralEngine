export function initializeAmbientMotion() {
  const glows = Array.from(document.querySelectorAll(".ambient-glow"));

  if (!glows.length) {
    return;
  }

  document.addEventListener("mousemove", (event) => {
    const offsetX = (event.clientX / window.innerWidth - 0.5) * 28;
    const offsetY = (event.clientY / window.innerHeight - 0.5) * 28;

    glows.forEach((glow, index) => {
      const direction = index % 2 === 0 ? 1 : -1;
      glow.style.transform = `translate(${offsetX * direction}px, ${offsetY * direction}px)`;
    });
  });
}
