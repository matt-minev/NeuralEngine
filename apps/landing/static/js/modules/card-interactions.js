function addRipple(event, element) {
  const ripple = document.createElement("span");
  const rect = element.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);

  ripple.className = "button__ripple";
  ripple.style.width = `${size}px`;
  ripple.style.height = `${size}px`;
  ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
  ripple.style.top = `${event.clientY - rect.top - size / 2}px`;

  element.appendChild(ripple);
  window.setTimeout(() => ripple.remove(), 560);
}

export function initializeCardInteractions() {
  const cards = Array.from(document.querySelectorAll("[data-app-card]"));
  const buttons = Array.from(document.querySelectorAll("[data-ripple]"));

  buttons.forEach((button) => {
    button.addEventListener("click", (event) => addRipple(event, button));
  });

  cards.forEach((card) => {
    card.addEventListener("mouseenter", () => {
      card.classList.add("is-hovered");
    });

    card.addEventListener("mouseleave", () => {
      card.classList.remove("is-hovered");
    });
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.animate(
            [
              { opacity: 0, transform: "translateY(24px)" },
              { opacity: 1, transform: "translateY(0)" },
            ],
            {
              duration: 520,
              easing: "cubic-bezier(0.22, 1, 0.36, 1)",
              fill: "forwards",
            }
          );
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.12,
      rootMargin: "0px 0px -40px 0px",
    }
  );

  cards.forEach((card) => observer.observe(card));
}
