import { qs } from "../core/utils.js";

export function setupToggle(buttonId, contentId) {
  const toggleBtn = qs(`#${buttonId}`);
  const content = qs(`#${contentId}`);

  if (!toggleBtn || !content) {
    return;
  }

  const icon = toggleBtn.querySelector(".toggle-icon");

  toggleBtn.addEventListener("click", () => {
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    if (icon) {
      icon.textContent = isHidden ? "▲" : "▼";
    }
  });
}
