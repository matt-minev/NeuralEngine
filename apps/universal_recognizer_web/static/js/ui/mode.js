import { qs, qsa } from "../core/utils.js";

export class ModeHandler {
  constructor() {
    this.currentMode = "draw";
  }

  init() {
    const drawBtn = qs("#drawModeBtn");
    const testBtn = qs("#testModeBtn");

    if (drawBtn) {
      drawBtn.addEventListener("click", () => {
        this.switchMode("draw");
      });
    }

    if (testBtn) {
      testBtn.addEventListener("click", () => {
        this.switchMode("test");
      });
    }
  }

  switchMode(mode) {
    this.currentMode = mode;

    qsa(".mode-btn").forEach((btn) => {
      btn.classList.remove("active");
    });

    const activeButton = qs(`#${mode}ModeBtn`);
    if (activeButton) {
      activeButton.classList.add("active");
    }

    qsa(".mode-content").forEach((content) => {
      content.classList.remove("active");
    });

    const activeContent = qs(`#${mode}ModeContent`);
    if (activeContent) {
      activeContent.classList.add("active");
    }
  }
}
