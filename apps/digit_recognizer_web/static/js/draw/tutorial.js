export class TutorialOverlay {
  constructor() {
    this.storageKey = "digit_recognizer_tutorial_seen";
  }

  maybeShow() {
    if (window.localStorage.getItem(this.storageKey)) {
      return;
    }

    this.element = document.createElement("div");
    this.element.className = "tutorial-overlay show";
    this.element.innerHTML = `
      <div class="tutorial-card">
        <span class="section-kicker">Workspace Guide</span>
        <h2>Draw, compare, and inspect live predictions.</h2>
        <ul>
          <li>Draw on the canvas and the model will stream predictions as you go.</li>
          <li>Press <strong>C</strong> to clear, <strong>Esc</strong> to dismiss overlays, and <strong>0-9</strong> for quick digit flashes.</li>
          <li>Use the showcase page to inspect dataset samples and layer activations.</li>
        </ul>
        <button class="button button--primary" type="button">Enter Workspace</button>
      </div>
    `;
    this.element.querySelector("button").addEventListener("click", () => this.close());
    document.body.appendChild(this.element);
  }

  close() {
    if (!this.element) {
      return;
    }

    this.element.classList.remove("show");
    window.localStorage.setItem(this.storageKey, "true");
    window.setTimeout(() => {
      if (this.element) {
        this.element.remove();
      }
    }, 280);
  }
}
