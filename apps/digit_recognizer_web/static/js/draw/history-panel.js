export class HistoryPanel {
  constructor(root) {
    this.root = root;
    this.items = [];
    this.maxSize = 10;
  }

  add(digit, confidence) {
    const lastItem = this.items[0];
    if (lastItem && lastItem.digit === digit && Math.abs(lastItem.confidence - confidence) < 0.1) {
      return;
    }

    this.items.unshift({ digit, confidence });
    if (this.items.length > this.maxSize) {
      this.items.pop();
    }
    this.render();
  }

  render() {
    this.root.innerHTML = this.items
      .map((item) => {
        let tone = "var(--red)";
        if (item.confidence > 80) {
          tone = "var(--green)";
        } else if (item.confidence > 60) {
          tone = "var(--amber)";
        }

        return `
          <div class="history-item">
            <strong style="color:${tone}; font-size:1.2rem;">${item.digit}</strong>
            <span>${item.confidence.toFixed(1)}%</span>
          </div>
        `;
      })
      .join("");
  }
}
