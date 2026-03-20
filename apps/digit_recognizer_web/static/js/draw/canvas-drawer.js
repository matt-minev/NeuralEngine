export class CanvasDrawer {
  constructor({ canvas, overlay, onStroke, onCommit }) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.onStroke = onStroke;
    this.onCommit = onCommit;
    this.ctx = this.canvas.getContext("2d");
    this.isDrawing = false;
    this.brushSize = 15;
    this.lastPoint = null;

    this.resetSurface();
    this.bindEvents();
  }

  bindEvents() {
    this.canvas.addEventListener("pointerdown", (event) => this.start(event));
    this.canvas.addEventListener("pointermove", (event) => this.move(event));
    this.canvas.addEventListener("pointerup", () => this.stop());
    this.canvas.addEventListener("pointerleave", () => this.stop());
    this.canvas.addEventListener("pointercancel", () => this.stop());
  }

  setBrushSize(size) {
    this.brushSize = size;
  }

  resetSurface() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.strokeStyle = "#fff";
    this.ctx.fillStyle = "#fff";
  }

  clear() {
    this.resetSurface();
    if (this.overlay) {
      this.overlay.classList.add("show");
    }
  }

  toDataURL() {
    return this.canvas.toDataURL("image/png");
  }

  start(event) {
    this.isDrawing = true;
    if (typeof this.canvas.setPointerCapture === "function") {
      this.canvas.setPointerCapture(event.pointerId);
    }
    const point = this.getPoint(event);
    this.lastPoint = point;
    if (this.overlay) {
      this.overlay.classList.remove("show");
    }

    this.ctx.beginPath();
    this.ctx.arc(point.x, point.y, this.brushSize / 2, 0, Math.PI * 2);
    this.ctx.fill();
    if (this.onStroke) {
      this.onStroke();
    }
  }

  move(event) {
    if (!this.isDrawing || !this.lastPoint) {
      return;
    }

    const point = this.getPoint(event);
    this.ctx.lineWidth = this.brushSize;
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
    this.ctx.lineTo(point.x, point.y);
    this.ctx.stroke();
    this.lastPoint = point;
    if (this.onStroke) {
      this.onStroke();
    }
  }

  stop() {
    if (!this.isDrawing) {
      return;
    }

    this.isDrawing = false;
    this.lastPoint = null;
    if (this.onCommit) {
      this.onCommit();
    }
  }

  getPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * this.canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * this.canvas.height,
    };
  }
}
