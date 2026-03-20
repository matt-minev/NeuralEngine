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
    this.strokes = [];
    this.currentStroke = null;

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
    this.strokes = [];
    this.currentStroke = null;
    if (this.overlay) {
      this.overlay.classList.add("show");
    }
  }

  toDataURL() {
    return this.canvas.toDataURL("image/png");
  }

  hasInk() {
    return this.strokes.length > 0 || Boolean(this.currentStroke && this.currentStroke.points.length > 0);
  }

  getPredictionPayload() {
    const allStrokes = this.currentStroke && this.currentStroke.points.length > 0
      ? [...this.strokes, this.currentStroke]
      : this.strokes;

    return {
      strokes: allStrokes.map((stroke) => ({
        points: stroke.points.map((point) => ({
          x: point.x,
          y: point.y,
        })),
      })),
      raster: this.toDataURL(),
      canvas: {
        width: this.canvas.width,
        height: this.canvas.height,
      },
    };
  }

  start(event) {
    this.isDrawing = true;
    if (typeof this.canvas.setPointerCapture === "function") {
      this.canvas.setPointerCapture(event.pointerId);
    }
    const point = this.getPoint(event);
    this.lastPoint = point;
    this.currentStroke = { points: [] };
    this.addPointToStroke(point);
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
    this.addPointToStroke(point);
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
    if (this.currentStroke && this.currentStroke.points.length > 0) {
      this.strokes.push(this.currentStroke);
    }
    this.currentStroke = null;
    if (this.onCommit) {
      this.onCommit();
    }
  }

  addPointToStroke(point) {
    if (!this.currentStroke) {
      return;
    }

    this.currentStroke.points.push({
      x: point.x / this.canvas.width,
      y: point.y / this.canvas.height,
    });
  }

  getPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * this.canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * this.canvas.height,
    };
  }
}
