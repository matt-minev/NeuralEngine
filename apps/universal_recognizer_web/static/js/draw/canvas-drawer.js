export class CanvasDrawer {
  constructor({ canvas, overlay }) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.ctx = this.canvas.getContext("2d");

    this.isDrawing = false;
    this.brushSize = Math.max(12, Math.min(20, this.canvas.width / 20));
    this.strokes = [];
    this.currentStroke = null;
    this.lastX = 0;
    this.lastY = 0;

    this.setupCanvas();
    this.setupEvents();
  }

  setupCanvas() {
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  setupEvents() {
    this.canvas.addEventListener("mousedown", (event) => this.startDrawing(event));
    this.canvas.addEventListener("mousemove", (event) => this.draw(event));
    this.canvas.addEventListener("mouseup", () => this.stopDrawing());
    this.canvas.addEventListener("mouseout", () => this.stopDrawing());

    this.canvas.addEventListener("touchstart", (event) => {
      event.preventDefault();
      const touch = event.touches[0];
      const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.canvas.dispatchEvent(mouseEvent);
    });

    this.canvas.addEventListener("touchmove", (event) => {
      event.preventDefault();
      const touch = event.touches[0];
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.canvas.dispatchEvent(mouseEvent);
    });

    this.canvas.addEventListener("touchend", (event) => {
      event.preventDefault();
      this.canvas.dispatchEvent(new MouseEvent("mouseup", {}));
    });
  }

  startDrawing(event) {
    this.isDrawing = true;
    const rect = this.canvas.getBoundingClientRect();
    this.lastX = event.clientX - rect.left;
    this.lastY = event.clientY - rect.top;
    this.currentStroke = [{ x: this.lastX, y: this.lastY, t: Date.now() }];
    this.hideOverlay();
  }

  draw(event) {
    if (!this.isDrawing) {
      return;
    }

    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    this.ctx.strokeStyle = "#fff";
    this.ctx.lineWidth = this.brushSize;
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(x, y);
    this.ctx.stroke();

    this.lastX = x;
    this.lastY = y;

    if (this.currentStroke) {
      this.currentStroke.push({ x, y, t: Date.now() });
    }
  }

  stopDrawing() {
    if (this.currentStroke && this.currentStroke.length > 0) {
      this.strokes.push({ points: this.currentStroke });
    }
    this.currentStroke = null;
    this.isDrawing = false;
  }

  clear() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.showOverlay();
    this.strokes = [];
    this.currentStroke = null;
  }

  hasDrawing() {
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const data = imageData.data;
    for (let index = 0; index < data.length; index += 4) {
      if (data[index] > 0 || data[index + 1] > 0 || data[index + 2] > 0) {
        return true;
      }
    }
    return false;
  }

  getPayload() {
    return {
      input: {
        canvas: { width: this.canvas.width, height: this.canvas.height },
        strokes: this.strokes,
        raster: this.canvas.toDataURL("image/png"),
      },
    };
  }

  hideOverlay() {
    if (this.overlay) {
      this.overlay.classList.remove("show");
    }
  }

  showOverlay() {
    if (this.overlay) {
      this.overlay.classList.add("show");
    }
  }
}
