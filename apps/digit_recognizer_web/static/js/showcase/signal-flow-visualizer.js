import { clamp, qs, sleep } from "../core/utils.js";

function layerMeta(index, totalLayers) {
  if (index === 0) {
    return { name: "Input", detail: "Pixel intensities" };
  }
  if (index === totalLayers - 1) {
    return { name: "Output", detail: "Digit classes" };
  }
  return { name: `Hidden ${index}`, detail: "Feature synthesis" };
}

export class SignalFlowVisualizer {
  constructor(container, inspector) {
    this.container = container;
    this.inspector = inspector;
    this.animationSpeed = 1600;
    this.architecture = null;
    this.nodes = [];
    this.lines = [];
    this.selectedNode = null;
    window.addEventListener("resize", () => this.redrawConnections());
  }

  updateArchitecture(architecture) {
    this.architecture = architecture;
    this.render();
  }

  setAnimationSpeed(speed) {
    this.animationSpeed = speed;
  }

  render() {
    if (!this.architecture) {
      return;
    }

    this.container.innerHTML = `
      <svg class="signal-flow__svg" viewBox="0 0 1000 620" preserveAspectRatio="none"></svg>
      <div class="signal-flow__layers"></div>
    `;

    const svg = qs(".signal-flow__svg", this.container);
    const layersRoot = qs(".signal-flow__layers", this.container);
    const layerSizes = [this.architecture.inputSize, ...this.architecture.hiddenLayers, this.architecture.outputSize];
    layersRoot.style.setProperty("--layer-count", layerSizes.length);

    this.nodes = [];
    layerSizes.forEach((size, layerIndex) => {
      const visibleCount = Math.min(layerIndex === 0 ? 12 : layerIndex === layerSizes.length - 1 ? 10 : 8, size);
      const layer = document.createElement("section");
      layer.className = "signal-layer";
      const meta = layerMeta(layerIndex, layerSizes.length);
      layer.innerHTML = `
        <header class="signal-layer__header">
          <span>${meta.name}</span>
          <strong>${size}</strong>
          <small>${meta.detail}</small>
        </header>
        <div class="signal-layer__nodes"></div>
      `;
      const nodesRoot = qs(".signal-layer__nodes", layer);

      for (let nodeIndex = 0; nodeIndex < visibleCount; nodeIndex += 1) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "signal-node";
        button.dataset.layer = String(layerIndex);
        button.dataset.index = String(nodeIndex);
        button.dataset.kind = layerIndex === 0 ? "input" : layerIndex === layerSizes.length - 1 ? "output" : "hidden";
        button.textContent = "0.00";
        button.addEventListener("click", () => this.selectNode(layerIndex, nodeIndex));
        nodesRoot.appendChild(button);
        this.nodes.push({
          element: button,
          layerIndex,
          nodeIndex,
          activation: 0,
          kind: button.dataset.kind,
        });
      }

      layersRoot.appendChild(layer);
    });

    this.lines = [];
    for (let layerIndex = 0; layerIndex < layerSizes.length - 1; layerIndex += 1) {
      const sourceNodes = this.nodes.filter((node) => node.layerIndex === layerIndex);
      const targetNodes = this.nodes.filter((node) => node.layerIndex === layerIndex + 1);

      sourceNodes.forEach((sourceNode, sourceIndex) => {
        targetNodes.forEach((targetNode, targetIndex) => {
          if (Math.abs(targetIndex - sourceIndex) <= 2 || targetIndex % 3 === sourceIndex % 3) {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("stroke", "rgba(103,232,249,0.12)");
            line.setAttribute("stroke-width", "1.6");
            line.setAttribute("stroke-linecap", "round");
            svg.appendChild(line);
            this.lines.push({ line, sourceNode, targetNode, layerIndex, length: 0 });
          }
        });
      });
    }

    this.redrawConnections();
    this.reset();
  }

  redrawConnections() {
    const svgRect = this.container.getBoundingClientRect();
    if (!svgRect.width || !svgRect.height) {
      return;
    }

    this.lines.forEach((connection) => {
      const { line, sourceNode, targetNode } = connection;
      const sourceRect = sourceNode.element.getBoundingClientRect();
      const targetRect = targetNode.element.getBoundingClientRect();
      const x1 = ((sourceRect.left + sourceRect.width / 2 - svgRect.left) / svgRect.width) * 1000;
      const y1 = ((sourceRect.top + sourceRect.height / 2 - svgRect.top) / svgRect.height) * 620;
      const x2 = ((targetRect.left + targetRect.width / 2 - svgRect.left) / svgRect.width) * 1000;
      const y2 = ((targetRect.top + targetRect.height / 2 - svgRect.top) / svgRect.height) * 620;
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      connection.length = Math.hypot(x2 - x1, y2 - y1);
    });
  }

  reset() {
    this.nodes.forEach((node) => {
      node.activation = 0;
      node.element.textContent = "0.00";
      node.element.style.background = "rgba(114, 146, 170, 0.12)";
      node.element.style.boxShadow = "none";
      node.element.classList.remove("is-selected");
    });

    this.lines.forEach(({ line }) => {
      line.setAttribute("stroke", "rgba(103,232,249,0.12)");
      line.setAttribute("stroke-width", "1.6");
      line.setAttribute("opacity", "1");
      line.style.strokeDasharray = "";
      line.style.strokeDashoffset = "";
    });

    this.selectedNode = null;
    this.inspector.innerHTML = "<p>Click a node in the signal-flow stage to inspect its role and activation.</p>";
  }

  selectNode(layerIndex, nodeIndex) {
    this.nodes.forEach((node) => node.element.classList.remove("is-selected"));
    const node = this.nodes.find((candidate) => candidate.layerIndex === layerIndex && candidate.nodeIndex === nodeIndex);
    if (!node) {
      return;
    }

    node.element.classList.add("is-selected");
    this.selectedNode = node;
    const meta = layerMeta(layerIndex, this.architecture.hiddenLayers.length + 2);
    this.inspector.innerHTML = `
      <p><strong>${meta.name} node ${nodeIndex + 1}</strong></p>
      <p>Role: ${meta.detail}</p>
      <p>Activation: <strong>${node.activation.toFixed(3)}</strong></p>
      <p>Layer size: ${[this.architecture.inputSize, ...this.architecture.hiddenLayers, this.architecture.outputSize][layerIndex]}</p>
      ${layerIndex === this.architecture.hiddenLayers.length + 1 ? `<p>Represents digit: <strong>${nodeIndex}</strong></p>` : ""}
    `;
  }

  setLayerActivations(layerIndex, activations) {
    const layerNodes = this.nodes.filter((node) => node.layerIndex === layerIndex);
    const outputLayerIndex = this.architecture.hiddenLayers.length + 1;

    layerNodes.forEach((node, index) => {
      const activation = clamp(activations[index] ?? 0, 0, 1);
      node.activation = activation;
      node.element.textContent = activation.toFixed(2);
      const hue = layerIndex === outputLayerIndex ? "255,180,84" : "103,232,249";
      node.element.style.background = `rgba(${hue}, ${0.18 + activation * 0.62})`;
      node.element.style.boxShadow = `0 0 ${6 + activation * 18}px rgba(${hue}, ${0.15 + activation * 0.45})`;
    });

    if (this.selectedNode && this.selectedNode.layerIndex === layerIndex) {
      this.selectNode(this.selectedNode.layerIndex, this.selectedNode.nodeIndex);
    }
  }

  pulseConnections(layerIndex) {
    const activeLines = this.lines.filter((item) => item.layerIndex === layerIndex);
    const duration = Math.max(520, this.animationSpeed * 0.82);

    const animations = activeLines.map(({ line, length }, index) => {
      const dashLength = length || 1;
      const leadBias = ((index % 5) - 2) * 0.06;
      const firstControl = Math.max(0.08, 0.2 + leadBias);
      const secondControl = Math.min(0.92, 0.78 - leadBias);
      line.setAttribute("stroke", "rgba(103,232,249,0.7)");
      line.setAttribute("stroke-width", "2.4");
      line.style.strokeDasharray = `${dashLength}`;
      line.style.strokeDashoffset = `${dashLength}`;
      const animation = line.animate(
        [
          { opacity: 0.12, strokeDashoffset: `${dashLength}` },
          { opacity: 0.42, strokeDashoffset: `${dashLength * (0.72 - leadBias * 0.45)}` },
          { opacity: 0.82, strokeDashoffset: `${dashLength * (0.28 + leadBias * 0.3)}` },
          { opacity: 0.9, strokeDashoffset: "0" },
        ],
        {
          duration,
          easing: `cubic-bezier(${firstControl}, 0.75, ${secondControl}, 1)`,
          fill: "forwards",
        }
      );

      return animation.finished.catch(() => undefined).then(() => {
        line.style.strokeDasharray = `${dashLength}`;
        line.style.strokeDashoffset = "0";
        line.setAttribute("opacity", "0.55");
      });
    });

    return Promise.all(animations);
  }

  async animateForwardPass(activationData, onStep) {
    this.reset();
    for (let layerIndex = 0; layerIndex < activationData.length; layerIndex += 1) {
      if (layerIndex > 0) {
        await this.pulseConnections(layerIndex - 1);
      }
      this.setLayerActivations(layerIndex, activationData[layerIndex]);
      if (onStep) {
        onStep(layerIndex);
      }
      await sleep(Math.max(260, this.animationSpeed * 0.28));
    }
  }
}
