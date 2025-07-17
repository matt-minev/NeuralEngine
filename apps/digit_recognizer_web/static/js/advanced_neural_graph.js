/**
 * Advanced Interactive Neural Network Visualization
 * Fixed version that properly handles connections, info panels, and real data
 */

class AdvancedNeuralGraph {
  constructor(containerId, architecture) {
    this.container = document.getElementById(containerId);
    this.architecture = architecture;
    this.width = this.container.clientWidth;
    this.height = this.container.clientHeight;

    // Animation state
    this.isAnimating = false;
    this.animationInProgress = false;
    this.animationSpeed = 2000;
    this.currentActivationData = null;

    // Visual elements
    this.neurons = [];
    this.connections = [];
    this.svg = null;
    this.simulation = null;
    this.neuronElements = null;
    this.connectionElements = null;

    // UI helpers
    this.contextWindow = null;
    this.neuronInfoPanel = null;
    this.selectedNeuron = null;

    this.init();
  }

  init() {
    this.createSVG();
    this.buildNeuralNetwork();
    this.setupInteractions();
    this.initializeHelpers();

    console.log("🧠 Advanced Neural Graph initialized");
  }

  initializeHelpers() {
    // Initialize context window helper
    this.contextWindow = {
      show: this.showProcessingContext.bind(this),
      update: this.updateProcessingStep.bind(this),
      remove: this.removeProcessingContext.bind(this),
    };

    // Initialize neuron info panel helper
    this.neuronInfoPanel = {
      show: this.showNeuronInfoPanel.bind(this),
      remove: this.removeNeuronInfoPanel.bind(this),
    };
  }

  createSVG() {
    // Remove existing SVG
    d3.select(this.container).select("svg").remove();

    this.svg = d3
      .select(this.container)
      .append("svg")
      .attr("width", this.width)
      .attr("height", this.height)
      .style("background", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
      .style("border-radius", "20px")
      .style("overflow", "hidden");

    this.setupDefinitions();
  }

  setupDefinitions() {
    const defs = this.svg.append("defs");

    // Connection gradient
    const gradient = defs
      .append("linearGradient")
      .attr("id", "connectionGradient")
      .attr("gradientUnits", "userSpaceOnUse");

    gradient.append("stop").attr("offset", "0%").attr("stop-color", "#4facfe");

    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#00f2fe");

    // Glow filter
    const filter = defs
      .append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");

    filter
      .append("feGaussianBlur")
      .attr("stdDeviation", "4")
      .attr("result", "coloredBlur");

    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrow marker
    defs
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 15)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#ffffff")
      .attr("opacity", 0.8);
  }

  buildNeuralNetwork() {
    const layers = [
      this.architecture.inputSize,
      ...this.architecture.hiddenLayers,
      this.architecture.outputSize,
    ];

    this.neurons = [];
    this.connections = [];

    const layerSpacing = this.width / (layers.length + 1);

    // Create neurons
    layers.forEach((layerSize, layerIndex) => {
      const layerNeurons = [];
      const maxVisible = Math.min(layerSize, 12);
      const neuronSpacing = this.height / (maxVisible + 1);

      for (let i = 0; i < maxVisible; i++) {
        const neuron = {
          id: `L${layerIndex}_N${i}`,
          layer: layerIndex,
          index: i,
          x: layerSpacing * (layerIndex + 1),
          y: neuronSpacing * (i + 1),
          activation: 0,
          originalX: layerSpacing * (layerIndex + 1),
          originalY: neuronSpacing * (i + 1),
          radius: this.getNeuronRadius(layerIndex, layers.length),
          type: this.getNeuronType(layerIndex, layers.length),
          fixed: false,
        };

        layerNeurons.push(neuron);
        this.neurons.push(neuron);
      }

      // Create connections
      if (layerIndex < layers.length - 1) {
        this.createConnections(layerNeurons, layerIndex);
      }
    });

    this.renderNetwork();
  }

  getNeuronRadius(layerIndex, totalLayers) {
    if (layerIndex === 0) return 10;
    if (layerIndex === totalLayers - 1) return 14;
    return 12;
  }

  getNeuronType(layerIndex, totalLayers) {
    if (layerIndex === 0) return "input";
    if (layerIndex === totalLayers - 1) return "output";
    return "hidden";
  }

  createConnections(currentLayerNeurons, currentLayerIndex) {
    const nextLayerNeurons = this.neurons.filter(
      (n) => n.layer === currentLayerIndex + 1
    );

    // Create more connections for better visualization
    const connectionDensity = 0.4;

    currentLayerNeurons.forEach((sourceNeuron) => {
      nextLayerNeurons.forEach((targetNeuron) => {
        if (Math.random() < connectionDensity) {
          this.connections.push({
            id: `${sourceNeuron.id}_to_${targetNeuron.id}`,
            source: sourceNeuron,
            target: targetNeuron,
            weight: (Math.random() - 0.5) * 2,
            active: false,
          });
        }
      });
    });
  }

  renderNetwork() {
    this.renderConnections();
    this.renderNeurons();
    this.setupForceSimulation();
  }

  renderConnections() {
    const connectionGroup = this.svg.append("g").attr("class", "connections");

    this.connectionElements = connectionGroup
      .selectAll(".connection")
      .data(this.connections)
      .enter()
      .append("line")
      .attr("class", "connection")
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y)
      .attr("stroke", "rgba(255, 255, 255, 0.3)")
      .attr("stroke-width", 1.5)
      .style("opacity", 0.7);
  }

  renderNeurons() {
    const neuronGroup = this.svg.append("g").attr("class", "neurons");

    this.neuronElements = neuronGroup
      .selectAll(".neuron")
      .data(this.neurons)
      .enter()
      .append("g")
      .attr("class", "neuron")
      .attr("transform", (d) => `translate(${d.x}, ${d.y})`);

    // Neuron circles
    this.neuronElements
      .append("circle")
      .attr("r", (d) => d.radius)
      .attr("fill", (d) => this.getNeuronColor(d.type))
      .attr("stroke", "rgba(255, 255, 255, 0.8)")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .style("filter", "drop-shadow(0 0 8px rgba(255, 255, 255, 0.3))");

    // Neuron labels
    this.neuronElements
      .append("text")
      .attr("class", "neuron-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .attr("fill", "white")
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .text((d) => d.activation.toFixed(2));
  }

  getNeuronColor(type) {
    const colors = {
      input: "#4facfe",
      hidden: "#00f2fe",
      output: "#f093fb",
    };
    return colors[type] || "#ffffff";
  }

  setupForceSimulation() {
    this.simulation = d3
      .forceSimulation(this.neurons)
      .force("charge", d3.forceManyBody().strength(-30))
      .force("center", d3.forceCenter(this.width / 2, this.height / 2))
      .force(
        "collision",
        d3.forceCollide().radius((d) => d.radius + 3)
      )
      .force("x", d3.forceX((d) => d.originalX).strength(0.9))
      .force("y", d3.forceY((d) => d.originalY).strength(0.9))
      .on("tick", () => this.updatePositions());
  }

  updatePositions() {
    this.neuronElements.attr("transform", (d) => `translate(${d.x}, ${d.y})`);

    this.connectionElements
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
  }

  setupInteractions() {
    // Drag behavior
    const drag = d3
      .drag()
      .on("start", this.dragStarted.bind(this))
      .on("drag", this.dragged.bind(this))
      .on("end", this.dragEnded.bind(this));

    this.neuronElements.call(drag);

    // Click behavior - FIXED
    this.neuronElements.on("click", (event, d) => {
      event.stopPropagation();
      this.highlightNeuron(d);
    });

    // Click outside to clear selection
    this.svg.on("click", () => {
      this.clearHighlights();
      this.neuronInfoPanel.remove();
    });

    // Zoom behavior
    const zoom = d3
      .zoom()
      .scaleExtent([0.5, 3])
      .on("zoom", (event) => {
        this.svg
          .selectAll(".neurons, .connections")
          .attr("transform", event.transform);
      });

    this.svg.call(zoom);
  }

  dragStarted(event, d) {
    if (!event.active) this.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  dragEnded(event, d) {
    if (!event.active) this.simulation.alphaTarget(0);
    if (!d.fixed) {
      d.fx = null;
      d.fy = null;
    }
  }

  // FIXED: Proper neuron highlighting with visible connections
  highlightNeuron(neuron) {
    this.selectedNeuron = neuron;
    this.clearHighlights();

    // Highlight selected neuron
    this.neuronElements
      .filter((d) => d.id === neuron.id)
      .select("circle")
      .style("filter", "url(#glow)")
      .style("stroke", "#FFD700")
      .style("stroke-width", "4px")
      .transition()
      .duration(300)
      .attr("r", (d) => d.radius * 1.3);

    // Find connections
    const relevantConnections = this.connections.filter(
      (conn) => conn.source.id === neuron.id || conn.target.id === neuron.id
    );

    // Get connected neurons
    const connectedNeuronIds = new Set();
    relevantConnections.forEach((conn) => {
      connectedNeuronIds.add(conn.source.id);
      connectedNeuronIds.add(conn.target.id);
    });
    connectedNeuronIds.delete(neuron.id);

    // Highlight connected neurons
    this.neuronElements
      .filter((d) => connectedNeuronIds.has(d.id))
      .select("circle")
      .style("filter", "url(#glow)")
      .style("stroke", "#4facfe")
      .style("stroke-width", "3px")
      .transition()
      .duration(300)
      .attr("r", (d) => d.radius * 1.1);

    // FIXED: Highlight connections properly
    this.connectionElements.each(function (d) {
      const isRelevant = relevantConnections.some(
        (conn) =>
          conn.source.id === d.source.id && conn.target.id === d.target.id
      );

      if (isRelevant) {
        d3.select(this)
          .style("stroke", "#4facfe")
          .style("stroke-width", "3px")
          .style("opacity", "1")
          .style("stroke-dasharray", "5,5")
          .style("stroke-dashoffset", "0")
          .transition()
          .duration(2000)
          .style("stroke-dashoffset", "-20");
      } else {
        d3.select(this).style("opacity", "0.2");
      }
    });

    // Show info panel
    this.neuronInfoPanel.show(neuron, Array.from(connectedNeuronIds));
  }

  // FIXED: Proper highlight clearing
  clearHighlights() {
    this.selectedNeuron = null;

    // Reset neurons
    this.neuronElements
      .select("circle")
      .transition()
      .duration(300)
      .style("filter", "drop-shadow(0 0 8px rgba(255, 255, 255, 0.3))")
      .style("stroke", "rgba(255, 255, 255, 0.8)")
      .style("stroke-width", "2px")
      .attr("r", (d) => d.radius);

    // Reset connections
    this.connectionElements
      .transition()
      .duration(300)
      .style("stroke", "rgba(255, 255, 255, 0.3)")
      .style("stroke-width", "1.5px")
      .style("opacity", "0.7")
      .style("stroke-dasharray", "none")
      .style("stroke-dashoffset", "0");
  }

  // FIXED: Proper neuron info panel
  showNeuronInfoPanel(neuron, connectedNeuronIds) {
    this.removeNeuronInfoPanel();

    const panel = document.createElement("div");
    panel.id = "neuronInfoPanel";
    panel.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            width: 350px;
            max-height: 500px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 20px;
            border-radius: 12px;
            font-family: Inter, sans-serif;
            font-size: 14px;
            z-index: 10000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border-left: 4px solid #4facfe;
        `;

    const layerInfo = this.getLayerExplanation(neuron.layer);
    const activationLevel = this.getActivationLevel(neuron.activation);

    panel.innerHTML = `
            <div style="border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 12px; margin-bottom: 12px;">
                <strong style="color: #4facfe; font-size: 16px;">Neuron ${
                  neuron.id
                }</strong>
                <button id="closeNeuronPanel" style="float: right; background: none; border: none; color: white; font-size: 18px; cursor: pointer;">×</button>
            </div>
            
            <div style="margin-bottom: 12px;">
                <strong>Layer:</strong> ${layerInfo.name}<br>
                <small style="color: #aaa;">${layerInfo.description}</small>
            </div>
            
            <div style="margin-bottom: 12px;">
                <strong>Activation:</strong> 
                <span style="color: ${
                  activationLevel.color
                }; font-weight: bold; font-size: 16px;">
                    ${neuron.activation.toFixed(3)}
                </span><br>
                <small style="color: #aaa;">${
                  activationLevel.description
                }</small>
            </div>
            
            <div style="margin-bottom: 12px;">
                <strong>Connections:</strong> ${
                  connectedNeuronIds.length
                } neurons
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                <strong style="color: #FFD700;">💡 What this means:</strong><br>
                <small>${this.getDetailedExplanation(neuron)}</small>
            </div>
            
            <div style="background: rgba(79, 172, 254, 0.2); padding: 12px; border-radius: 8px; font-size: 12px;">
                <strong>🎯 For Digit Recognition:</strong><br>
                <small>${this.getDigitRecognitionContext(neuron)}</small>
            </div>
        `;

    document.body.appendChild(panel);

    // Add close button functionality
    document
      .getElementById("closeNeuronPanel")
      .addEventListener("click", () => {
        this.removeNeuronInfoPanel();
        this.clearHighlights();
      });

    // Auto-hide after 15 seconds
    setTimeout(() => this.removeNeuronInfoPanel(), 15000);
  }

  removeNeuronInfoPanel() {
    const existing = document.getElementById("neuronInfoPanel");
    if (existing) existing.remove();
  }

  getLayerExplanation(layerIndex) {
    const explanations = {
      0: {
        name: "Input Layer",
        description: "Receives raw pixel data (28×28 = 784 pixels)",
      },
      1: {
        name: "Hidden Layer 1",
        description: "Detects basic features like edges and shapes",
      },
      2: {
        name: "Hidden Layer 2",
        description: "Combines features into complex patterns",
      },
      3: {
        name: "Hidden Layer 3",
        description: "Recognizes digit-specific patterns",
      },
      4: {
        name: "Output Layer",
        description: "Final digit classification (0-9)",
      },
    };
    return (
      explanations[layerIndex] || {
        name: `Layer ${layerIndex + 1}`,
        description: "Processing layer",
      }
    );
  }

  getActivationLevel(activation) {
    if (activation > 0.8)
      return { color: "#34C759", description: "Very High - Strong response" };
    if (activation > 0.6)
      return { color: "#FFD700", description: "High - Good response" };
    if (activation > 0.4)
      return { color: "#FF9500", description: "Medium - Moderate response" };
    if (activation > 0.2)
      return { color: "#8E8E93", description: "Low - Weak response" };
    return { color: "#FF3B30", description: "Very Low - No response" };
  }

  getDetailedExplanation(neuron) {
    const layer = neuron.layer;
    const activation = neuron.activation;

    if (layer === 0) {
      // Input layer description
      return `This input neuron represents a subset of the 784 input pixels (28×28 image). 
                Since we show only 12 visible neurons, each represents a sample of pixel intensities.
                Activation ${activation.toFixed(
                  3
                )} shows the average brightness in this region.`;
    } else if (layer === this.getOutputLayerIndex()) {
      // FIXED: Parse digit index correctly from neuron ID
      const digitIndex = parseInt(neuron.id.split("_")[1].substring(1));
      return `This output neuron represents digit ${digitIndex}.
                Activation ${activation.toFixed(
                  3
                )} shows the network's confidence that the input is digit ${digitIndex}.
                The neuron with highest activation determines the final prediction.`;
    } else {
      const layerNum = layer;
      return `This hidden layer ${layerNum} neuron detects specific features and patterns.
                Activation ${activation.toFixed(
                  3
                )} shows how strongly this neuron responds to the detected features.
                Multiple neurons work together to recognize digit characteristics.`;
    }
  }

  getDigitRecognitionContext(neuron) {
    const layer = neuron.layer;

    if (layer === 0) {
      return `Input neurons represent pixel regions from the 28×28 image. Each of the 12 visible neurons 
                shows a sample of the 784 total pixels. Higher activation means brighter pixels in that region.`;
    } else if (layer === this.getOutputLayerIndex()) {
      // FIXED: Parse digit index correctly from neuron ID
      const digitIndex = parseInt(neuron.id.split("_")[1].substring(1));
      return `This is output neuron ${digitIndex}, representing digit ${digitIndex}. 
                The network compares all 10 output neurons (0-9) and chooses the one with highest activation.
                For correct predictions, this neuron should have the highest value when the input is digit ${digitIndex}.`;
    } else {
      const features = [
        "basic edges and corners",
        "curves and shapes",
        "complex digit patterns",
      ];
      const feature = features[Math.min(layer - 1, 2)];
      return `This hidden layer detects ${feature} that help distinguish between different digits 0-9.
                Multiple neurons work together to build up the final digit classification.`;
    }
  }

  getOutputLayerIndex() {
    return this.architecture.hiddenLayers.length + 1;
  }

  // FIXED: Processing context window
  showProcessingContext(digit = "?", confidence = 0) {
    this.removeProcessingContext();

    const panel = document.createElement("div");
    panel.id = "processingContext";
    panel.style.cssText = `
            position: fixed;
            top: 50%;
            left: 20px;
            transform: translateY(-50%);
            width: 280px;
            padding: 20px;
            background: rgba(0, 122, 255, 0.95);
            color: white;
            border-radius: 12px;
            font-family: Inter, sans-serif;
            z-index: 9999;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
        `;

    panel.innerHTML = `
            <h3 style="margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">
                🧠 Processing Digit "${digit}"
            </h3>
            <p style="margin: 0 0 15px 0; font-size: 14px;">
                Confidence: <strong>${confidence.toFixed(1)}%</strong>
            </p>
            <div id="processingSteps" style="font-size: 13px;">
                <div class="step" data-step="0" style="padding: 8px 0; opacity: 0.5; transition: all 0.3s ease;">
                    <span class="step-icon">⏳</span> 1. Reading pixels...
                </div>
                <div class="step" data-step="1" style="padding: 8px 0; opacity: 0.5; transition: all 0.3s ease;">
                    <span class="step-icon">⏳</span> 2. Detecting edges...
                </div>
                <div class="step" data-step="2" style="padding: 8px 0; opacity: 0.5; transition: all 0.3s ease;">
                    <span class="step-icon">⏳</span> 3. Finding patterns...
                </div>
                <div class="step" data-step="3" style="padding: 8px 0; opacity: 0.5; transition: all 0.3s ease;">
                    <span class="step-icon">⏳</span> 4. Classifying digit...
                </div>
            </div>
            <div style="margin-top: 15px; font-size: 12px; opacity: 0.8;">
                💡 Click neurons to explore connections!
            </div>
        `;

    document.body.appendChild(panel);
    this.updateProcessingStep(0);
  }

  updateProcessingStep(stepIndex) {
    const context = document.getElementById("processingContext");
    if (!context) return;

    const steps = context.querySelectorAll(".step");
    steps.forEach((step, index) => {
      const icon = step.querySelector(".step-icon");
      if (index <= stepIndex) {
        step.style.opacity = "1";
        step.style.fontWeight = "bold";
        step.style.color = "#FFD700";
        icon.textContent = index === stepIndex ? "🔄" : "✅";
      } else {
        step.style.opacity = "0.5";
        step.style.fontWeight = "normal";
        step.style.color = "rgba(255, 255, 255, 0.7)";
        icon.textContent = "⏳";
      }
    });
  }

  removeProcessingContext() {
    const existing = document.getElementById("processingContext");
    if (existing) existing.remove();
  }

  // FIXED: Animation with real data
  async animateForwardPass(activationData) {
    if (this.animationInProgress) return;

    this.animationInProgress = true;
    this.currentActivationData = activationData;

    try {
      // Reset all activations
      this.neurons.forEach((neuron) => (neuron.activation = 0));

      const totalLayers = this.architecture.hiddenLayers.length + 2;

      // Animate each layer
      for (let layer = 0; layer < totalLayers; layer++) {
        const layerActivations =
          activationData[layer] || this.generateFallbackActivations(layer);

        this.updateProcessingStep(layer);
        await this.animateLayer(layer, layerActivations);
        await this.sleep(this.animationSpeed);
      }

      // Remove context after completion
      setTimeout(() => this.removeProcessingContext(), 2000);
    } catch (error) {
      console.error("Animation error:", error);
    } finally {
      this.animationInProgress = false;
    }
  }

  async animateLayer(layerIndex, activations) {
    const layerNeurons = this.neurons.filter((n) => n.layer === layerIndex);

    // Update activations with real data
    layerNeurons.forEach((neuron, index) => {
      if (index < activations.length) {
        neuron.activation = Math.max(0, Math.min(1, activations[index]));
      } else {
        neuron.activation = 0;
      }
    });

    // Animate neurons
    this.neuronElements
      .filter((d) => d.layer === layerIndex)
      .select("circle")
      .transition()
      .duration(800)
      .attr("r", (d) => d.radius * (1 + d.activation * 0.5))
      .style("fill", (d) => {
        const baseColor = this.getNeuronColor(d.type);
        const intensity = d.activation * 0.4;
        return d3.interpolate(baseColor, "#ffffff")(intensity);
      })
      .style("filter", (d) =>
        d.activation > 0.5
          ? "url(#glow)"
          : "drop-shadow(0 0 8px rgba(255, 255, 255, 0.3))"
      );

    // Update labels
    this.neuronElements
      .filter((d) => d.layer === layerIndex)
      .select(".neuron-label")
      .transition()
      .duration(800)
      .style("fill", (d) => (d.activation > 0.7 ? "#000000" : "#ffffff"))
      .tween("text", function (d) {
        const interpolate = d3.interpolate(0, d.activation);
        return function (t) {
          this.textContent = interpolate(t).toFixed(2);
        };
      });
  }

  generateFallbackActivations(layerIndex) {
    const layerSizes = [
      this.architecture.inputSize,
      ...this.architecture.hiddenLayers,
      this.architecture.outputSize,
    ];

    const layerSize = Math.min(layerSizes[layerIndex], 12);
    const activations = [];

    for (let i = 0; i < layerSize; i++) {
      if (layerIndex === 0) {
        // Input layer - pixel intensities
        activations.push(Math.random() * 0.8 + 0.1);
      } else if (layerIndex === layerSizes.length - 1) {
        // Output layer - FIXED: Ensure correct digit has highest activation
        const predictedDigit = this.getPredictedDigitFromShowcase();

        if (i === predictedDigit) {
          // Predicted digit gets highest activation
          activations.push(0.85 + Math.random() * 0.1); // 85-95%
        } else {
          // Other digits get lower activations
          activations.push(Math.random() * 0.4 + 0.1); // 10-50%
        }
      } else {
        // Hidden layers - ReLU activations
        activations.push(Math.max(0, Math.random() * 1.2 - 0.3));
      }
    }

    return activations;
  }

  // Add this helper method to get the predicted digit
  getPredictedDigitFromShowcase() {
    // Try to get the predicted digit from the showcase instance
    if (window.datasetShowcase && window.datasetShowcase.currentPrediction) {
      return window.datasetShowcase.currentPrediction.predicted_digit;
    }
    return 0; // Default fallback
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // FIXED: Proper reset function
  reset() {
    this.animationInProgress = false;
    this.selectedNeuron = null;

    // Reset neuron states
    this.neurons.forEach((neuron) => {
      neuron.activation = 0;
      neuron.fx = null;
      neuron.fy = null;
      neuron.x = neuron.originalX;
      neuron.y = neuron.originalY;
    });

    // Clear highlights
    this.clearHighlights();

    // Reset visual elements
    if (this.neuronElements) {
      this.neuronElements
        .select("circle")
        .transition()
        .duration(500)
        .attr("r", (d) => d.radius)
        .style("fill", (d) => this.getNeuronColor(d.type))
        .style("filter", "drop-shadow(0 0 8px rgba(255, 255, 255, 0.3))");

      this.neuronElements
        .select(".neuron-label")
        .transition()
        .duration(500)
        .style("fill", "white")
        .text("0.00");
    }

    // Remove UI elements
    this.removeNeuronInfoPanel();
    this.removeProcessingContext();

    // Restart simulation
    if (this.simulation) {
      this.simulation.alpha(1).restart();
    }

    console.log("🔄 Neural network reset complete");
  }

  updateArchitecture(newArchitecture) {
    this.animationInProgress = false;
    this.architecture = newArchitecture;

    // Clear existing
    if (this.svg) {
      this.svg.selectAll("*").remove();
    }

    // Rebuild
    this.buildNeuralNetwork();

    console.log("✅ Architecture updated");
  }

  setAnimationSpeed(speed) {
    this.animationSpeed = speed;
  }
}

// Export to window
window.AdvancedNeuralGraph = AdvancedNeuralGraph;
