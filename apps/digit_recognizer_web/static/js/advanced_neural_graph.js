/**
 * Advanced Interactive Neural Network Visualization
 * Features: Draggable nodes, animated connections, real-time values, educational animations
 */

class AdvancedNeuralGraph {
  constructor(containerId, architecture) {
    this.container = document.getElementById(containerId);
    this.architecture = architecture;
    this.width = this.container.clientWidth;
    this.height = this.container.clientHeight;

    // Animation state
    this.isAnimating = false;
    this.animationSpeed = 2000; // ms per layer
    this.particles = [];
    this.activationValues = [];

    // Visual elements
    this.neurons = [];
    this.connections = [];
    this.svg = null;
    this.simulation = null;

    this.init();
  }

  init() {
    this.createSVG();
    this.buildNeuralNetwork();
    this.setupInteractions();
    this.startParticleSystem();
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

    // Add definitions for gradients and filters
    this.setupDefinitions();
  }

  setupDefinitions() {
    const defs = this.svg.append("defs");

    // Gradient for connections
    const gradient = defs
      .append("linearGradient")
      .attr("id", "connectionGradient")
      .attr("gradientUnits", "userSpaceOnUse");

    gradient.append("stop").attr("offset", "0%").attr("stop-color", "#4facfe");

    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#00f2fe");

    // Glow filter for active neurons
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

    // Arrow marker for connections
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

    // Create neurons for each layer
    layers.forEach((layerSize, layerIndex) => {
      const layerNeurons = [];
      const maxVisible = Math.min(layerSize, 15); // Limit for visibility
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

      // Create connections to next layer
      if (layerIndex < layers.length - 1) {
        this.createConnections(layerNeurons, layerIndex);
      }
    });

    this.renderNetwork();
  }

  getNeuronRadius(layerIndex, totalLayers) {
    if (layerIndex === 0) return 8; // Input layer
    if (layerIndex === totalLayers - 1) return 12; // Output layer
    return 10; // Hidden layers
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

    // Create subset of connections to avoid visual clutter
    const connectionDensity = 0.3; // 30% of possible connections

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
    // Render connections first (behind neurons)
    this.renderConnections();

    // Render neurons
    this.renderNeurons();

    // Setup force simulation
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
      .attr("stroke", "rgba(255, 255, 255, 0.2)")
      .attr("stroke-width", 1)
      .attr("marker-end", "url(#arrowhead)")
      .style("opacity", 0.6);
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

    // Neuron labels (activation values)
    this.neuronElements
      .append("text")
      .attr("class", "neuron-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .attr("fill", "white")
      .attr("font-size", "10px")
      .attr("font-weight", "bold")
      .text((d) => d.activation.toFixed(2));

    // Neuron info on hover
    this.neuronElements
      .append("title")
      .text(
        (d) =>
          `Layer ${d.layer}, Neuron ${
            d.index
          }\nActivation: ${d.activation.toFixed(3)}`
      );
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
      .force("charge", d3.forceManyBody().strength(-50))
      .force("center", d3.forceCenter(this.width / 2, this.height / 2))
      .force(
        "collision",
        d3.forceCollide().radius((d) => d.radius + 5)
      )
      .force("x", d3.forceX((d) => d.originalX).strength(0.8))
      .force("y", d3.forceY((d) => d.originalY).strength(0.8))
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

    // Click behavior
    this.neuronElements.on("click", (event, d) => {
      this.highlightNeuron(d);
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

  // Enhanced neuron click handler with proper connection highlighting
  highlightNeuron(neuron) {
    // Reset all visual states first
    this.resetHighlights();

    // Highlight the selected neuron with glow effect
    this.neuronElements
      .filter((d) => d.id === neuron.id)
      .select("circle")
      .style("filter", "url(#glow)")
      .transition()
      .duration(300)
      .attr("r", (d) => d.radius * 1.5);

    // Find and highlight all connected neurons
    const connectedNeurons = this.getConnectedNeurons(neuron);

    // Highlight connected neurons
    this.neuronElements
      .filter((d) => connectedNeurons.includes(d.id))
      .select("circle")
      .style("filter", "url(#glow)")
      .style("opacity", 0.8);

    // Highlight connecting edges with animated flow
    this.highlightConnections(neuron);

    // Show detailed neuron information panel
    this.showNeuronInfoPanel(neuron, connectedNeurons);
  }

  getConnectedNeurons(neuron) {
    return this.connections
      .filter(
        (conn) => conn.source.id === neuron.id || conn.target.id === neuron.id
      )
      .map((conn) =>
        conn.source.id === neuron.id ? conn.target.id : conn.source.id
      );
  }

  highlightConnections(neuron) {
    const relevantConnections = this.connections.filter(
      (conn) => conn.source.id === neuron.id || conn.target.id === neuron.id
    );

    this.connectionElements
      .filter((d) => relevantConnections.includes(d))
      .transition()
      .duration(500)
      .attr("stroke", (d) => {
        // Color based on connection strength
        const strength = Math.abs(d.weight || 0.5);
        return d3.interpolateViridis(strength);
      })
      .attr("stroke-width", (d) => 2 + Math.abs(d.weight || 0.5) * 4)
      .attr("opacity", 1)
      .style("stroke-dasharray", "5,5")
      .style("stroke-dashoffset", 0)
      .transition()
      .duration(2000)
      .style("stroke-dashoffset", -20); // Animated flow effect
  }

  resetHighlights() {
    // Reset all neurons
    this.neuronElements
      .select("circle")
      .style("filter", "drop-shadow(0 0 8px rgba(255, 255, 255, 0.3))")
      .style("opacity", 1)
      .attr("r", (d) => d.radius);

    // Reset all connections
    this.connectionElements
      .attr("stroke", "rgba(255, 255, 255, 0.2)")
      .attr("stroke-width", 1)
      .attr("opacity", 0.6)
      .style("stroke-dasharray", "none");
  }

  highlightConnections(neuron) {
    this.connectionElements
      .style("opacity", (d) =>
        d.source.id === neuron.id || d.target.id === neuron.id ? 1 : 0.2
      )
      .attr("stroke-width", (d) =>
        d.source.id === neuron.id || d.target.id === neuron.id ? 3 : 1
      );
  }

  async animateForwardPass(activationData) {
    if (this.isAnimating) return;

    this.isAnimating = true;

    // Reset all activations
    this.neurons.forEach((neuron) => (neuron.activation = 0));

    // Animate each layer
    for (let layer = 0; layer < activationData.length; layer++) {
      await this.animateLayer(layer, activationData[layer]);
      await this.sleep(this.animationSpeed);
    }

    this.isAnimating = false;
  }

  async animateLayer(layerIndex, activations) {
    const layerNeurons = this.neurons.filter((n) => n.layer === layerIndex);

    // Update neuron activations
    layerNeurons.forEach((neuron, index) => {
      if (index < activations.length) {
        neuron.activation = activations[index];
      }
    });

    // Animate neuron appearance
    this.neuronElements
      .filter((d) => d.layer === layerIndex)
      .select("circle")
      .transition()
      .duration(500)
      .attr("r", (d) => d.radius + d.activation * 5)
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
      .duration(500)
      .tween("text", function (d) {
        const interpolate = d3.interpolate(0, d.activation);
        return function (t) {
          this.textContent = interpolate(t).toFixed(2);
        };
      });

    // Animate connections from previous layer
    if (layerIndex > 0) {
      this.animateConnections(layerIndex - 1, layerIndex);
    }

    // Create particle effects
    this.createParticleEffects(layerIndex);
  }

  animateConnections(fromLayer, toLayer) {
    const relevantConnections = this.connections.filter(
      (c) => c.source.layer === fromLayer && c.target.layer === toLayer
    );

    this.connectionElements
      .filter((d) => relevantConnections.includes(d))
      .transition()
      .duration(1000)
      .attr("stroke", "url(#connectionGradient)")
      .attr("stroke-width", 3)
      .style("opacity", 1)
      .transition()
      .duration(500)
      .attr("stroke", "rgba(255, 255, 255, 0.2)")
      .attr("stroke-width", 1);
  }

  createParticleEffects(layerIndex) {
    const layerNeurons = this.neurons.filter((n) => n.layer === layerIndex);

    layerNeurons.forEach((neuron) => {
      if (neuron.activation > 0.3) {
        this.createParticle(neuron);
      }
    });
  }

  createParticle(neuron) {
    const particle = this.svg
      .append("circle")
      .attr("class", "particle")
      .attr("cx", neuron.x)
      .attr("cy", neuron.y)
      .attr("r", 2)
      .attr("fill", "#ffffff")
      .style("opacity", 0.8);

    particle
      .transition()
      .duration(1000)
      .attr("r", 8)
      .style("opacity", 0)
      .remove();
  }

  startParticleSystem() {
    // Continuous ambient particles
    setInterval(() => {
      if (!this.isAnimating) {
        this.createAmbientParticles();
      }
    }, 2000);
  }

  createAmbientParticles() {
    const randomNeuron =
      this.neurons[Math.floor(Math.random() * this.neurons.length)];

    const particle = this.svg
      .append("circle")
      .attr("class", "ambient-particle")
      .attr("cx", randomNeuron.x)
      .attr("cy", randomNeuron.y)
      .attr("r", 1)
      .attr("fill", "rgba(255, 255, 255, 0.6)");

    particle
      .transition()
      .duration(3000)
      .attr("r", 6)
      .style("opacity", 0)
      .remove();
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Public API methods
  updateArchitecture(newArchitecture) {
    this.architecture = newArchitecture;
    this.buildNeuralNetwork();
  }

  setAnimationSpeed(speed) {
    this.animationSpeed = speed;
  }

  reset() {
    this.neurons.forEach((neuron) => {
      neuron.activation = 0;
      neuron.fx = null;
      neuron.fy = null;
    });

    this.neuronElements
      .select("circle")
      .attr("r", (d) => d.radius)
      .style("filter", "drop-shadow(0 0 8px rgba(255, 255, 255, 0.3))");

    this.neuronElements.select(".neuron-label").text("0.00");

    this.connectionElements
      .attr("stroke", "rgba(255, 255, 255, 0.2)")
      .attr("stroke-width", 1)
      .style("opacity", 0.6);
  }
}

// Export for use in other modules
window.AdvancedNeuralGraph = AdvancedNeuralGraph;
