// Navigation management
const Navigation = {
  init() {
    // Set up navigation event listeners
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        // If it's an external link, allow default browser navigation
        if (link.classList.contains("external-link")) {
          return; // Let the browser handle the navigation
        }

        // For tab navigation, prevent default and handle internally
        e.preventDefault();
        const section = link.dataset.section;
        if (section) {
          this.showSection(section);
        }
      });
    });

    // Check for an explicit URL hash first, then default to dashboard.
    // Do not restore the last tab from localStorage on a fresh open, because
    // launch-page opens should always land on the dashboard.
    const hashSection = window.location.hash.substring(1); // Remove #
    const initialSection =
      hashSection && hashSection !== "dashboard" ? hashSection : "dashboard";

    // Validate section exists
    const validSections = [
      "dashboard",
      "data",
      "training",
      "model-management",
      "prediction",
      "analysis",
      "comparison",
    ];
    const sectionToShow = validSections.includes(initialSection)
      ? initialSection
      : "dashboard";

    // Prevent scroll if it's dashboard
    if (
      sectionToShow === "dashboard" &&
      window.location.hash === "#dashboard"
    ) {
      // Remove hash immediately to prevent browser scroll
      window.history.replaceState(null, null, window.location.pathname);
    }

    this.showSection(sectionToShow);

    // Listen for hash changes (browser back/forward)
    window.addEventListener("hashchange", () => {
      const hashSection = window.location.hash.substring(1);
      if (
        hashSection &&
        hashSection !== "dashboard" &&
        validSections.includes(hashSection)
      ) {
        this.showSection(hashSection);
      } else if (hashSection === "dashboard") {
        // Remove dashboard hash to prevent scrolling
        window.history.replaceState(null, null, window.location.pathname);
        this.showSection("dashboard");
      }
    });
  },

  showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll(".content-section").forEach((section) => {
      section.classList.remove("active");
    });

    // Show selected section
    const section = document.getElementById(sectionId);
    if (section) {
      section.classList.add("active");
      AppState.currentSection = sectionId;
    }

    // Update navigation active state
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.classList.remove("active");
    });

    const activeLink = document.querySelector(`[data-section="${sectionId}"]`);
    if (activeLink) {
      activeLink.classList.add("active");
    }

    // Save to localStorage for persistence across refreshes
    localStorage.setItem("quadratic_web_active_tab", sectionId);

    // Update URL hash for deep linking (without triggering page reload)
    // Special handling for dashboard: remove hash to prevent scrolling
    if (sectionId === "dashboard") {
      if (window.location.hash) {
        window.history.replaceState(null, null, window.location.pathname);
      }
    } else {
      if (window.location.hash !== `#${sectionId}`) {
        window.history.replaceState(null, null, `#${sectionId}`);
      }
    }

    // Section-specific initialization
    switch (sectionId) {
      case "data":
        DataSection.refresh();
        break;
      case "training":
        TrainingSection.refresh();
        break;
      case "model-management":
        ModelSection.refresh();
        ModelSection.refreshAppState();
        break;
      case "prediction":
        PredictionSection.refresh();
        break;
      case "analysis":
        AnalysisSection.refresh();
        break;
      case "comparison":
        ComparisonSection.refresh();
        break;
    }
  },
};

// Data section management
