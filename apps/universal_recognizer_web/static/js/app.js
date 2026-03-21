import { AccessibilityPanel } from "./accessibility/panel.js";
import { UniversalDrawApp } from "./draw/app.js";
import { TestModeHandler } from "./test/mode.js";
import { ModeHandler } from "./ui/mode.js";
import { setupToggle } from "./ui/toggles.js";

export function bootUniversalRecognizerApp() {
  const accessibilityPanel = new AccessibilityPanel();
  accessibilityPanel.clear();

  const drawApp = new UniversalDrawApp({ accessibilityPanel });
  const testMode = new TestModeHandler();
  const modeHandler = new ModeHandler();

  testMode.init();
  modeHandler.init();

  setupToggle("toggleAdvancedMetrics", "advancedMetricsContent");
  setupToggle("toggleDebugPanel", "debugPanelContent");

  return {
    accessibilityPanel,
    drawApp,
    testMode,
    modeHandler,
  };
}
