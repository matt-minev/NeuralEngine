import { initializeAmbientMotion } from "./modules/ambient.js";
import { initializeCardInteractions } from "./modules/card-interactions.js";

document.addEventListener("DOMContentLoaded", () => {
  document.documentElement.style.scrollBehavior = "smooth";
  initializeAmbientMotion();
  initializeCardInteractions();
});
