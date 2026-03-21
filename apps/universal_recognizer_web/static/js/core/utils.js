export function qs(selector, root = document) {
  return root.querySelector(selector);
}

export function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

export function createNotification(message, type = "info") {
  const notification = document.createElement("div");
  notification.className = `test-notification test-notification-${type}`;
  notification.textContent = message;
  return notification;
}

export function preserveScroll(callback) {
  const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
  return Promise.resolve(callback()).finally(() => {
    requestAnimationFrame(() => {
      window.scrollTo(0, scrollPosition);
    });
  });
}
