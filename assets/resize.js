document.addEventListener('toggle', function(event) {
  if (event.target.tagName === 'DETAILS' && event.target.open) {
    window.requestAnimationFrame(function() { window.dispatchEvent(new Event('resize')); });
  }
}, true);

// Graphs initially mounted in hidden tabs need resizing when their container appears.
document.addEventListener('DOMContentLoaded', function() {
  let pending = false;
  const sizes = new ResizeObserver(function(entries) {
    if (!pending && entries.some(function(entry) { return entry.contentRect.width > 0; })) {
      pending = true;
      window.requestAnimationFrame(function() {
        pending = false;
        window.dispatchEvent(new Event('resize'));
      });
    }
  });
  function charts(node, action) {
    if (!(node instanceof HTMLElement)) return;
    if (node.matches('.chart')) action(node);
    node.querySelectorAll('.chart').forEach(action);
  }
  charts(document.body, function(chart) { sizes.observe(chart); });
  new MutationObserver(function(records) {
    records.forEach(function(record) {
      record.removedNodes.forEach(function(node) { charts(node, function(chart) { sizes.unobserve(chart); }); });
      record.addedNodes.forEach(function(node) { charts(node, function(chart) { sizes.observe(chart); }); });
    });
  }).observe(document.body, {childList: true, subtree: true});
});
