document.addEventListener('toggle', function(event) {
  if (event.target.tagName === 'DETAILS' && event.target.open) {
    window.requestAnimationFrame(function() { window.dispatchEvent(new Event('resize')); });
  }
}, true);
