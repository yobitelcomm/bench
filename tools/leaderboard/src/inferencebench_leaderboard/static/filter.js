// Tiny vanilla-JS text filter for .ib-table tables. No frameworks, no CDNs.
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    var table = document.querySelector("table.ib-table");
    if (!table || !table.tBodies.length) return;
    var tbody = table.tBodies[0];
    var input = document.querySelector("input.ib-filter");
    var count = document.querySelector(".ib-filter-count");
    if (!input) {
      var wrap = document.createElement("div");
      wrap.className = "ib-filter-wrap";
      input = document.createElement("input");
      input.type = "search";
      input.className = "ib-filter";
      input.placeholder = "Filter by model, engine, hardware...";
      count = document.createElement("span");
      count.className = "ib-filter-count";
      wrap.appendChild(input);
      wrap.appendChild(count);
      table.parentNode.insertBefore(wrap, table);
    }

    var rows = Array.prototype.slice.call(tbody.rows);
    var total = rows.length;
    var render = function (visible) {
      if (count) count.textContent = visible + " of " + total + " matching";
    };
    render(total);

    input.addEventListener("input", function () {
      var tokens = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      var visible = 0;
      for (var i = 0; i < rows.length; i++) {
        var text = rows[i].textContent.toLowerCase();
        var match = true;
        for (var j = 0; j < tokens.length; j++) {
          if (text.indexOf(tokens[j]) === -1) { match = false; break; }
        }
        rows[i].style.display = match ? "" : "none";
        if (match) visible++;
      }
      render(visible);
    });
  });
})();
