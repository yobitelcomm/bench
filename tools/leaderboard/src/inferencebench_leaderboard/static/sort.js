// Tiny vanilla-JS sorter for .ib-sortable tables. No frameworks.
(function () {
  "use strict";
  function cellValue(row, idx) {
    var c = row.cells[idx];
    if (!c) return "";
    if (c.dataset && c.dataset.value !== undefined) return c.dataset.value;
    return c.textContent.trim();
  }
  function compare(idx, asc) {
    return function (a, b) {
      var av = cellValue(a, idx);
      var bv = cellValue(b, idx);
      var an = parseFloat(av);
      var bn = parseFloat(bv);
      var bothNum = !isNaN(an) && !isNaN(bn) && av !== "" && bv !== "";
      var cmp = bothNum ? an - bn : av.localeCompare(bv);
      return asc ? cmp : -cmp;
    };
  }
  function install(table) {
    var headers = table.tHead ? table.tHead.rows[0].cells : [];
    for (var i = 0; i < headers.length; i++) {
      (function (idx) {
        var th = headers[idx];
        th.addEventListener("click", function () {
          var tbody = table.tBodies[0];
          var rows = Array.prototype.slice.call(tbody.rows);
          var asc = !th.classList.contains("ib-sort-asc");
          for (var j = 0; j < headers.length; j++) {
            headers[j].classList.remove("ib-sort-asc", "ib-sort-desc");
          }
          th.classList.add(asc ? "ib-sort-asc" : "ib-sort-desc");
          rows.sort(compare(idx, asc));
          for (var k = 0; k < rows.length; k++) tbody.appendChild(rows[k]);
        });
      })(i);
    }
  }
  document.querySelectorAll("table.ib-sortable").forEach(install);
})();
