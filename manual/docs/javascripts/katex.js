// Math rendering for the LORRAX manual. Shared macros live here; add new ones only with the maintainer's agreement.
window.LX_MACROS = {
  "\\bk": "\\mathbf{k}",
  "\\bq": "\\mathbf{q}",
  "\\br": "\\mathbf{r}",
  "\\bG": "\\mathbf{G}",
  "\\bR": "\\mathbf{R}",
  "\\bQ": "\\mathbf{Q}"
};
(function () {
  function render(root) {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
      macros: Object.assign({}, window.LX_MACROS),
      throwOnError: false
    });
  }
  if (window.document$) window.document$.subscribe(({ body }) => render(body));
  else document.addEventListener("DOMContentLoaded", () => render(document.body));
})();
