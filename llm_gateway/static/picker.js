"use strict";
// A nicer drop-down than the browser's native <select> menu (which looks out of place on phones).
//
// The <select> stays in the page, hidden, and remains the source of truth: code keeps reading and
// setting select.value and listening for "change". The picker only draws it:
//   closed:  a button with the short label   (<option data-short="default">)
//   open:    a list with the full label and an optional hint line (<option data-hint="...">);
//            a bottom sheet on phones, a small menu under the button on larger screens.

function enhanceSelect(sel, title) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "picker";
  btn.setAttribute("aria-haspopup", "listbox");
  sel.hidden = true;
  sel.after(btn);

  const sync = () => {
    const o = sel.selectedOptions[0];
    btn.innerHTML = `<span class="picker-label"></span><span class="picker-caret" aria-hidden="true">▾</span>`;
    btn.querySelector(".picker-label").textContent = o ? (o.dataset.short || o.textContent) : "—";
    btn.title = o ? o.textContent : "";
    btn.disabled = sel.disabled || !sel.options.length;
  };
  // Code sets sel.value directly (no "change" event), so hook the setter of this one element.
  const native = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value");
  Object.defineProperty(sel, "value", {
    get() { return native.get.call(this); },
    set(v) { native.set.call(this, v); sync(); },
  });
  new MutationObserver(sync).observe(sel, { childList: true, subtree: true, attributes: true, attributeFilter: ["disabled"] });
  sel.addEventListener("change", sync);
  btn.addEventListener("click", () => openPicker(sel, btn, title));
  sync();
}

function closePicker() {
  document.querySelector(".picker-layer")?.remove();
  document.removeEventListener("keydown", pickerKeys, true);
}
function pickerKeys(e) { if (e.key === "Escape") { e.stopPropagation(); closePicker(); } }

function openPicker(sel, btn, title) {
  closePicker();
  const phone = matchMedia("(max-width: 760px)").matches;
  const layer = document.createElement("div");
  layer.className = "picker-layer" + (phone ? " sheet" : "");
  const items = [...sel.options].map((o) => `
    <button type="button" class="picker-item${o.selected ? " on" : ""}" role="option" aria-selected="${o.selected}"
            data-value="${esc(o.value)}"${o.disabled ? " disabled" : ""}>
      <span class="pi-text"><span class="pi-label">${esc(o.textContent)}</span>${o.dataset.hint ? `<span class="pi-hint">${esc(o.dataset.hint)}</span>` : ""}</span>
      <span class="pi-check" aria-hidden="true">${o.selected ? "✓" : ""}</span>
    </button>`).join("");
  layer.innerHTML = `<div class="picker-scrim"></div>
    <div class="picker-menu" role="listbox" aria-label="${esc(title)}">
      ${phone ? `<div class="picker-title">${esc(title)}</div>` : ""}${items}</div>`;
  document.body.appendChild(layer);
  const menu = layer.querySelector(".picker-menu");
  if (!phone) {   // place under the button, kept inside the window
    const r = btn.getBoundingClientRect();
    menu.style.minWidth = r.width + "px";
    menu.style.top = r.bottom + 4 + "px";
    menu.style.left = Math.max(8, Math.min(r.left, innerWidth - menu.offsetWidth - 8)) + "px";
  }
  layer.querySelector(".picker-scrim").addEventListener("click", closePicker);
  menu.addEventListener("click", (e) => {
    const item = e.target.closest(".picker-item");
    if (!item || item.disabled) return;
    closePicker();
    if (sel.value !== item.dataset.value) {
      sel.value = item.dataset.value;
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    }
    btn.focus();
  });
  document.addEventListener("keydown", pickerKeys, true);
  (menu.querySelector(".picker-item.on") || menu.querySelector(".picker-item"))?.focus();
}
