const path = require("path");
const { C, FONT, IMG } = require("./lib");
// 그림 비율 (가로/세로)
const AR = { lr_effect: 10 / 2.9, valley: 10 / 3.2, ema: 10 / 3.2, step_size: 6 / 3.4, bias_correction: 6 / 3.2, adamw_decay: 6 / 3.2, schedules: 10 / 3.0, sharp_flat: 10 / 3.2, compare: 10 / 3.3 };
function img(s, name, x, y, w) { s.addImage({ path: path.join(IMG, name + ".png"), x, y, w, h: w / AR[name] }); return w / AR[name]; }
function F(d, s, t, { x, y, w, h = 0.6, size = 20, fill = C.gray }) {
  s.addText(t, { shape: d.pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08, fontFace: "Cambria Math", fontSize: size, color: C.ink, align: "center", valign: "middle", margin: 0.05 });
}
function table(s, head, rows, o) {
  const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
  const c = (t) => (typeof t === "string" ? { text: t, options: { align: "left" } } : t);
  s.addTable([head.map(H), ...rows.map(r => r.map(c))], { fontFace: FONT, color: C.ink, valign: "middle", border: { type: "solid", color: C.line, pt: 1 }, fill: { color: C.white }, ...o });
}
module.exports = { img, F, AR, table };
