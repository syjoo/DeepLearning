const path = require("path");
const { C, FONT, IMG } = require("./lib");
// 그림 비율 (가로/세로)
const AR = { perm_importance: 9 / 3.4, shap_student: 9.6 / 3.6, gradcam_toy: 12 / 2.6, gradcam_digits: 8.4 / 5.4, attention_heatmap: 11.2 / 4.2, fraud_case: 12 / 3.5 };
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
