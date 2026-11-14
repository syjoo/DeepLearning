const path = require("path");
const { C, FONT, IMG } = require("./lib");
// 그림 비율 (가로/세로)
const AR = { act_std: 7.2 / 4.2, act_hist: 7.4 / 4.6, xor: 7.6 / 3.4, activations: 11 / 2.6, gauss: 6.4 / 3.4, bn_ln_table: 8 / 3.2, bn_effect: 9 / 2.8, train_compare: 7.2 / 4.2 };
function img(s, name, x, y, w) { s.addImage({ path: path.join(IMG, name + ".png"), x, y, w, h: w / AR[name] }); return w / AR[name]; }
// 수식 박스 (유니코드 수식)
function F(d, s, t, { x, y, w, h = 0.6, size = 20, fill = C.gray }) {
  s.addText(t, { shape: d.pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08, fontFace: "Cambria Math", fontSize: size, color: C.ink, align: "center", valign: "middle", margin: 0.05 });
}
module.exports = { img, F, AR };
