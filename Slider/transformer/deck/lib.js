// 공통 헬퍼: 색상, 텍스트, 카드, 코드 박스, 도식 부품
const path = require("path");
const IMG = path.join(__dirname, "..", "images");

const C = {
  ink: "1F2933", muted: "5B6770", line: "CBD5DC",
  teal: "0E7490", tealL: "E0F2F6",
  orange: "C2410C", orangeL: "FFEDD9",
  purple: "6D28D9", purpleL: "EDE7FB",
  green: "15803D", greenL: "E3F5E8",
  red: "B91C1C", redL: "FDE8E8",
  gray: "F3F4F6", white: "FFFFFF", paper: "F5FAFB",
  cmt: "6B7B73",
};
const FONT = "Malgun Gothic";
const MONO = "Consolas";

// ---- 텍스트 높이 추정 (한글은 넓게) : 넘치면 경고를 찍는다 ----
function widthUnits(s) {
  let u = 0;
  for (const ch of s) u += /[ㄱ-힝①-⑳←-⇿∀-⋿─-◿]/.test(ch) ? 1.0 : (ch === " " ? 0.3 : 0.58);
  return u;
}
function estHeight(lines, size, wIn, lh = 1.3) {
  let n = 0;
  for (const l of lines) n += Math.max(1, Math.ceil((widthUnits(l) * size) / (wIn * 72)));
  return (n * size * lh) / 72;
}

// 영역에 맞는 가장 큰 글자 크기를 찾는다 (min ~ max)
function fitSize(lines, wIn, hIn, min, max, extra = 0, fill = 0.85) {
  for (let sz = max; sz > min; sz -= 0.5) {
    if (estHeight(lines, sz, wIn) + extra * (sz / 14) <= hIn * fill) return sz;
  }
  return min;
}

class Deck {
  constructor(pres, title) {
    this.pres = pres; this.n = 0; this.title = title; this.warns = [];
  }
  warn(msg) { this.warns.push(`slide ${this.n}: ${msg}`); }

  // ---- 새 슬라이드 (제목 + 푸터) ----
  slide(title, { tag = "", notes = "", bg = C.white } = {}) {
    const s = this.pres.addSlide();
    this.n += 1;
    s.background = { color: bg };
    if (tag) s.addText(tag, { x: 0.6, y: 0.3, w: 9, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: C.teal, margin: 0 });
    s.addText(title, { x: 0.6, y: 0.58, w: 12.1, h: 0.7, fontFace: FONT, fontSize: 28, bold: true, color: C.ink, margin: 0, valign: "middle" });
    if (widthUnits(title) * 28 > 12.1 * 72) this.warn(`제목이 길다: ${title}`);
    s.addImage({ path: path.join(IMG, "logo.png"), x: 0.6, y: 7.0, w: 1.4, h: 1.4 * 118 / 756 });
    s.addText(`Transformer 입문  |  ${this.n}`, { x: 9.2, y: 6.98, w: 3.5, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "right", margin: 0 });
    if (notes) s.addNotes(notes);
    return s;
  }

  // ---- 일반 텍스트 ----
  text(s, t, o) {
    const size = o.size || 16;
    const lines = Array.isArray(t) ? t.map(r => (typeof r === "string" ? r : r.text)) : String(t).split("\n");
    if (estHeight(lines, size, o.w - 0.1) > o.h + 0.05) this.warn(`텍스트 넘침 가능: "${lines[0].slice(0, 20)}"`);
    s.addText(t, { fontFace: FONT, fontSize: size, color: C.ink, margin: 0, valign: "top", ...o, size: undefined });
  }

  // ---- 불릿 목록 ----
  bullets(s, items, { x, y, w, h, size = 16, gap = 8, color = C.ink }) {
    const runs = items.map((it, i) => {
      const txt = typeof it === "string" ? it : it.t;
      const b = typeof it === "string" ? false : !!it.b;
      return { text: txt, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1, bold: b, paraSpaceAfter: gap } };
    });
    const lines = items.map(it => (typeof it === "string" ? it : it.t));
    size = fitSize(lines, w - 0.35, h, size, Math.min(size + 4, 20), (items.length * gap) / 72);
    runs.forEach(r => { r.options.paraSpaceAfter = gap * size / 14; });
    if (estHeight(lines, size, w - 0.35, 1.3) + (items.length * gap) / 72 > h + 0.05) this.warn(`불릿 넘침 가능: "${lines[0].slice(0, 20)}"`);
    s.addText(runs, { x, y, w, h, fontFace: FONT, fontSize: size, color, margin: 0, valign: "top" });
  }

  // ---- 카드 (연한 배경 + 둥근 모서리) ----
  card(s, { x, y, w, h, fill = C.paper, title = "", body = "", tColor = C.teal, size = 14, tSize = 16, line = null }) {
    s.addShape(this.pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1 } : { color: fill, width: 0 }, rectRadius: 0.12 });
    let ty = y + 0.15;
    if (title) {
      s.addText(title, { x: x + 0.2, y: ty, w: w - 0.4, h: 0.4, fontFace: FONT, fontSize: tSize, bold: true, color: tColor, margin: 0, valign: "middle" });
      ty += 0.45;
    }
    if (body) {
      const lines = String(body).split("\n");
      if (estHeight(lines, size, w - 0.45) > h - (ty - y) - 0.1) this.warn(`카드 넘침 가능: "${(title || lines[0]).slice(0, 20)}"`);
      s.addText(body, { x: x + 0.2, y: ty, w: w - 0.4, h: h - (ty - y) - 0.1, fontFace: FONT, fontSize: size, color: C.ink, margin: 0, valign: "top", paraSpaceAfter: 4 });
    }
  }

  // ---- 한 줄 박스 (도식용) ----
  box(s, t, { x, y, w, h, fill = C.tealL, color = C.ink, size = 14, bold = false, line = null, shape = "ROUNDED_RECTANGLE", align = "center" }) {
    s.addText(t, { shape: this.pres.shapes[shape], x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1.25 } : { color: fill, width: 0 }, rectRadius: 0.1, fontFace: FONT, fontSize: size, color, bold, align, valign: "middle", margin: 0.05 });
    if (estHeight(String(t).split("\n"), size, w - 0.1) > h + 0.02) this.warn(`박스 넘침 가능: "${String(t).slice(0, 20)}"`);
  }

  arrow(s, x1, y1, x2, y2, color = C.muted, width = 2) {
    const flipH = x2 < x1, flipV = y2 < y1;
    s.addShape(this.pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH, flipV, line: { color, width, endArrowType: "triangle" } });
  }

  // ---- 숫자 표 (행렬 그림) ----
  grid(s, rows, { x, y, cw = 0.75, ch = 0.42, fill = C.tealL, size = 13, color = C.ink, hi = null }) {
    rows.forEach((r, i) => r.forEach((v, j) => {
      const f = hi && hi(i, j) ? hi(i, j) : fill;
      s.addText(String(v), { shape: this.pres.shapes.RECTANGLE, x: x + j * cw, y: y + i * ch, w: cw, h: ch, fill: { color: f }, line: { color: C.white, width: 1.5 }, fontFace: FONT, fontSize: size, color, align: "center", valign: "middle", margin: 0 });
    }));
  }

  // ---- 코드 박스: 주석(#)은 회색-초록, 나머지는 어두운 색 ----
  code(s, lines, { x, y, w, h, size = 11, title = "" }) {
    s.addShape(this.pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.gray }, line: { color: C.line, width: 1 }, rectRadius: 0.08 });
    let top = y;
    if (title) {
      s.addText(title, { x: x + 0.15, y: y + 0.06, w: w - 0.3, h: 0.28, fontFace: MONO, fontSize: 10, bold: true, color: C.teal, margin: 0 });
      top += 0.32;
    }
    const runs = [];
    let maxW = 0;
    // 폭·높이에 모두 맞는 가장 큰 글자 크기 선택 (size = 최소 크기)
    const wNeed = (sz) => lines.reduce((m, ln) => { let u = 0; for (const ch of ln) u += /[ㄱ-힝]/.test(ch) ? 1.0 : 0.55; return Math.max(m, u * sz / 72); }, 0);
    const hNeed = (sz) => lines.length * sz * 1.3 / 72 + 0.25 + (title ? 0.32 : 0);
    let best = size;
    for (let sz = 16; sz > size; sz -= 0.5) { if (wNeed(sz) <= w - 0.4 && hNeed(sz) <= h) { best = sz; break; } }
    size = best;
    lines.forEach((ln, i) => {
      const k = ln.indexOf("#");
      const code = k >= 0 ? ln.slice(0, k) : ln;
      const cm = k >= 0 ? ln.slice(k) : "";
      maxW = Math.max(maxW, widthUnits(ln) );
      const last = i === lines.length - 1;
      if (code.length || !cm) runs.push({ text: code.length ? code : " ", options: { color: C.ink, fontFace: MONO, breakLine: last ? false : !cm } });
      if (cm) runs.push({ text: cm, options: { color: C.cmt, fontFace: FONT, italic: false, breakLine: !last } });
    });
    const unitW = size * 0.5 / 72;           // 영문 모노스페이스 한 글자 폭(인치)
    const w_needed = lines.reduce((m, ln) => {
      let u = 0; for (const ch of ln) u += /[ㄱ-힝]/.test(ch) ? 1.0 : 0.55; return Math.max(m, u * size / 72);
    }, 0);
    if (w_needed > w - 0.3) this.warn(`코드 폭 초과(${w_needed.toFixed(1)}in > ${(w - 0.3).toFixed(1)}in): ${lines.find(l => l.length > 10)?.slice(0, 30)}`);
    const needH = lines.length * size * 1.3 / 72 + 0.2 + (title ? 0.32 : 0);
    if (needH > h) this.warn(`코드 높이 초과(${needH.toFixed(2)}in > ${h}in)`);
    s.addText(runs, { x: x + 0.15, y: top + 0.1, w: w - 0.3, h: h - (top - y) - 0.15, fontFace: MONO, fontSize: size, color: C.ink, margin: 0, valign: "top", wrap: false, lineSpacingMultiple: 1.0 });
  }

  // ---- 코드 설명 패널 (오른쪽) ----
  explain(s, items, { x, y, w, h, size = 13, title = "코드 읽는 법" }) {
    s.addShape(this.pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.tealL }, line: { color: C.tealL, width: 0 }, rectRadius: 0.1 });
    s.addText(title, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.35, fontFace: FONT, fontSize: 15, bold: true, color: C.teal, margin: 0 });
    const nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦"];
    const runs = items.map((t, i) => ({ text: `${nums[i]} ${t}`, options: { breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
    const lines = items.map((t, i) => `${nums[i]} ${t}`);
    size = fitSize(lines, w - 0.45, h - 0.6, size, 17, (items.length * 6) / 72);
    runs.forEach(r => { r.options.paraSpaceAfter = 6 * size / 13; });
    if (estHeight(lines, size, w - 0.45) + (items.length * 6) / 72 > h - 0.6) this.warn(`설명 패널 넘침 가능: "${items[0].slice(0, 20)}"`);
    s.addText(runs, { x: x + 0.2, y: y + 0.55, w: w - 0.4, h: h - 0.65, fontFace: FONT, fontSize: size, color: C.ink, margin: 0, valign: "top" });
  }

  // ---- 구역 표지 ----
  section(num, title, sub, notes = "") {
    const s = this.pres.addSlide();
    this.n += 1;
    s.background = { color: C.paper };
    s.addText(num, { x: 0.9, y: 2.0, w: 11, h: 1.4, fontFace: FONT, fontSize: num.length > 4 ? 60 : 80, bold: true, color: C.teal, margin: 0 });
    s.addText(title, { x: 0.9, y: 3.5, w: 11.5, h: 0.9, fontFace: FONT, fontSize: 38, bold: true, color: C.ink, margin: 0 });
    s.addText(sub, { x: 0.9, y: 4.5, w: 11.5, h: 0.9, fontFace: FONT, fontSize: 18, color: C.muted, margin: 0, valign: "top" });
    s.addImage({ path: path.join(IMG, "logo.png"), x: 0.6, y: 7.0, w: 1.4, h: 1.4 * 118 / 756 });
    s.addText(`Transformer 입문  |  ${this.n}`, { x: 9.2, y: 6.98, w: 3.5, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "right", margin: 0 });
    if (notes) s.addNotes(notes);
    return s;
  }
}

module.exports = { C, FONT, MONO, IMG, Deck, widthUnits, estHeight };
