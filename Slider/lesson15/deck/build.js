const path = require("path");
const pptxgen = require("pptxgenjs");
const { Deck } = require("./lib");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Lesson 15. Explainability";
pres.author = "TuringLab. Programming Academy";
const d = new Deck(pres, "Lesson 15");
require("./part1")(d, pres);
require("./part2")(d, pres);
require("./part3")(d, pres);
require("./part4")(d, pres);
const out = path.join(__dirname, "..", "Lesson15_lecture.pptx");
pres.writeFile({ fileName: out }).then(() => {
  console.log("written", out, "slides:", d.n);
  if (d.warns.length) { console.log("WARNINGS:"); d.warns.forEach(w => console.log(" -", w)); }
});
