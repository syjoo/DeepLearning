const path = require("path");
const pptxgen = require(path.join(process.env.PPTX_NODE_MODULES, "pptxgenjs"));
const { Deck } = require("./lib");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Transformer 입문";
pres.author = "TuringLab. Programming Academy";
const d = new Deck(pres, "Transformer 입문");

require("./part1")(d, pres);
require("./part2")(d, pres);

const out = path.join(__dirname, "..", "Transformer_lecture.pptx");
pres.writeFile({ fileName: out }).then(() => {
  console.log("written", out, "slides:", d.n);
  if (d.warns.length) { console.log("WARNINGS:"); d.warns.forEach(w => console.log(" -", w)); }
});
