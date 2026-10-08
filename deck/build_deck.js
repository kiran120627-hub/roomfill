// RoomFill hackathon deck (7 slides, dark theme). Run: node build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const A = (f) => path.join(__dirname, "assets", f);
const THEME = {
  name: "RoomFill Dark",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0E0F12", lt1: "EDEDF0", dk2: "1A1C22", lt2: "A3A6B1",
    accent1: "E0479E", accent2: "5FC79A", accent3: "7AA2F7", accent4: "F2B84B",
    accent5: "2A2D36", accent6: "6F7380", hlink: "E0479E", folHlink: "C17BB0",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "RoomFill team, Karunya";
pres.title = "RoomFill: HackNex 2026";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const BG = C.text1, SURF = C.text2, TXT = C.background1, MUTED = C.background2;
const ACC = C.accent1, OBS = C.accent2, CARD2 = C.accent5;

// ---------- layouts (frames) ----------
const logoTile = (x, y, s) => [
  { rect: { x, y, w: s, h: s * 1.2, fill: { color: "FFFFFF" }, rectRadius: 0.08 } },
  { image: { x: x + s * 0.08, y: y + s * 0.08, w: s * 0.84, h: s * 1.02, path: A("karunya_logo.png") } },
];
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk1 },
  objects: [...logoTile(0.6, 0.55, 0.62)],
});
pres.defineSlideMaster({
  title: "CONTENT_DARK",
  background: { color: HEX.dk1 },
  objects: [
    ...logoTile(12.28, 0.36, 0.48),
    { text: { text: "RoomFill  |  HackNex 2026  |  Karunya Institute of Technology and Sciences",
              options: { x: 0.6, y: 7.02, w: 9, h: 0.3, fontSize: 10, color: HEX.lt2, isTextBox: true, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.38, w: 11.4, h: 0.75,
      fontSize: 34, bold: true, color: HEX.lt1, fontFace: THEME.headFontFace, valign: "middle", margin: 0 },
      text: "Slide title" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.lt2, align: "right" },
});

const card = (s, x, y, w, h, name, fill = SURF) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: HEX.accent5, width: 0.75 }, objectName: name });
const txt = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, color: TXT, fontSize: 14, valign: "top", ...o });
const pill = (s, x, y, w, label, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.21, fill: { color: ACC, transparency: 82 },
    line: { color: HEX.accent1, width: 1 }, objectName: name });
  txt(s, label, { x, y, w, h: 0.42, fontSize: 12, bold: true, color: ACC, align: "center", valign: "middle" });
};
const numDot = (s, x, y, n, color = ACC) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color }, line: { color, width: 0 }, objectName: `dot-${n}` });
  txt(s, String(n), { x, y, w: 0.42, h: 0.42, fontSize: 13, bold: true, color: BG, align: "center", valign: "middle" });
};

// ======================= 1. Title =======================
pres.addSection({ title: "RoomFill" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "RoomFill" });
s.addImage({ path: A("demo_45.jpg"), x: 6.55, y: 0.55, w: 6.2, h: 3.07, objectName: "hero" });
s.addImage({ path: A("ceiling_magenta.jpg"), x: 6.55, y: 3.78, w: 3.02, h: 1.5, sizing: { type: "cover", w: 3.02, h: 1.5 }, objectName: "hero-generated" });
s.addImage({ path: A("demo_400.jpg"), x: 9.73, y: 3.78, w: 3.02, h: 1.5, objectName: "hero-compare" });
txt(s, "HackNex 2026  |  HNX26EPS06  |  Mode B (+ Mode A bonus)", { x: 0.6, y: 1.55, w: 5.6, h: 0.35, fontSize: 13, color: MUTED });
txt(s, "RoomFill", { x: 0.6, y: 1.95, w: 5.6, h: 1.0, fontSize: 54, bold: true, fontFace: THEME.headFontFace });
txt(s, [
  { text: "A 51-second phone video becomes a metric, walkable 3D room. ", options: { color: TXT } },
  { text: "Surfaces the camera never saw are filled in and marked as generated.", options: { color: ACC, italic: true } },
], { x: 0.6, y: 3.0, w: 5.6, h: 1.1, fontSize: 18 });
const stats1 = [["15.5 cm", "Chamfer to the real room\n(baseline 22.2 cm)"], ["90%", "of true surfaces covered\n(baseline 52%)"], ["0", "LLMs used\nRs 0 API cost"]];
stats1.forEach(([big, small], i) => {
  const x = 0.6 + i * 1.9;
  card(s, x, 4.45, 1.75, 1.45, `stat-${i}`);
  txt(s, big, { x: x + 0.15, y: 4.55, w: 1.5, h: 0.6, fontSize: 26, bold: true, color: i === 2 ? OBS : ACC, fontFace: THEME.headFontFace });
  txt(s, small, { x: x + 0.15, y: 5.15, w: 1.55, h: 0.7, fontSize: 11, color: MUTED });
});
txt(s, "Live demo: kiran120627-hub.github.io/roomfill/viewer      Code: github.com/kiran120627-hub/roomfill",
  { x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontSize: 12, color: MUTED });
s.addNotes("Good morning. We are from Karunya Institute of Technology and Sciences, and this is RoomFill, our solution to HNX26EPS06. You walk through a room with an ordinary phone for under a minute. RoomFill turns that video into a 3D model of the room at real-world scale, fills in the parts the camera never saw, like most of the ceiling, and marks every filled-in surface so a guess is never mistaken for a measurement. On our own classroom, the shape error against a tape measure fell from 22.2 cm to 15.5 cm, and the share of the room surface we reconstruct went from 52% to 90%. It runs offline on one laptop GPU, at zero cost per run, and there is no LLM anywhere in it.");

// ======================= 2. Problem and idea =======================
pres.addSection({ title: "Problem and approach" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Problem and approach" });
s.addText("Completing the unseen is the hard part", { placeholder: "title" });
card(s, 0.6, 1.45, 5.85, 4.15, "problem-card");
txt(s, "The problem", { x: 0.95, y: 1.7, w: 5.2, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace });
const probs = [
  "Standard 3D Gaussian Splatting leaves holes and floaters wherever the camera did not look",
  "It has no real-world scale, so a room cannot be measured",
  "It gives no signal about which parts were actually seen, so guesses look like facts",
];
probs.forEach((p, i) => {
  numDot(s, 0.95, 2.35 + i * 1.0, i + 1, MUTED);
  txt(s, p, { x: 1.55, y: 2.3 + i * 1.0, w: 4.65, h: 0.9, fontSize: 15 });
});
card(s, 6.85, 1.45, 5.85, 4.15, "idea-card");
txt(s, "Our idea", { x: 7.2, y: 1.7, w: 5.2, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: ACC });
const ideas = [
  ["Structure first", "Fit the room shell (floor, ceiling, 4 walls) and complete only on it, so nothing floats in mid-air"],
  ["See before you fill", "Every shell point is checked against all camera views; only never-seen points are generated"],
  ["Honest by design", "Each 3D point carries a generated flag and confidence, kept in every exported file"],
];
ideas.forEach(([h, b], i) => {
  numDot(s, 7.2, 2.35 + i * 1.0, i + 1);
  txt(s, [{ text: h + "  ", options: { bold: true, color: TXT } }, { text: b, options: { color: MUTED } }],
    { x: 7.8, y: 2.3 + i * 1.0, w: 4.65, h: 0.9, fontSize: 14 });
});
card(s, 0.6, 5.85, 12.1, 0.95, "no-llm", CARD2);
txt(s, [
  { text: "No LLM in the system.  ", options: { bold: true, color: OBS } },
  { text: "RoomFill uses classic geometry (COLMAP), per-scene neural rendering (3D Gaussian Splatting) and two vision models (LaMa inpainting, Depth Anything V2), all running locally on one laptop GPU. No ChatGPT, Claude or Gemini calls; Rs 0 API cost.", options: { color: TXT } },
], { x: 0.95, y: 5.95, w: 11.5, h: 0.78, fontSize: 14, valign: "middle" });
s.addNotes("The standard tool today is 3D Gaussian Splatting. It looks great where the camera went, but it has three problems. One: wherever the camera did not look you get holes and floating blobs. Two: it has no idea of real size, so you cannot measure the room. Three: it never tells you which parts are real and which are guessed. Our three ideas answer those directly. First, fit the room shell (floor, ceiling, four walls) and complete only on that shell, so nothing floats. Second, before filling anything, check every point of the shell against all 400 camera views; only points no camera ever saw get filled. Third, every filled point carries a generated flag and a confidence that survive into every exported file. And to be clear: no large language model anywhere. It is geometry, a per-scene neural renderer and two small vision models.");

// ======================= 3. System architecture =======================
pres.addSection({ title: "System architecture" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "System architecture" });
s.addText("System architecture", { placeholder: "title" });
const inputs = ["Phone video (51 s)", "5 held-out photos", "Floor plan image", "Tape measurements"];
txt(s, "INPUTS", { x: 0.6, y: 1.38, w: 1.2, h: 0.3, fontSize: 11, bold: true, color: MUTED });
inputs.forEach((t, i) => {
  const x = 1.75 + i * 2.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.5, h: 0.45, rectRadius: 0.22, fill: { color: SURF },
    line: { color: HEX.accent6, width: 0.75 }, objectName: `input-${i}` });
  txt(s, t, { x, y: 1.3, w: 2.5, h: 0.45, fontSize: 12, align: "center", valign: "middle" });
});
const stages = [
  ["Frame selection", "400 sharpest + 78 densified"],
  ["COLMAP SfM", "camera poses, 483 images"],
  ["3D Gaussian Splatting", "gsplat on RTX 5070"],
  ["Floater pruning", "0.5 m around camera path"],
  ["Room shell fit", "floor, ceiling, Manhattan walls"],
  ["Metric scale", "floor plan or camera prior"],
  ["Visibility test", "every texel vs 400 views"],
  ["LaMa completion", "only never-seen texels"],
  ["Honesty tags", "generated flag + confidence"],
  ["Export + viewer", "PLY, .splat, GLB, USD, web"],
];
const BW = 2.18, BH = 1.05, GAP = 0.3, X0 = 0.6;
const boxAt = (i) => {
  const row = i < 5 ? 0 : 1;
  const col = row === 0 ? i : 9 - i;            // second row runs right to left (snake)
  return { x: X0 + col * (BW + GAP), y: row === 0 ? 2.2 : 3.85 };
};
stages.forEach(([h, b], i) => {
  const { x, y } = boxAt(i);
  const hot = i >= 6 && i <= 8;                   // our core contribution
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: BW, h: BH, rectRadius: 0.12,
    fill: { color: hot ? ACC : SURF, transparency: hot ? 80 : 0 },
    line: { color: hot ? HEX.accent1 : HEX.accent5, width: hot ? 1.25 : 0.75 }, objectName: `stage-${i + 1}` });
  txt(s, `${i + 1}  ${h}`, { x: x + 0.14, y: y + 0.14, w: BW - 0.28, h: 0.38, fontSize: 13, bold: true, color: hot ? ACC : TXT });
  txt(s, b, { x: x + 0.14, y: y + 0.52, w: BW - 0.28, h: 0.45, fontSize: 11, color: MUTED });
});
for (let i = 0; i < 9; i++) {
  const a = boxAt(i), b = boxAt(i + 1);
  if (i === 4) {
    s.addShape(pres.shapes.LINE, { x: a.x + BW / 2, y: a.y + BH, w: 0, h: b.y - a.y - BH,
      line: { color: HEX.lt2, width: 1.5, endArrowType: "triangle" }, objectName: "arrow-down" });
  } else if (i < 4) {
    s.addShape(pres.shapes.LINE, { x: a.x + BW + 0.03, y: a.y + BH / 2, w: GAP - 0.06, h: 0,
      line: { color: HEX.lt2, width: 1.5, endArrowType: "triangle" }, objectName: `arrow-${i}` });
  } else {
    s.addShape(pres.shapes.LINE, { x: b.x + BW + 0.03, y: a.y + BH / 2, w: GAP - 0.06, h: 0, flipH: true,
      line: { color: HEX.lt2, width: 1.5, endArrowType: "triangle" }, objectName: `arrow-${i}` });
  }
}
card(s, 0.6, 5.25, 7.4, 1.55, "models-card");
txt(s, "Models and tools (all local, all open source)", { x: 0.85, y: 5.38, w: 7, h: 0.35, fontSize: 14, bold: true });
txt(s, "COLMAP 3.12  |  gsplat 1.6 (3DGS, MCMC)  |  LaMa inpainting  |  Depth Anything V2 Metric Indoor (scale check)  |  LPIPS-VGG (evaluation only)  |  three.js + Spark web viewer",
  { x: 0.85, y: 5.78, w: 6.95, h: 0.95, fontSize: 12, color: MUTED });
card(s, 8.25, 5.25, 4.45, 1.55, "llm-card", CARD2);
txt(s, "0 LLMs", { x: 8.5, y: 5.35, w: 2, h: 0.6, fontSize: 30, bold: true, color: OBS, fontFace: THEME.headFontFace });
txt(s, "No language model anywhere in the pipeline. Runs offline on one laptop GPU; Rs 0 API cost.",
  { x: 8.5, y: 5.95, w: 4.0, h: 0.8, fontSize: 12, color: TXT });
s.addNotes("Four inputs: the walkthrough video, five photos we keep hidden for testing, the floor plan, and our tape measurements, used only for scoring and scale checks. Stage 1 picks the 400 sharpest frames. Stage 2, COLMAP, works out where the camera was for every frame. Stage 3 trains 3D Gaussian Splatting on the GPU. Stage 4 deletes floaters: blobs within half a metre of the path we actually walked, which must be empty air. Stage 5 fits the room box; stage 6 gives it real-world scale from the floor plan, with a camera-height fallback. The magenta stages are our core contribution: stage 7 tests every texel of the shell against every camera to find what was truly never seen; stage 8 fills only those texels with LaMa inpainting; stage 9 tags them as generated with a confidence. Stage 10 exports PLY, splat, GLB and USD, plus the web viewer. Bottom right: zero LLMs, zero API cost.");

// ======================= 4. Data =======================
pres.addSection({ title: "Data" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Data" });
s.addText("All the data we used", { placeholder: "title" });
const hdr = (t) => ({ text: t, options: { bold: true, color: TXT, fill: { color: HEX.accent5 }, fontSize: 13 } });
const cell = (t, o = {}) => ({ text: t, options: { color: TXT, fontSize: 12, ...o } });
const rows = [
  [hdr("Data"), hdr("Details")],
  [cell("Walkthrough video", { bold: true }), cell("Karunya classroom, 57 s, 1920 x 1080, 30 fps; first 51 s used (inside the room)")],
  [cell("Training frames", { bold: true }), cell("400 sharpest frames + 78 extra from the soft 38-51 s stretch")],
  [cell("Held-out photos", { bold: true }), cell("5 phone stills from new positions, never used for training")],
  [cell("Ground truth", { bold: true }), cell("Tape: 9.23 x 8.88 m floor, 3.5 m ceiling, door 1.0 x 2.05 m")],
  [cell("Floor plan", { bold: true }), cell("2D plan with walls, door and labelled dimensions")],
  [cell("Development scene", { bold: true }), cell("Deep Blending \"Playroom\" (public) to test every stage first")],
  [cell("Pretrained models", { bold: true }), cell("LaMa, Depth Anything V2 Metric Indoor S, VGG16 (LPIPS only)")],
  [cell("Hardware", { bold: true }), cell("Laptop RTX 5070 (8 GB), CUDA 13, WSL Ubuntu")],
];
s.addTable(rows, { x: 0.6, y: 1.45, w: 7.4, colW: [1.95, 5.45], rowH: 0.52, fill: { color: SURF },
  border: { type: "solid", pt: 0.5, color: HEX.accent5 }, valign: "middle", fontFace: THEME.bodyFontFace });
s.addImage({ path: A("input_frame.jpg"), x: 8.35, y: 1.45, w: 4.35, h: 2.45, objectName: "input-photo" });
txt(s, "Input: one frame of the phone walkthrough", { x: 8.35, y: 3.95, w: 4.35, h: 0.3, fontSize: 11, color: MUTED });
s.addImage({ path: A("floorplan_parsed.png"), x: 9.3, y: 4.35, w: 2.45, h: 2.25, objectName: "floorplan" });
txt(s, "Floor plan, parsed: walls, interior, door", { x: 8.35, y: 6.65, w: 4.35, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
s.addNotes("Everything except the development scene we captured ourselves, in a Karunya classroom. A 57-second 1080p phone walkthrough; we used the first 51 seconds, inside the room. From it, 400 sharp frames plus 78 extra from a blurry stretch. Five still photos from new positions are held out: the model never sees them, and we use them to score image quality honestly. Ground truth is a tape measure: 9.23 by 8.88 metres, 3.5 metre ceiling, a 1.0 by 2.05 metre door. We also used the room's floor plan. Every stage was built and tested first on the public Playroom scene. The video and photos show people, so they stay private; the code and the reconstruction are public.");

// ======================= 5. Results =======================
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Results" });
s.addText("RoomFill beats the baseline on every metric", { placeholder: "title" });
const chartBase = (title) => ({
  showTitle: true, title, titleFontSize: 14, titleColor: HEX.lt1, titleFontFace: "+mj-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: HEX.lt1, dataLabelFontSize: 13, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.lt2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, chartColors: [HEX.accent6, HEX.accent1], barGapWidthPct: 60,
  plotArea: { fill: { color: HEX.dk2 } },
});
card(s, 0.6, 1.45, 3.9, 3.6, "chart1-card");
s.addChart(pres.charts.BAR, [{ name: "Chamfer (cm)", labels: ["Baseline 3DGS", "RoomFill"], values: [22.2, 15.5] }],
  { x: 0.75, y: 1.55, w: 3.6, h: 3.4, ...chartBase("Chamfer distance, cm (lower is better)"), valAxisMinVal: 0, valAxisMaxVal: 26,
    dataLabelFormatCode: "0.0" });
card(s, 4.75, 1.45, 3.9, 3.6, "chart2-card");
s.addChart(pres.charts.BAR, [{ name: "Coverage", labels: ["Baseline 3DGS", "RoomFill"], values: [52, 90] }],
  { x: 4.9, y: 1.55, w: 3.6, h: 3.4, ...chartBase("Room surfaces covered within 25 cm"), valAxisMinVal: 0, valAxisMaxVal: 100,
    dataLabelFormatCode: "0\"%\"" });
card(s, 8.9, 1.45, 3.8, 3.6, "quality-card");
txt(s, "Held-out photos (full resolution)", { x: 9.15, y: 1.6, w: 3.4, h: 0.35, fontSize: 14, bold: true, fontFace: THEME.headFontFace });
const q = [["", "Baseline", "RoomFill"], ["PSNR (raw)", "10.68", "11.26"], ["PSNR", "15.79", "16.15"], ["SSIM", "0.720", "0.724"], ["LPIPS (lower)", "0.605", "0.601"]];
s.addTable(q.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 13, bold: i === 0 || j === 2,
  color: j === 2 && i > 0 ? HEX.accent1 : (i === 0 ? HEX.lt2 : HEX.lt1), align: j === 0 ? "left" : "right" } }))),
  { x: 9.15, y: 2.05, w: 3.35, colW: [1.35, 1.0, 1.0], rowH: 0.42, fill: { color: HEX.dk2 }, border: { type: "none" }, valign: "middle" });
txt(s, "PSNR / SSIM / LPIPS colour-aligned: one colour map per photo, because phone photo mode and video mode differ in exposure. Raw shown too.",
  { x: 9.15, y: 4.2, w: 3.4, h: 0.8, fontSize: 10, color: MUTED });
const st = [["3.7-4.0%", "length / width error vs tape"], ["6.4%", "ceiling height error (independent check)"], ["-30%", "Chamfer distance vs baseline"], ["Rs 0", "API cost, fully local"]];
st.forEach(([b, l], i) => {
  const x = 0.6 + i * 3.05;
  card(s, x, 5.3, 2.9, 1.5, `res-stat-${i}`);
  txt(s, b, { x: x + 0.2, y: 5.4, w: 2.5, h: 0.65, fontSize: 28, bold: true, color: i === 3 ? OBS : ACC, fontFace: THEME.headFontFace });
  txt(s, l, { x: x + 0.2, y: 6.05, w: 2.55, h: 0.6, fontSize: 12, color: MUTED });
});
s.addNotes("The baseline is plain 3D Gaussian Splatting trained on exactly the same frames for the same number of steps, so the only difference is our pipeline. Left, geometry: we compare reconstructed walls, floor and ceiling with a box built from the tape measurements. Error drops from 22.2 to 15.5 centimetres, about 30% better, and the share of the room within 25 centimetres of the truth goes from 52 to 90 percent. Right, image quality on the five hidden photos: we beat the baseline on PSNR, SSIM and LPIPS. Each photo is colour-aligned first because phone photo mode and video mode expose differently; the raw PSNR is shown too and still favours us. Bottom: room length and width within 4% of the tape, height within 6.4%.");

// ======================= 6. Honesty =======================
pres.addSection({ title: "Completion and honesty" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Completion and honesty" });
s.addText("Filled where unseen, and always marked", { placeholder: "title" });
s.addImage({ path: A("2_honesty_view1.jpg"), x: 0.6, y: 1.45, w: 7.4, h: 2.07, objectName: "honesty-ceiling" });
txt(s, "Looking up at a ceiling the video never filmed: filled in (left), the same view with generated regions in magenta (right). Fans and beams the camera saw stay untouched.",
  { x: 0.6, y: 3.6, w: 7.4, h: 0.6, fontSize: 12, color: MUTED });
s.addImage({ path: A("demo_400.jpg"), x: 0.6, y: 4.3, w: 4.85, h: 2.4, objectName: "compare" });
txt(s, "New viewpoint: the baseline shows floater streaks, RoomFill stays clean", { x: 5.6, y: 4.4, w: 2.45, h: 1.6, fontSize: 13, color: TXT });
card(s, 8.3, 1.45, 4.4, 5.25, "surfaces-card");
s.addChart(pres.charts.BAR, [
  { name: "Observed", labels: ["Ceiling", "Floor", "Board wall", "Window wall", "Back wall", "Side wall"], values: [39, 76, 57, 73, 39, 40] },
  { name: "Opening", labels: ["Ceiling", "Floor", "Board wall", "Window wall", "Back wall", "Side wall"], values: [0, 0, 2, 0, 0, 8] },
  { name: "Generated", labels: ["Ceiling", "Floor", "Board wall", "Window wall", "Back wall", "Side wall"], values: [61, 24, 41, 27, 61, 52] },
], { x: 8.45, y: 1.55, w: 4.1, h: 5.05, barDir: "bar", barGrouping: "percentStacked",
  showTitle: true, title: "Each surface: observed / opening / generated", titleFontSize: 13, titleColor: HEX.lt1, titleFontFace: "+mj-lt",
  chartColors: [HEX.accent2, HEX.lt2, HEX.accent1], showValue: false,
  catAxisLabelColor: HEX.lt1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelColor: HEX.lt2, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt", valGridLine: { color: HEX.accent5, size: 0.5 },
  catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendColor: HEX.lt1, legendFontSize: 11, legendFontFace: "+mn-lt",
  plotArea: { fill: { color: HEX.dk2 } } });
s.addNotes("This is the honesty part. Top: looking up at the ceiling, which the walkthrough barely filmed. Left is the filled-in result; right is the same view with everything we generated in magenta. Stronger magenta means lower confidence, further from real evidence. The fans and beams the camera did see are untouched. Bottom: a viewpoint nobody filmed; the baseline smears floaters across it, RoomFill stays clean. Right: for each surface, how much was observed, seen through an opening like a door or window, and generated. The ceiling and back wall were 61% generated, the floor only 24%. Openings the camera saw through are protected, so we never paint a wall over a doorway.");

// ======================= 7. Bonus + close =======================
pres.addSection({ title: "Beyond the minimum" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Beyond the minimum" });
s.addText("All five stretch goals, plus a live demo", { placeholder: "title" });
const bonus = [
  ["Both modes in one pipeline", "Floor plan parsed to 3D; it sets metric scale and its door aligns 77% with the doorway the video saw through"],
  ["Observed vs generated", "Magenta overlay with a confidence ramp, in the viewer and in every exported file"],
  ["Object-level", "Furniture split into 36 objects (18 desk units, 6 fans and lights) with boxes"],
  ["Sparse photos", "30 frames match the 400-frame video on views they cover (17.06 vs 17.13 PSNR)"],
  ["Standard formats", "PLY, compact .splat, textured GLB and USD; open in Blender"],
];
bonus.forEach(([h, b], i) => {
  const y = 1.45 + i * 0.98;
  card(s, 0.6, y, 6.6, 0.85, `bonus-${i}`);
  numDot(s, 0.82, y + 0.21, i + 1);
  txt(s, [{ text: h, options: { bold: true, color: TXT, breakLine: true } }, { text: b, options: { color: MUTED, fontSize: 12 } }],
    { x: 1.42, y: y + 0.09, w: 5.65, h: 0.72, fontSize: 14 });
});
s.addImage({ path: A("demo_45.jpg"), x: 7.5, y: 1.45, w: 5.2, h: 2.58, objectName: "demo-photo" });
card(s, 7.5, 4.2, 5.2, 1.4, "links-card", CARD2);
txt(s, [
  { text: "Live demo", options: { bold: true, color: ACC, breakLine: true } },
  { text: "kiran120627-hub.github.io/roomfill/viewer", options: { color: TXT, breakLine: true } },
  { text: "Code: github.com/kiran120627-hub/roomfill", options: { color: MUTED, breakLine: true } },
  { text: "Press T, click two points: real distance in metres", options: { color: OBS, fontSize: 12 } },
], { x: 7.75, y: 4.28, w: 4.8, h: 1.28, fontSize: 14 });
card(s, 7.5, 5.75, 5.2, 1.05, "limits-card");
txt(s, [
  { text: "Limits we state openly: ", options: { bold: true, color: TXT } },
  { text: "rectangular rooms only; we complete surfaces, not hidden furniture; a small depth model was 2x off here, so a sanity check falls back to a camera-height prior.", options: { color: MUTED } },
], { x: 7.75, y: 5.83, w: 4.8, h: 0.92, fontSize: 11 });
s.addNotes("Beyond the core task we covered all five stretch goals: floor-plan mode in the same pipeline, the observed-versus-generated overlay, splitting furniture into 36 objects, a sparse mode where 30 frames nearly match 400, and export to PLY, GLB and USD that open in Blender. Limits: rectangular rooms only, and we complete surfaces, not furniture hidden behind other furniture. LIVE DEMO: open the link. Drag to look around, scroll to walk. Press G: magenta shows what was generated. Press B: the baseline, with its floaters. Press 2 to look up at the ceiling. Press T and click both ends of a desk: the real length appears in metres (about 1.1 m). Esc to finish.");

(async () => {
  const out = path.join(__dirname, "RoomFill_HackNex2026.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
