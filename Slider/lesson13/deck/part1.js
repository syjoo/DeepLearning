// 1교시: 왜 증강인가, 증강의 정의, 레이블 일관성
const path = require("path");
const { C, FONT, IMG } = require("./lib");
const { img, F, table } = require("./helpers");

module.exports = function part1(d, pres) {
  // ---------------- 표지 ----------------
  {
    const s = pres.addSlide(); d.n += 1; s.background = { color: C.paper };
    s.addText("Lesson 13", { x: 0.9, y: 1.3, w: 11, h: 0.7, fontFace: FONT, fontSize: 28, bold: true, color: C.teal, margin: 0 });
    s.addText("Data Augmentation", { x: 0.9, y: 2.0, w: 11.5, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    s.addText("데이터 증강: 같은 데이터로 더 튼튼한 모델 만들기", { x: 0.9, y: 3.1, w: 11.5, h: 0.6, fontFace: FONT, fontSize: 24, color: C.teal, bold: true, margin: 0 });
    s.addText("레이블 보존 변환 · 불변성 · 온라인/오프라인 증강 · Cutout · Mixup · CutMix", { x: 0.9, y: 3.8, w: 11.5, h: 0.5, fontFace: FONT, fontSize: 18, color: C.muted, margin: 0 });
    ["돌려도", "밀어도", "같은 숫자!"].forEach((t, i) => d.box(s, t, { x: 0.9 + i * 2.2, y: 4.8, w: 2.0, h: 0.7, fill: [C.orangeL, C.tealL, C.greenL][i], size: 20, bold: true }));
    s.addText("튜링랩 프로그래밍학원\nTuringLab. Programming Academy", { x: 0.9, y: 5.9, w: 6, h: 0.7, fontFace: FONT, fontSize: 14, color: C.ink, margin: 0 });
    s.addImage({ path: path.join(IMG, "logo.png"), x: 9.5, y: 6.0, w: 3.0, h: 3.0 * 118 / 756 });
    s.addNotes("강의 소개. 오늘 3시간(50분 x 3교시)의 흐름: 1교시 증강이 필요한 이유와 '레이블 일관성', 2교시 변환을 NumPy로 직접 구현, 3교시 온라인/오프라인 파이프라인과 종합 실험, 최신 기법(Mixup, CutMix 등). 모든 숫자는 sklearn 손글씨 숫자(8x8)로 직접 돌려서 얻은 값이다.");
  }

  // ---------------- 로드맵 ----------------
  {
    const s = d.slide("오늘의 로드맵: 3교시 (50분 수업 + 10분 휴식) × 3", { tag: "OVERVIEW" });
    table(s, ["교시", "주제", "직접 해보는 것"], [
      [{ text: "1교시", options: { bold: true } }, "왜 증강인가 · 증강 vs 생성 · 레이블 일관성", "Step 2  어떤 변환이 정답을 바꾸는지 숫자로 확인"],
      [{ text: "2교시", options: { bold: true } }, "변환 하나하나: 뒤집기·이동·회전·확대·밝기·노이즈", "Step 1  NumPy로 변환 직접 구현"],
      [{ text: "3교시", options: { bold: true } }, "온라인/오프라인 · 파이프라인 · 종합 실험 · 최신 기법", "Step 3~5  증강 파이프라인, 비교 실험"],
      [{ text: "마무리", options: { bold: true } }, "실습 문제 5개 + 솔루션, 원본 자료 검토 결과", "손계산 · 구현 · 예측 후 실험"],
    ], { x: 0.6, y: 1.6, w: 12.1, colW: [1.5, 6.4, 4.2], fontSize: 16, rowH: 0.7 });
    d.text(s, "준비물: Python 3.9+, NumPy, scikit-learn(손글씨 데이터), PyTorch, matplotlib   |   코드는 lesson13/code 폴더", { x: 0.6, y: 5.7, w: 12.1, h: 0.4, size: 15, color: C.muted });
  }

  // ---------------- 학습 목표 ----------------
  {
    const s = d.slide("이 강의가 끝나면 할 수 있는 일", { tag: "GOAL" });
    const items = [
      ["1", "설명하기", "증강이 무엇이고, 왜 과적합을 줄이며, 생성과 어떻게 다른지 내 말로 설명한다"],
      ["2", "판단하기", "어떤 변환이 '레이블을 지키는지' 데이터 종류별로 가려낸다"],
      ["3", "구현하기", "뒤집기·회전·이동·밝기·노이즈와 온라인 증강 파이프라인을 직접 만든다"],
      ["4", "검증하기", "증강 유무를 실험으로 비교하고, 과한 증강과 데이터 누수를 알아챈다"],
    ];
    items.forEach((it, i) => {
      const y = 1.6 + i * 1.2;
      s.addText(it[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 22, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, it[1], { x: 1.7, y, w: 3.2, h: 0.5, size: 20, bold: true });
      d.text(s, it[2], { x: 1.7, y: y + 0.47, w: 10.8, h: 0.5, size: 16, color: C.muted });
    });
  }

  // ---------------- 1교시 ----------------
  d.section("1교시", "왜 데이터 증강인가", "학습은 잘 됐는데 현실에서 실패하는 이유 · 증강의 정의 · 레이블 일관성", "1교시 시작. 이전 시간(Lesson 12)에는 학습을 안정적으로 만드는 초기화와 정규화를 배웠다. 오늘은 '데이터' 쪽에서 일반화를 높이는 방법이다.");

  {
    const s = d.slide("학습은 잘 되는데, 왜 현장에서 실패할까?", { tag: "1교시 · 문제 제기" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 2.4, fill: C.greenL, title: "연구실에서", tColor: C.green, size: 16, body: "· 학습 손실이 매끄럽게 줄어든다\n· 학습 정확도 거의 100%\n· 사진이 깨끗하고 가운데에 반듯하게 찍혀 있다" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 2.4, fill: C.redL, title: "현장에서", tColor: C.red, size: 16, body: "· 사진이 흔들리고, 기울고, 어둡다\n· 물체가 한쪽 구석에 있다\n· 센서 노이즈가 낀다\n→ 성능이 뚝 떨어진다" });
    d.bullets(s, [
      { t: "원인: 모델이 '규칙'이 아니라 '학습에 쓴 사진 그대로'를 외웠다 (과적합)", b: true },
      "현실의 데이터는 노이즈가 많고 일관되지 않아서, 깨끗하게 정제된 학습 샘플과 닮은 구석이 없다",
      "일반화의 열쇠는 네트워크 구조만이 아니라 데이터 자체에 있는 경우가 많다",
    ], { x: 0.6, y: 4.3, w: 12.1, h: 2.4, size: 18 });
    s.addNotes("비유: 시험 문제를 그대로 외운 학생은 숫자만 바뀌어도 못 푼다. 모델도 같다.");
  }

  {
    const s = d.slide("데이터 증강(Data Augmentation)이란?", { tag: "1교시 · 정의" });
    F(d, s, "새 샘플  x′ = T(x),     정답(레이블)은 그대로 y", { x: 0.6, y: 1.5, w: 12.1, h: 0.9, size: 26, fill: C.tealL });
    d.bullets(s, [
      { t: "레이블을 보존하는 변환을 적용해 데이터셋을 확장·다양화하는 과정", b: true },
      "현실에서 일어나는 여러 변화(기울어짐, 밝기, 노이즈…)를 미리 흉내 낸다",
      "모델이 '쓸데없는 변화'에 흔들리지 않게 만든다 → 이것을 불변성(invariance)이라 한다",
      "그 결과 모델의 강건성(robustness)이 높아진다",
    ], { x: 0.6, y: 2.7, w: 7.0, h: 3.4, size: 18 });
    d.card(s, { x: 7.9, y: 2.7, w: 4.8, h: 3.4, fill: C.orangeL, title: "쉬운 비유", tColor: C.orange, size: 16, body: "고양이를 가르칠 때 정면 사진 한 장만 보여주지 않고, 옆모습·어두운 방·멀리서 찍은 사진을 함께 보여주는 것.\n\n고양이는 똑같은 고양이다. 달라진 건 '찍힌 모습'뿐." });
    s.addNotes("핵심 단어 세 개: 레이블 보존, 불변성, 강건성. 칠판에 적어 둔다.");
  }

  {
    const s = d.slide("증강(Augmentation) vs 생성(Generation)", { tag: "1교시 · 개념 구분" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 3.0, fill: C.tealL, title: "증강: 같은 샘플의 다른 관점", tColor: C.teal, size: 17, body: "· 원본 x에서 출발한다\n· 의미(레이블)가 그대로다\n· 예: 3을 살짝 돌린 3\n· 목적: 불변성, 일반화" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 3.0, fill: C.purpleL, title: "생성: 무(無)에서 새 샘플", tColor: C.purple, size: 17, body: "· 모델(GAN, 확산 모델 등)이 새로 그린다\n· 원본과 직접 대응이 없다\n· 예: '존재하지 않는 얼굴' 만들기\n· 목적: 데이터 자체를 새로 만들기" });
    d.bullets(s, [
      "원본 교재 제목의 '증강 vs 일반화'는 '증강 vs 생성(Generation)'의 오기이다 (원본 검토 참고)",
      "경계는 흐려지는 중: 생성 모델로 만든 합성 데이터로 학습셋을 늘리는 연구도 활발하다 — 이때도 '레이블이 맞는지' 검증이 필수",
    ], { x: 0.6, y: 4.85, w: 12.1, h: 1.9, size: 16 });
  }

  {
    const s = d.slide("수식으로 보는 증강 (식 13.1)", { tag: "1교시 · 수식" });
    F(d, s, "x′ = T(x),     f(x′) ≈ f(x),     label(x′) = y", { x: 0.6, y: 1.5, w: 12.1, h: 0.9, size: 28, fill: C.tealL });
    table(s, ["기호", "뜻", "예: 손글씨 3"], [
      ["x, y", "원본 입력과 정답", "3 모양 이미지, 정답 '3'"],
      ["T", "허용 가능한 변형들의 분포에서 뽑은 변환 함수", "'−15°~15° 사이 무작위 회전'"],
      ["x′", "변환된 입력", "조금 기울어진 3"],
      ["f", "모델", "숫자 분류기"],
      ["f(x′) ≈ f(x)", "변환해도 모델의 출력이 거의 같다 = 불변성", "기울어도 '3'이라고 답한다"],
    ], { x: 0.6, y: 2.75, w: 12.1, colW: [2.6, 5.5, 4.0], fontSize: 14, rowH: 0.5 });
    d.text(s, "주의: 모든 과제가 '불변'을 원하는 것은 아니다. 물체 검출처럼 위치가 답인 과제는 이미지를 옮기면 정답(박스)도 함께 옮겨야 한다. 이것은 '불변'이 아니라 '동변(equivariance)'.", { x: 0.6, y: 6.15, w: 12.1, h: 0.8, size: 13, color: C.muted });
    s.addNotes("원본에 없는 보강: 검출/분할 과제에서는 변환을 레이블에도 똑같이 적용해야 한다. torchvision v2는 박스·마스크도 함께 변환해 준다.");
  }

  {
    const s = d.slide("언제 증강이 특히 도움이 될까?", { tag: "1교시 · 효과" });
    d.card(s, { x: 0.6, y: 1.5, w: 3.9, h: 2.3, fill: C.tealL, title: "데이터가 적을 때", tColor: C.teal, size: 15, body: "모델이 몇 장을 통째로 외우기 쉽다. 증강은 '외울 수 없을 만큼' 변형을 만든다." });
    d.card(s, { x: 4.7, y: 1.5, w: 3.9, h: 2.3, fill: C.orangeL, title: "불균형할 때", tColor: C.orange, size: 15, body: "소수 클래스만 더 많이 변형하면 균형을 맞출 수 있다 (단, 레이블 일관성 확인)." });
    d.card(s, { x: 8.8, y: 1.5, w: 3.9, h: 2.3, fill: C.purpleL, title: "다양성이 부족할 때", tColor: C.purple, size: 15, body: "학습 사진은 모두 낮, 시험 사진은 밤이라면? 밝기 증강이 다리를 놓아 준다." });
    d.text(s, "증강이 만드는 신호 (좋은 증강일 때)", { x: 0.6, y: 4.1, w: 12.1, h: 0.4, size: 18, bold: true, color: C.teal });
    table(s, ["지표", "증강 후 기대되는 변화", "이유"], [
      ["학습 손실", "약간 증가", "더 어렵고 다양한 입력을 보기 때문"],
      ["검증 손실", "감소", "외우기 대신 규칙을 배운다"],
      ["분포 밖(OOD) 정확도", "크게 증가", "기울기·노이즈 같은 변화에 강해진다"],
    ], { x: 0.6, y: 4.55, w: 12.1, colW: [2.8, 3.5, 5.8], fontSize: 15, rowH: 0.55 });
    s.addNotes("이 표는 원본의 설명이다. 3교시 실험에서 실제 숫자로 확인한다. 특히 '깨끗한 검증셋 정확도'는 크게 안 오를 수 있고, '변형이 낀 검증셋'에서 크게 오른다는 점을 미리 예고한다.");
  }

  {
    const s = d.slide("이미지는 숫자의 표다", { tag: "1교시 · 준비" });
    d.text(s, "8×8 손글씨 숫자 '3': 각 칸은 0(흰색) ~ 16(검정)의 밝기. 이 강의에서는 0~1로 나누어 쓴다.", { x: 0.6, y: 1.4, w: 12.1, h: 0.5, size: 17 });
    const g = [[0, 1, 8, 12, 15, 14, 4, 0], [0, 3, 11, 8, 8, 12, 12, 0], [0, 0, 0, 0, 2, 13, 7, 0]];
    const hi = (i, j) => (g[i][j] >= 12 ? "9AA5AD" : g[i][j] >= 6 ? "C9D2D8" : null);
    d.grid(s, g, { x: 0.8, y: 2.2, cw: 0.78, ch: 0.5, size: 14, hi });
    d.text(s, "…(8줄 중 위쪽 3줄만 표시)", { x: 0.8, y: 3.75, w: 6, h: 0.4, size: 13, color: C.muted });
    d.bullets(s, [
      "'돌리기·밀기·뒤집기'는 이 표의 숫자를 다른 칸으로 옮기는 일",
      "'밝기·대비·노이즈'는 칸 안의 숫자 값을 바꾸는 일",
      "그래서 NumPy 배열 연산만으로 구현할 수 있다 (2교시)",
    ], { x: 7.3, y: 2.2, w: 5.4, h: 2.7, size: 17 });
    d.code(s, ["from sklearn.datasets import load_digits", "X = load_digits().images / 16.0   # (1797, 8, 8), 값 0~1", "print(X[0].shape, X[0].max())"], { x: 0.6, y: 4.9, w: 12.1, h: 1.5, size: 14, title: "step1_transforms.py" });
    s.addNotes("표는 transforms.png의 '원본' 3 이미지의 위쪽 3줄 실제 값(16 만점). 위치를 옮기는 변환과 값을 바꾸는 변환 두 종류로 나뉜다는 구분이 2교시의 뼈대.");
  }

  {
    const s = d.slide("한눈에 보는 변환들", { tag: "1교시 · 미리보기" });
    img(s, "transforms", 0.6, 1.45, 7.6);
    d.bullets(s, [
      { t: "위치를 바꾸는 변환", b: true },
      "뒤집기 · 이동 · 회전 · 확대",
      { t: "값을 바꾸는 변환", b: true },
      "밝기 · 대비 · 노이즈",
      "전부 정답은 그대로 '3'… 일까?",
      "→ 다음 장: 그렇지 않은 경우",
    ], { x: 8.6, y: 1.6, w: 4.2, h: 4.5, size: 17 });
    d.text(s, "그림: transforms.png (make_images.py)", { x: 0.6, y: 5.85, w: 8, h: 0.3, size: 12, color: C.muted });
  }

  // 레이블 일관성
  {
    const s = d.slide("가장 중요한 규칙: 레이블 일관성", { tag: "1교시 · 레이블 일관성" });
    img(s, "label_break", 0.6, 1.45, 12.1);
    d.bullets(s, [
      { t: "6을 180° 돌리면 9와 구분되지 않는다. 그래도 정답을 '6'으로 두면 모델에게 거짓말을 가르치는 것", b: true },
      "이것은 증강이 아니라 의미적 손상(semantic corruption)이다 — 학습에 노이즈가 섞이고 성능이 떨어진다",
      "숫자 2를 좌우로 뒤집으면 3처럼 보인다. 사람이 봐도 헷갈리는 변환은 의심한다",
    ], { x: 0.6, y: 4.5, w: 12.1, h: 2.3, size: 17 });
    s.addNotes("원본의 '손성(損性)'은 '손상'의 오타. 오른쪽 두 쌍이 이번 시간의 대표 반례.");
  }

  {
    const s = d.slide("실험으로 확인: 변환 후 정답이 얼마나 유지되나?", { tag: "1교시 · 실험 Step 2" });
    d.code(s, [
      "means = 숫자별 평균 이미지 (10장)",
      "def guess(img):                      # 평균 이미지와 가장 닮은 숫자",
      "    return argmin(sum((means - img)**2))",
      "six = 6인 이미지들 (181장)",
      "after = [guess(rotate(img,180)) for img in six]",
      "# 결과: 9로 판정 148장 (82%), 6으로 판정 0장",
    ], { x: 0.5, y: 1.5, w: 7.2, h: 3.2, size: 13, title: "step2_label.py" });
    d.explain(s, ["아주 단순한 판정기: 숫자별 평균 그림과 비교", "6을 180° 돌린 이미지 181장 중 148장(82%)이 '9'로 읽힌다", "즉 이 변환은 정답을 바꿔 버린다 → 6에는 쓰면 안 된다", "좌우 뒤집기도 같은 이유로 숫자에 부적합"], { x: 7.9, y: 1.5, w: 4.8, h: 3.2, title: "결과 읽는 법" });
    d.card(s, { x: 0.6, y: 4.95, w: 12.1, h: 1.75, fill: C.orangeL, title: "한 가지 더 알게 된 점", tColor: C.orange, size: 15, body: "이 평균-이미지 판정기는 숫자를 1칸만 옮겨도 정답 유지율이 90% → 30%로 떨어진다. 즉 이동에 '불변'이 아니라 아주 취약하다. 이런 약점을 증강으로 메우는 것이 오늘의 목표이다." });
    s.addNotes("실제 실행 결과(code/step2_label.py): 변환 없음 0.90, 이동 1칸 0.30, 회전 10도 0.86, 15도 0.80, 45도 0.30, 90도 0.13, 좌우 뒤집기 0.36, 회전 180도 0.39. 6을 180도 돌리면 9로 판정 148 / 181.");
  }

  {
    const s = d.slide("도메인마다 '허용되는 변환'은 다르다", { tag: "1교시 · 도메인" });
    const ok = { text: "○", options: { align: "center", color: C.green, bold: true } };
    const no = { text: "✕", options: { align: "center", color: C.red, bold: true } };
    const mid = { text: "△", options: { align: "center", color: C.orange, bold: true } };
    table(s, ["데이터", "좌우 뒤집기", "작은 회전(±15°)", "180° 회전", "밝기·노이즈", "이유"], [
      ["손글씨 숫자", no, ok, no, ok, "2·5, 6·9가 서로 바뀐다"],
      ["고양이/개 사진", ok, ok, no, ok, "거꾸로 선 고양이는 현실에 드물다"],
      ["교통 표지판", no, mid, no, ok, "'좌회전'과 '우회전'이 뒤집힌다"],
      ["흉부 X-선", mid, mid, no, mid, "심장 위치(왼쪽)가 진단 정보일 수 있다"],
      ["위성/현미경 사진", ok, ok, ok, ok, "위쪽이 어디인지 정해져 있지 않다"],
      ["문자(b / d)", no, mid, no, ok, "뒤집으면 다른 글자"],
    ], { x: 0.5, y: 1.5, w: 12.3, colW: [2.1, 1.6, 1.9, 1.5, 1.6, 3.6], fontSize: 14, rowH: 0.62 });
    d.text(s, "○ 대체로 안전   △ 조건부 (도메인 전문가에게 확인)   ✕ 정답이 바뀔 수 있음", { x: 0.5, y: 6.2, w: 12.3, h: 0.4, size: 14, color: C.muted });
    s.addNotes("'△'는 상황에 따라 다르다는 뜻. 이 표는 일반적 지침이며 최종 판단은 도메인 전문가와 눈으로 검증.");
  }

  {
    const s = d.slide("1교시 정리", { tag: "1교시 · 정리" });
    d.bullets(s, [
      { t: "증강 = 레이블을 지키는 변환으로 같은 샘플의 다른 모습을 만드는 것 (x′ = T(x), y 그대로)", b: true },
      "생성과 다르다: 무에서 만들지 않는다",
      "목적은 불변성과 강건성, 특히 데이터가 적을 때 과적합을 줄이는 것",
      "가장 중요한 규칙은 레이블 일관성: 6↔9, 2↔3(뒤집기)처럼 정답이 바뀌는 변환은 금지",
      "허용되는 변환은 도메인마다 다르다",
    ], { x: 0.6, y: 1.6, w: 12.1, h: 3.6, size: 19 });
    d.card(s, { x: 0.6, y: 5.3, w: 12.1, h: 1.4, fill: C.tealL, title: "생각해 보기", tColor: C.teal, size: 15, body: "내가 만들고 싶은 AI(예: 번호판 인식, 피부 사진 진단)에서 안전한 변환 3개와 위험한 변환 1개를 골라 보자. 이유와 함께!" });
    s.addNotes("10분 휴식 전에 옆 사람과 2분간 토론하게 한다.");
  }
};
