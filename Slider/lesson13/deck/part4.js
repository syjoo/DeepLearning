// 실습 문제, 솔루션, 원본 자료 검토 결과
const { C, FONT } = require("./lib");
const { F, table } = require("./helpers");

module.exports = function part4(d, pres) {
  d.section("실습", "실습 문제 5개", "판단 · 계산 · 구현 · 예측 후 실험  (솔루션은 문제 뒤에 이어집니다)", "문제를 먼저 풀 시간을 충분히 준다(약 20분). 솔루션 슬라이드는 풀이 후에 연다.");

  {
    const s = d.slide("실습 문제 ①~③", { tag: "실습 문제" });
    d.card(s, { x: 0.6, y: 1.5, w: 12.1, h: 1.75, fill: C.tealL, title: "문제 1. 변환은 정답을 지키는가? (○ / △ / ✕ 와 이유)", tColor: C.teal, size: 15, body: "(a) 교통 표지판 '좌회전 금지'를 좌우로 뒤집기   (b) 고양이 사진을 좌우로 뒤집기   (c) 손글씨 '6'을 180° 회전   (d) 위성 사진을 90° 회전   (e) 흉부 X-선을 좌우로 뒤집기" });
    d.card(s, { x: 0.6, y: 3.4, w: 12.1, h: 1.55, fill: C.orangeL, title: "문제 2. 오프라인 증강의 저장 용량", tColor: C.orange, size: 15, body: "28×28 흑백(1바이트/픽셀) 1,000장과 224×224 컬러(3채널) 1,000장을 각각 10배로 증강해 저장한다. 필요한 용량은? 온라인 증강으로 100 에포크 학습하면 한 장당 몇 번 다른 모습을 보게 되나?" });
    d.card(s, { x: 0.6, y: 5.1, w: 12.1, h: 1.6, fill: C.purpleL, title: "문제 3. Mixup 손계산", tColor: C.purple, size: 15, body: "λ = 0.7, x₁ = [1, 0, 0.4] (정답: 고양이), x₂ = [0, 1, 0.8] (정답: 개). 섞인 입력 x̃ 와 섞인 정답 ỹ 를 구하라. (정답은 원-핫: 고양이=[1,0,0], 개=[0,1,0])" });
    s.addNotes("5분씩 풀게 한다.");
  }
  {
    const s = d.slide("실습 문제 ④~⑤", { tag: "실습 문제" });
    d.card(s, { x: 0.6, y: 1.5, w: 12.1, h: 2.3, fill: C.greenL, title: "문제 4. 밝기·대비 무작위 변환 구현", tColor: C.green, size: 15, body: "random_brightness_contrast(img, rng) 를 구현하라. 밝기 b는 [−0.2, 0.2], 대비 c는 [0.8, 1.2] 에서 무작위로 뽑고 (img − 0.5)·c + 0.5 + b 를 0~1로 자른다.\n검사: ① 결과가 항상 0~1  ② b=0, c=1 이면 원본과 같다." });
    d.card(s, { x: 0.6, y: 4.0, w: 12.1, h: 2.7, fill: C.redL, title: "문제 5. 예측 후 실험: 증강을 분할 전에 하면?", tColor: C.red, size: 15, body: "원본 300장을 5배로 증강한 뒤 (1,500장) 섞어서 1,000장 학습 / 500장 검증으로 나눈다 (A).\n원본을 200 / 100장으로 먼저 나누고 학습 쪽만 5배로 증강한다 (B).\n예측: A와 B 중 검증 점수가 더 높게 나오는 쪽은? 그 이유는? 코드(practice_solutions.py)를 돌려 확인하라." });
    s.addNotes("문제 5의 포인트: 같은 원본의 변형본이 학습과 검증 양쪽에 있으면 검증이 쉬워진다(데이터 누수).");
  }

  d.section("솔루션", "실습 문제 풀이", "코드: lesson13/code/practice_solutions.py", "풀이를 하나씩 공개한다.");

  {
    const s = d.slide("솔루션 ①: 변환은 정답을 지키는가?", { tag: "솔루션" });
    const ok = { text: "○", options: { align: "center", color: C.green, bold: true } };
    const no = { text: "✕", options: { align: "center", color: C.red, bold: true } };
    const mid = { text: "△", options: { align: "center", color: C.orange, bold: true } };
    table(s, ["", "상황", "판정", "이유"], [
      ["(a)", "표지판 '좌회전 금지' 좌우 뒤집기", no, "우회전 금지가 되어 의미(정답)가 바뀐다"],
      ["(b)", "고양이 좌우 뒤집기", ok, "고양이는 좌우 대칭에 가깝고, 뒤집어도 고양이"],
      ["(c)", "'6'을 180° 회전", no, "'9'와 구분되지 않는다 (실험: 82%가 9로 읽힘)"],
      ["(d)", "위성 사진 90° 회전", ok, "위쪽이 어디인지 정해져 있지 않다"],
      ["(e)", "흉부 X-선 좌우 뒤집기", mid, "심장이 왼쪽에 있다는 정보가 진단에 쓰일 수 있음 → 전문가 확인 필요"],
    ], { x: 0.6, y: 1.5, w: 12.1, colW: [0.8, 4.3, 1.0, 6.0], fontSize: 15, rowH: 0.72 });
    d.text(s, "원칙: 정답은 '데이터 종류와 과제'가 결정한다. 같은 변환도 도메인에 따라 ○가 ✕가 된다.", { x: 0.6, y: 5.8, w: 12.1, h: 0.5, size: 16, color: C.muted });
  }

  {
    const s = d.slide("솔루션 ②: 오프라인 저장 용량", { tag: "솔루션" });
    F(d, s, "28×28×1 B × 1,000 × 10 = 7,840,000 B ≈ 7.84 MB", { x: 0.6, y: 1.6, w: 12.1, h: 0.85, size: 22, fill: C.tealL });
    F(d, s, "224×224×3 B × 1,000 × 10 = 1,505,280,000 B ≈ 1.51 GB", { x: 0.6, y: 2.7, w: 12.1, h: 0.85, size: 22, fill: C.orangeL });
    d.bullets(s, ["온라인 100 에포크: 한 장이 100번, 매번 새로 뽑은 변형으로 등장한다 (1,000장이면 총 100,000번). 추가 저장 용량은 0", "(JPEG 같은 압축을 쓰면 실제 파일 용량은 더 작지만, 학습 중 메모리에 올라가는 크기는 위 계산과 같다)"], { x: 0.6, y: 3.9, w: 12.1, h: 2.5, size: 17 });
    d.code(s, ["storage = lambda h, w, c, n, k: h * w * c * n * k", "storage(28, 28, 1, 1000, 10)    # 7840000", "storage(224, 224, 3, 1000, 10)  # 1505280000"], { x: 0.6, y: 5.6, w: 12.1, h: 1.2, size: 13 });
  }

  {
    const s = d.slide("솔루션 ③: Mixup 손계산", { tag: "솔루션" });
    table(s, ["항목", "계산", "결과"], [
      ["x̃ 첫째 값", "0.7×1 + 0.3×0", "0.7"],
      ["x̃ 둘째 값", "0.7×0 + 0.3×1", "0.3"],
      ["x̃ 셋째 값", "0.7×0.4 + 0.3×0.8 = 0.28 + 0.24", "0.52"],
      ["ỹ", "0.7×[1,0,0] + 0.3×[0,1,0]", "[0.7, 0.3, 0]"],
    ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.6, 6.5, 3.0], fontSize: 17, rowH: 0.7 });
    d.code(s, ["lam = 0.7", "lam*x1 + (1-lam)*x2   # [0.7  0.3  0.52]", "lam*y1 + (1-lam)*y2   # [0.7  0.3  0.  ]"], { x: 0.6, y: 4.7, w: 7.2, h: 1.6, size: 14, title: "practice_solutions.py" });
    d.text(s, "모델은 '이 사진은 고양이 70%, 개 30%'라는 부드러운 정답을 배우게 된다. 그래서 정답을 너무 확신하는 버릇이 줄어든다.", { x: 8.1, y: 4.7, w: 4.6, h: 1.8, size: 15, color: C.ink });
  }

  {
    const s = d.slide("솔루션 ④: 밝기·대비 무작위 변환", { tag: "솔루션" });
    d.code(s, [
      "def random_brightness_contrast(img, rng,",
      "                               b_max=0.2, c_lo=0.8, c_hi=1.2):",
      "    b = rng.uniform(-b_max, b_max)       # 이미지 하나는 하나의 b",
      "    c = rng.uniform(c_lo, c_hi)          # 이미지 하나는 하나의 c",
      "    return np.clip((img - 0.5) * c + 0.5 + b, 0, 1)",
      "",
      "# 검사",
      "assert 결과가 항상 0~1                  # 100번 반복 확인",
      "assert b=0, c=1 이면 원본과 같다",
      "# 100번 적용한 평균 밝기: 0.184~0.494 (원본 0.287)",
    ], { x: 0.4, y: 1.45, w: 7.9, h: 4.6, size: 12.5, title: "practice_solutions.py" });
    d.explain(s, ["b와 c를 '이미지마다 한 번' 뽑는다 (픽셀마다 뽑으면 노이즈가 된다)", "대비는 중간 회색 0.5를 기준으로 늘리고 줄인다", "np.clip이 없으면 값이 0~1을 벗어난다", "검사 두 개: 범위, 항등"], { x: 8.5, y: 1.45, w: 4.3, h: 4.6 });
    d.text(s, "자주 하는 실수: b를 픽셀마다 따로 뽑기 → 밝기 변화가 아니라 노이즈가 된다.", { x: 0.4, y: 6.2, w: 12.4, h: 0.5, size: 15, color: C.muted });
  }

  {
    const s = d.slide("솔루션 ⑤: 증강 → 분할이면 검증 점수가 부풀려진다", { tag: "솔루션" });
    d.bullets(s, [
      { t: "예측: A(증강 → 분할)의 검증 점수가 더 높다. 같은 원본의 변형본이 학습에도 들어 있어 검증이 '이미 본 문제의 쌍둥이'가 되기 때문", b: true },
      "측정 (5개 시드 평균, 변형본끼리 비교):  A 86.1%  vs  B 79.2%  → 6.9%p 부풀려짐",
      "완전히 따로 둔 원본 시험 점수는 A 93.0%, B 92.3% 로 거의 같다 → 모델 성능은 같은데 '검증 점수'만 거짓으로 높았다",
      "규칙: 분할 먼저, 증강은 학습 데이터에만",
    ], { x: 0.6, y: 1.5, w: 12.1, h: 3.1, size: 17 });
    d.code(s, ["# A: 늘린 뒤 분할  (나쁨)", "xs = cat([X] + [aug(X) for _ in range(4)]);  tr, va = split(xs)", "# B: 분할 뒤 학습쪽만 증강  (좋음)", "tr_o, va_o = split(X);  xs = cat([X[tr_o]] + [aug(X[tr_o]) for _ in range(4)])"], { x: 0.6, y: 4.85, w: 12.1, h: 1.9, size: 13, title: "practice_solutions.py" });
    s.addNotes("실제 실행 결과: 문제5: 늘린 뒤 분할 -> 검증 86.1% (진짜 시험 93.0%) / 분할 뒤 늘림 -> 검증 79.2% (진짜 시험 92.3%).");
  }

  // ---------------- 원본 자료 검토 ----------------
  {
    const s = d.slide("원본 자료 검토 결과 (강사용): 바로잡고 보강한 내용", { tag: "APPENDIX" });
    const c = (t) => ({ text: t, options: { align: "left" } });
    table(s, ["원본 위치", "문제", "이 강의에서의 처리"], [
      [c("'증강 vs. 일반화' 소제목"), c("영문은 Augmentation vs. Generation 인데 한글이 '일반화'. 본문도 '생성'을 설명"), c("'증강 vs. 생성'으로 정정")],
      [c("레이블 일관성 단락"), c("'의미적 손성'(오타). '손상'이 맞음"), c("손상으로 정정")],
      [c("식 13.1"), c("f(x′)≈f(x)는 '불변성'만 다룸. 검출·분할처럼 위치가 정답인 과제는 동변(equivariance)이라 레이블도 함께 변환해야 함"), c("[보강] 설명 슬라이드 추가")],
      [c("'언제 도움을 주나' 단락"), c("'검증 손실이 눈에 띄게 감소'는 조건부. 이 실험에서는 오차 범위, 분포 밖 정확도에서만 뚜렷"), c("[검증] 실제 숫자 제시, 학습 길이 함정 추가")],
      [c("Implementation"), c("Albumentations만 소개. 그림 13.1(증강 예제)은 원본 md에서 이미지 파일이 없어 확인 불가"), c("torchvision.transforms.v2 추가(실행 확인), Albumentations는 미설치로 미검증 표기")],
      [c("전반 / 파일명"), c("Mixup·CutMix·RandAugment·TTA 등 최신 기법, 데이터 누수 주의 없음. 파일명 'Foundataion' 오타"), c("[최신 자료] 슬라이드 추가 / 파일명은 원본 유지")],
    ], { x: 0.4, y: 1.45, w: 12.5, colW: [2.5, 6.0, 4.0], fontSize: 12, rowH: 0.7 });
    s.addNotes("교재 오류 검사 결과를 투명하게 공유. 학생에게는 필요할 때만 언급. 원본 md 의 그림 파일(assets\\image-...png)은 디렉터리에 없어 내용을 확인하지 못했다.");
  }

  {
    const s = pres.addSlide(); d.n += 1; s.background = { color: C.paper };
    s.addText("수고하셨습니다!", { x: 0.9, y: 2.4, w: 11, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    s.addText("오늘의 한 줄: 증강은 '현장에서 일어날 변화'에 대한 보험이다. 단, 정답을 지키는 선에서!", { x: 0.9, y: 3.6, w: 11.5, h: 0.6, fontFace: FONT, fontSize: 20, color: C.teal, bold: true, margin: 0 });
    s.addText("튜링랩 프로그래밍학원  |  TuringLab. Programming Academy", { x: 0.9, y: 5.2, w: 11, h: 0.4, fontFace: FONT, fontSize: 16, color: C.muted, margin: 0 });
  }
};
