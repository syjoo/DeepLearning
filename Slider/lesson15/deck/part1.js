// 1교시: 왜 설명이 필요한가 · 설명 기법 지도 · 사례 소개 · Grad-CAM
const path = require("path");
const { C, FONT, IMG } = require("./lib");
const { img, F, table } = require("./helpers");

module.exports = function part1(d, pres) {
  // ---------------- 표지 ----------------
  {
    const s = pres.addSlide(); d.n += 1; s.background = { color: C.paper };
    s.addText("Lesson 15", { x: 0.9, y: 1.3, w: 11, h: 0.7, fontFace: FONT, fontSize: 28, bold: true, color: C.teal, margin: 0 });
    s.addText("Explainability", { x: 0.9, y: 2.0, w: 11.5, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    s.addText("설명 가능성: 모델은 왜 그렇게 답했을까?", { x: 0.9, y: 3.1, w: 11.5, h: 0.6, fontFace: FONT, fontSize: 24, color: C.teal, bold: true, margin: 0 });
    s.addText("Grad-CAM · 순열 중요도 · SHAP(샤플리 값) · 어텐션 · 사기 탐지 사례", { x: 0.9, y: 3.8, w: 11.5, h: 0.5, fontFace: FONT, fontSize: 18, color: C.muted, margin: 0 });
    ["정확하기만 하면", "충분할까?", "이유를 말해 봐!"].forEach((t, i) => d.box(s, t, { x: 0.9 + i * 2.5, y: 4.8, w: 2.3, h: 0.7, fill: [C.orangeL, C.tealL, C.greenL][i], size: 20, bold: true }));
    s.addText("튜링랩 프로그래밍학원\nTuringLab. Programming Academy", { x: 0.9, y: 5.9, w: 6, h: 0.7, fontFace: FONT, fontSize: 14, color: C.ink, margin: 0 });
    s.addImage({ path: path.join(IMG, "logo.png"), x: 9.5, y: 6.0, w: 3.0, h: 3.0 * 118 / 756 });
    s.addNotes("강의 소개. 오늘 3시간(50분 x 3교시)의 흐름: 1교시 설명 가능성이 왜 필요한지와 Grad-CAM, 2교시 순열 중요도와 SHAP(샤플리 값), 3교시 어텐션·종합 사례 실험·한계·실습 문제. 슬라이드의 모든 숫자는 직접 돌려서 얻은 값이다.");
  }

  // ---------------- 로드맵 ----------------
  {
    const s = d.slide("오늘의 로드맵: 3교시 (50분 수업 + 10분 휴식) × 3", { tag: "OVERVIEW" });
    table(s, ["교시", "주제", "직접 해보는 것"], [
      [{ text: "1교시", options: { bold: true } }, "설명 가능성이 필요한 이유 · 사례 연구 소개 · Grad-CAM", "Step 3·4  NumPy/PyTorch 로 Grad-CAM, '도장' 지름길 찾기"],
      [{ text: "2교시", options: { bold: true } }, "순열 중요도(식 15.3) · SHAP/샤플리 값(식 15.4)", "Step 1·2  순열 중요도, 정확한 샤플리 값 계산"],
      [{ text: "3교시", options: { bold: true } }, "어텐션 가중치(식 15.5) · 사기 탐지 종합 사례 · 한계", "Step 5·6  어텐션 해부, 사기 탐지 모델 감사"],
      [{ text: "마무리", options: { bold: true } }, "요약 · 원본 검토 결과 · 실습 문제 5개와 솔루션", "손계산 + 코드로 검증"],
    ], { x: 0.6, y: 1.6, w: 12.1, colW: [1.5, 6.4, 4.2], fontSize: 16, rowH: 0.7 });
    d.text(s, "준비물: Python 3.9+, NumPy, PyTorch, scikit-learn(손글씨 데이터), matplotlib   |   코드는 lesson15/code 폴더 (NumPy 먼저, PyTorch 는 확인용)", { x: 0.6, y: 5.7, w: 12.1, h: 0.5, size: 15, color: C.muted });
  }

  // ---------------- 학습 목표 ----------------
  {
    const s = d.slide("이 강의가 끝나면 할 수 있는 일", { tag: "GOAL" });
    const items = [
      ["1", "설명하기", "설명 가능성이 왜 필요한지, 전역/국소 설명의 차이를 내 말로 설명한다"],
      ["2", "계산하기", "Grad-CAM, 순열 중요도, 샤플리 값, 어텐션 가중치를 손과 NumPy 로 직접 계산한다"],
      ["3", "찾아내기", "모델이 엉뚱한 지름길(도장, 도시A 같은 단서)에 기대고 있는지 설명 기법으로 가려낸다"],
      ["4", "의심하기", "각 기법의 한계를 알고, 설명 결과를 '진단을 위한 관점'으로만 쓴다"],
    ];
    items.forEach((it, i) => {
      const y = 1.6 + i * 1.2;
      s.addText(it[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 22, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, it[1], { x: 1.7, y, w: 3.2, h: 0.5, size: 20, bold: true });
      d.text(s, it[2], { x: 1.7, y: y + 0.47, w: 10.8, h: 0.5, size: 16, color: C.muted });
    });
  }

  // ---------------- 1교시 ----------------
  d.section("1교시", "왜 모델의 이유를 알아야 하는가", "설명 가능성 · 딥러닝이 어려운 이유 · 설명 기법 지도 · 사기 탐지 사례 · Grad-CAM", "1교시 시작. 앞 시간까지는 모델을 더 정확하게, 더 안정적으로 학습시키는 법을 배웠다(최적화, 증강). 이번 주제는 정확한 모델이 '왜' 그렇게 답했는지 들여다보는 법이다.");

  {
    const s = d.slide("설명 가능성(Explainability)이란?", { tag: "1교시 · 정의" });
    F(d, s, "설명 가능성 = 모델이 특정 출력을 낸 '이유'에 대한 통찰", { x: 0.6, y: 1.5, w: 12.1, h: 0.8, size: 24, fill: C.tealL });
    d.card(s, { x: 0.6, y: 2.65, w: 2.9, h: 2.4, fill: C.greenL, title: "① 신뢰", tColor: C.green, size: 15, body: "사용자가 결과를 믿고 쓸 수 있게 한다" });
    d.card(s, { x: 3.7, y: 2.65, w: 2.9, h: 2.4, fill: C.orangeL, title: "② 디버깅", tColor: C.orange, size: 15, body: "개발자가 예상 밖의 동작 원인을 찾는다" });
    d.card(s, { x: 6.8, y: 2.65, w: 2.9, h: 2.4, fill: C.purpleL, title: "③ 규제 준수", tColor: C.purple, size: 15, body: "금융·의료는 판단 근거 제시를 요구한다" });
    d.card(s, { x: 9.9, y: 2.65, w: 2.8, h: 2.4, fill: C.redL, title: "④ 공정성·윤리", tColor: C.red, size: 15, body: "편향된 단서에 기대는지 발견한다" });
    d.bullets(s, [
      { t: "정확도만으로는 부족한 분야: 의료, 금융, 자율 시스템, 보안", b: true },
      "궁극적으로, 설명할 수 없는 모델은 신뢰할 수 없는 모델이다",
    ], { x: 0.6, y: 5.35, w: 12.1, h: 1.4, size: 18 });
    s.addNotes("비유: 의사가 '검사 결과 위험합니다'라고만 하면 환자는 불안하다. '이 수치가 이래서 위험합니다'라고 말해야 수긍한다. 모델도 같다.");
  }

  {
    const s = d.slide("딥러닝은 왜 해석하기 어려운가", { tag: "1교시 · 문제" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 2.5, fill: C.greenL, title: "선형 모델 · 결정 트리: 투명하다", tColor: C.green, size: 16, body: "· 계수(w)나 분기 규칙을 그대로 읽는다\n· 예: '공부시간이 1시간 늘면 점수 +4점'\n· 구조 자체가 설명이다" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 2.5, fill: C.redL, title: "딥 신경망: 불투명하다", tColor: C.red, size: 16, body: "· 여러 층의 비선형 변환, 수백만 개 파라미터\n· 학습된 표현이 분산·얽혀 있다\n· 사후 분석 없이는 해석이 거의 불가능" });
    d.bullets(s, [
      { t: "그래서 어떤 정보가 이 결정에 영향을 줬는지 알기 어렵다", b: true },
      "허위 상관관계(spurious correlation)나 교란 변수에 기대는지 탐지하기 어렵다",
      "해결: 모델이 만들어진 뒤에 따로 분석하는 사후(post-hoc) 설명 기법들 → 오늘의 주제",
    ], { x: 0.6, y: 4.35, w: 12.1, h: 2.4, size: 18 });
    s.addNotes("허위 상관관계: 정답과 우연히 같이 다니는 단서. 예) 소 사진은 늘 초원이라서 '초원이면 소'라고 배우는 모델. 오늘 실습에서 직접 만들어 본다.");
  }

  {
    const s = d.slide("설명 기법 지도: 오늘 배울 네 가지", { tag: "1교시 · 지도" });
    table(s, ["기법", "무엇을 알려 주나", "범위", "대상 모델", "교시"], [
      [{ text: "Grad-CAM", options: { bold: true } }, "이미지의 어느 영역이 이 클래스 판단에 기여했나", "국소 (한 장)", "CNN 전용", "1"],
      [{ text: "순열 중요도", options: { bold: true } }, "특성을 섞으면 성능이 얼마나 떨어지나", "전역 (모델 전체)", "모델 불문", "2"],
      [{ text: "SHAP", options: { bold: true } }, "이 예측에 각 특성이 얼마씩 기여했나 (게임 이론)", "국소 (한 건)", "모델 불문", "2"],
      [{ text: "어텐션 가중치", options: { bold: true } }, "토큰이 서로를 얼마나 보는가 (토큰 상호작용)", "국소 (한 문장)", "트랜스포머", "3"],
    ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.0, 5.6, 1.9, 1.7, 0.9], fontSize: 15, rowH: 0.75 });
    d.bullets(s, [
      { t: "국소(local) 설명: 한 개의 예측을 설명 / 전역(global) 설명: 모델이 전반적으로 무엇을 중요하게 보는지", b: true },
      "모델 불문(model-agnostic): 모델 안을 보지 않고 입출력만 사용 / 모델 전용: 구조(기울기, 어텐션)를 사용",
    ], { x: 0.6, y: 5.6, w: 12.1, h: 1.3, size: 15 });
  }

  {
    const s = d.slide("사례 연구: 은행의 사기 탐지 (1) 문제 상황", { tag: "1교시 · 사례" });
    ["딥러닝 사기 탐지 모델 배포\n테스트 정확도 높음", "규제 준수 팀:\n각 이상 거래에\n설명을 요구", "오탐이 특정 도시 주변에\n집중되는 패턴 발견"].forEach((t, i) => {
      d.box(s, t, { x: 0.6 + i * 4.2, y: 1.6, w: 3.7, h: 1.3, fill: [C.greenL, C.orangeL, C.redL][i], size: 16, bold: true });
      if (i < 2) d.arrow(s, 4.35 + i * 4.2, 2.25, 4.75 + i * 4.2, 2.25);
    });
    d.bullets(s, [
      { t: "SHAP 설명: '거래 위치' 특성의 기여도가 높다 — 그런데 특정 가맹점 유형과 결합될 때만", b: true },
      "감사 결과: 모델이 '정상적인 행동의 조합'을 '의심스러운 활동'으로 학습했다",
      "동시에 영수증 이미지를 보는 CNN 서브모듈에 Grad-CAM 을 적용: 거래 내용이 아니라 '강조 표시된 텍스트'에 주목",
      "원인: 학습 데이터의 편향 → 모델이 진짜 단서가 아닌 지름길을 배웠다",
    ], { x: 0.6, y: 3.3, w: 12.1, h: 3.4, size: 18 });
    s.addNotes("교재의 사례를 그대로 소개한다. 3교시에 같은 구조의 가상 데이터를 직접 만들어 모델을 감사해 본다(Step 6). 사례 속 SHAP/Grad-CAM 이 서로 다른 데이터 유형(표/이미지)에 쓰였음을 강조.");
  }

  {
    const s = d.slide("사례 연구 (2) 서로 다른 수준의 설명이 서로를 보완한다", { tag: "1교시 · 사례" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 2.5, fill: C.tealL, title: "표 형태 거래 정보 → SHAP", tColor: C.teal, size: 16, body: "한 건 한 건에 대해 '거래 위치 +0.4, 온라인 가맹점 +0.1 …' 처럼 기여도를 수치로 분해한다" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 2.5, fill: C.orangeL, title: "영수증 이미지 → Grad-CAM", tColor: C.orange, size: 16, body: "CNN 이 이미지의 어느 곳을 보았는지 히트맵으로 확인한다. 강조 표시된 글자에만 꽂혀 있었다" });
    F(d, s, "두 설명 → 재학습 + 데이터 파이프라인 개선 + 규제 요구 충족", { x: 0.6, y: 4.4, w: 12.1, h: 0.8, size: 22, fill: C.greenL });
    d.text(s, "설명 기법은 '정답'을 알려 주는 도구가 아니라, 어디를 고쳐야 할지 알려 주는 진단 도구이다.", { x: 0.6, y: 5.6, w: 12.1, h: 0.8, size: 18, color: C.muted });
  }

  d.section("Grad-CAM", "CNN 은 이미지의 어디를 봤을까?", "특징 맵 · 기울기 · 식 15.1 · 식 15.2 · NumPy/PyTorch 구현 · 도장(지름길) 실험", "여기서부터 Grad-CAM. 수학이 처음 나오므로 천천히: 기울기(미분)의 의미부터 시작한다.");

  {
    const s = d.slide("Grad-CAM 한눈에 보기", { tag: "1교시 · Grad-CAM" });
    d.box(s, "입력 이미지", { x: 0.5, y: 1.8, w: 1.9, h: 0.9, fill: C.gray, size: 15, bold: true });
    d.box(s, "CNN\n마지막 합성곱 층\n특징 맵 A^k", { x: 3.0, y: 1.6, w: 2.6, h: 1.3, fill: C.tealL, size: 15, bold: true });
    d.box(s, "클래스 점수\ny^c (고양이 …)", { x: 6.2, y: 1.8, w: 2.4, h: 0.9, fill: C.orangeL, size: 15, bold: true });
    d.arrow(s, 2.4, 2.25, 3.0, 2.25); d.arrow(s, 5.6, 2.25, 6.2, 2.25);
    d.box(s, "← 역전파: 기울기 ∂y^c/∂A^k", { x: 3.0, y: 3.15, w: 5.6, h: 0.55, fill: C.redL, size: 15, bold: true });
    d.box(s, "α_k^c = 기울기의\n공간 평균 (식 15.1)", { x: 3.0, y: 4.15, w: 2.6, h: 1.0, fill: C.purpleL, size: 14, bold: true });
    d.box(s, "ReLU( Σ_k α_k^c · A^k )\n(식 15.2)", { x: 6.2, y: 4.15, w: 2.4, h: 1.0, fill: C.purpleL, size: 14, bold: true });
    d.arrow(s, 5.6, 4.65, 6.2, 4.65);
    d.box(s, "입력 크기로\n업샘플 → 히트맵", { x: 9.2, y: 4.15, w: 2.3, h: 1.0, fill: C.greenL, size: 14, bold: true });
    d.arrow(s, 8.6, 4.65, 9.2, 4.65);
    d.bullets(s, ["핵심 질문: 클래스 c 점수에 어느 특징 맵이, 그 맵의 어느 위치가 기여했는가?", "답: '점수를 올리는 방향(기울기)'으로 특징 맵을 가중합한다"], { x: 0.6, y: 5.5, w: 12.1, h: 1.3, size: 17 });
    s.addNotes("전체 지도를 먼저 보여주고 세부로 들어간다. 이후 슬라이드는 이 그림의 각 상자를 하나씩 풀어 설명한다.");
  }

  {
    const s = d.slide("준비 1: 특징 맵 A^k 는 '가중치'가 아니라 '반응 지도'", { tag: "1교시 · Grad-CAM" });
    d.card(s, { x: 0.6, y: 1.55, w: 6.0, h: 2.9, fill: C.tealL, title: "특징 맵(feature map)이란?", tColor: C.teal, size: 16, body: "· 합성곱 필터 하나가 이미지를 훑은 결과\n· k번째 필터 → k번째 맵 A^k (H×W 칸)\n· A^k_ij = 필터 k 가 (i, j) 위치에서 얼마나 강하게 반응했나\n· 예) '동그란 눈' 필터 → 눈이 있는 곳이 밝다" });
    d.card(s, { x: 6.9, y: 1.55, w: 5.8, h: 2.9, fill: C.orangeL, title: "원본 교재의 표현 바로잡기", tColor: C.orange, size: 16, body: "교재는 A^k_ij 를 '(i,j) 위치의 가중치(weight)'라고 쓴다.\n하지만 A 는 학습되는 값(가중치)이 아니라 입력에 따라 달라지는 활성값(activation)이다.\n가중치는 필터 자체(3×3 숫자들)다." });
    table(s, ["구분", "정해지는 때", "예"], [
      ["가중치(weight)", "학습 중에 조정, 학습이 끝나면 고정", "3×3 필터의 숫자들"],
      ["활성값 A^k_ij", "이미지를 넣을 때마다 새로 계산", "그 필터가 (i,j)에서 보인 반응 크기"],
    ], { x: 0.6, y: 4.8, w: 12.1, colW: [2.8, 5.0, 4.3], fontSize: 15, rowH: 0.55 });
    s.addNotes("이 슬라이드는 원본 검토 항목 1번이다. 기호 A 는 Activation 의 A 라고 기억하면 쉽다.");
  }

  {
    const s = d.slide("준비 2: 기울기는 '나사를 살짝 돌렸을 때 전구가 얼마나 밝아지나'", { tag: "1교시 · Grad-CAM" });
    d.card(s, { x: 0.6, y: 1.55, w: 6.0, h: 2.6, fill: C.orangeL, title: "비유: 조절 나사가 수만 개인 전구", tColor: C.orange, size: 16, body: "전구 밝기 = y,   나사 하나의 위치 = A.\n나사 하나만 살짝 돌려 보고, 나머지는 '대리석처럼 꼼짝 않게' 고정한다. 밝기가 얼마나 변하는지가 ∂y/∂A (편미분)." });
    d.card(s, { x: 6.9, y: 1.55, w: 5.8, h: 2.6, fill: C.tealL, title: "읽는 법: 분자 = 결과, 분모 = 원인", tColor: C.teal, size: 16, body: "∂y^c / ∂A^k_ij\n= 'A^k_ij 를 아주 조금 바꾸면 클래스 c 점수 y^c 가 얼마나 변하나'\n기호 ∂ (라운드 디) = '다른 건 고정하고 이것만'" });
    table(s, ["부호", "뜻", "Grad-CAM 에서"], [
      ["∂y^c/∂A > 0", "그 칸이 밝아지면 클래스 c 점수가 오른다", "찬성 증거"],
      ["∂y^c/∂A < 0", "그 칸이 밝아지면 클래스 c 점수가 내려간다", "반대 증거"],
      ["∂y^c/∂A ≈ 0", "그 칸은 클래스 c 와 상관이 없다", "무관"],
    ], { x: 0.6, y: 4.5, w: 12.1, colW: [2.6, 6.2, 3.3], fontSize: 15, rowH: 0.55 });
    s.addNotes("교재 요약표(전구/나사)를 활용. 왜 하필 클래스 c 에 대해서만 미분하나? 정답 클래스 점수가 어디서 왔는지 알고 싶기 때문. 다른 클래스 점수는 무시한다.");
  }

  {
    const s = d.slide("식 15.1: 채널 중요도 α 구하기", { tag: "1교시 · 식 15.1" });
    F(d, s, "α_k^c  =  (1/Z) · Σ_i Σ_j  ∂y^c / ∂A^k_ij", { x: 0.6, y: 1.5, w: 12.1, h: 0.9, size: 28, fill: C.tealL });
    table(s, ["기호", "뜻", "쉬운 말"], [
      ["y^c", "클래스 c 의 점수 (softmax 이전 값)", "'고양이' 점수"],
      ["A^k_ij", "k번째 특징 맵의 (i, j) 칸 값", "필터 k 의 (i,j) 반응"],
      ["∂y^c/∂A^k_ij", "그 칸이 점수에 미치는 영향", "나사 한 개의 밝기 영향"],
      ["Z = H·W", "특징 맵의 칸 수", "평균 내려고 나누는 수"],
      ["α_k^c", "k번째 맵의 중요도 (한 숫자)", "필터 k 가 고양이 판정에 얼마나 중요?"],
    ], { x: 0.6, y: 2.7, w: 12.1, colW: [2.6, 5.3, 4.2], fontSize: 15, rowH: 0.52 });
    d.text(s, "한 줄 요약: 각 채널(필터)마다 모든 칸의 기울기를 평균 내어 '이 필터는 클래스 c 에 찬성인가 반대인가, 얼마나 세게?'를 한 숫자로 만든다.", { x: 0.6, y: 5.9, w: 12.1, h: 0.8, size: 16, color: C.muted });
  }

  {
    const s = d.slide("식 15.1 손계산: 2×2 특징 맵 두 장", { tag: "1교시 · 식 15.1" });
    d.text(s, "특징 맵 A¹, A²  (클래스 c 의 기울기 ∂y/∂A 도 함께 주어졌다)", { x: 0.6, y: 1.5, w: 12.1, h: 0.4, size: 17, bold: true, color: C.teal });
    d.text(s, "A¹", { x: 0.6, y: 2.15, w: 0.6, h: 0.4, size: 18, bold: true });
    d.grid(s, [[1, 0], [2, 1]], { x: 1.2, y: 2.1, cw: 0.8, ch: 0.55, size: 16 });
    d.text(s, "∂y/∂A¹", { x: 3.0, y: 2.2, w: 1.3, h: 0.4, size: 15, bold: true });
    d.grid(s, [[2, 2], [2, 2]], { x: 4.3, y: 2.1, cw: 0.8, ch: 0.55, size: 16, fill: C.greenL });
    d.text(s, "A²", { x: 6.6, y: 2.15, w: 0.6, h: 0.4, size: 18, bold: true });
    d.grid(s, [[0, 2], [1, 0]], { x: 7.2, y: 2.1, cw: 0.8, ch: 0.55, size: 16 });
    d.text(s, "∂y/∂A²", { x: 9.0, y: 2.2, w: 1.3, h: 0.4, size: 15, bold: true });
    d.grid(s, [[-1, -1], [-1, -1]], { x: 10.3, y: 2.1, cw: 0.8, ch: 0.55, size: 16, fill: C.redL });
    F(d, s, "α¹ = (2+2+2+2) / 4 = 2          α² = (−1−1−1−1) / 4 = −1", { x: 0.6, y: 3.5, w: 12.1, h: 0.8, size: 24, fill: C.tealL });
    d.bullets(s, [
      { t: "α¹ = +2 : 채널 1 이 켜지면 클래스 점수가 오른다 → 찬성 증거", b: true },
      { t: "α² = −1 : 채널 2 가 켜지면 점수가 내려간다 → 반대 증거", b: true },
      "Z = H·W = 4, 기울기를 모두 더해 4로 나눈 것 = 그냥 평균",
    ], { x: 0.6, y: 4.7, w: 12.1, h: 2.0, size: 18 });
    s.addNotes("실습 문제 3번과 같은 숫자이다. 학생이 직접 계산해 보게 시간을 준다. 다음 슬라이드에서 이어서 L 을 구한다.");
  }

  {
    const s = d.slide("식 15.2: 가중합 후 ReLU → 히트맵", { tag: "1교시 · 식 15.2" });
    F(d, s, "L^c_Grad-CAM  =  ReLU( Σ_k α_k^c · A^k )", { x: 0.6, y: 1.5, w: 12.1, h: 0.9, size: 28, fill: C.tealL });
    d.text(s, "앞 슬라이드의 숫자로 계속:  2·A¹ + (−1)·A²", { x: 0.6, y: 2.65, w: 12.1, h: 0.4, size: 17, bold: true, color: C.teal });
    d.grid(s, [[2, 0], [4, 2]], { x: 0.7, y: 3.2, cw: 0.8, ch: 0.55, size: 16 });
    d.text(s, "−", { x: 2.35, y: 3.3, w: 0.5, h: 0.5, size: 22, bold: true, align: "center" });
    d.grid(s, [[0, 2], [1, 0]], { x: 2.9, y: 3.2, cw: 0.8, ch: 0.55, size: 16 });
    d.text(s, "=", { x: 4.55, y: 3.3, w: 0.5, h: 0.5, size: 22, bold: true, align: "center" });
    d.grid(s, [[2, -2], [3, 2]], { x: 5.2, y: 3.2, cw: 0.8, ch: 0.55, size: 16, fill: C.orangeL });
    d.text(s, "ReLU →", { x: 6.85, y: 3.35, w: 1.1, h: 0.4, size: 16, bold: true });
    d.grid(s, [[2, 0], [3, 2]], { x: 8.0, y: 3.2, cw: 0.8, ch: 0.55, size: 16, fill: C.greenL });
    d.card(s, { x: 9.9, y: 2.95, w: 2.8, h: 1.95, fill: C.paper, title: "ReLU(x)=max(0,x)", tColor: C.teal, size: 13, body: "음수는 0, 양수는 그대로.\n'클래스 c 에 불리한 영역'은 지우고 유리한 근거만 남긴다." });
    d.bullets(s, [
      { t: "주의: ReLU 는 맵별 α 가 아니라 '가중합 결과(칸별)'에 적용된다. 음수 α 를 가진 채널이 사라지는 것이 아니라, 합이 음수인 칸이 지워진다", b: true },
      "히트맵은 (보통) 입력 이미지 크기로 키워(업샘플링) 원본 위에 겹쳐 본다",
    ], { x: 0.6, y: 5.1, w: 12.1, h: 1.7, size: 16 });
    s.addNotes("원본 검토 항목: 교재의 그림 15.2 캡션은 '음의 값은 ReLU 에 의해 제거되었다'고 쓰여 있어 α_k 가 음수인 맵이 제거되는 것처럼 읽힌다. 정확히는 합산 후 칸별 ReLU 이다. 예를 들어 2×2 예에서 2·A¹ − A² = −2 인 칸이 0이 된다.");
  }

  {
    const s = d.slide("Grad-CAM 계산 순서 5단계", { tag: "1교시 · 정리" });
    const st = [["1", "순전파", "이미지를 넣어 마지막 합성곱 층의 A^k 와 점수 y^c 를 얻는다"], ["2", "역전파", "y^c 에서 A^k 까지만 거슬러 올라가 ∂y^c/∂A^k 를 얻는다"], ["3", "평균", "채널마다 기울기를 공간 평균 → α_k^c (식 15.1)"], ["4", "가중합 + ReLU", "Σ α·A 를 만든 뒤 ReLU (식 15.2)"], ["5", "업샘플", "입력 크기로 키우고 0~1 로 정규화해 원본 위에 겹쳐 그린다"]];
    st.forEach((r, i) => {
      const y = 1.55 + i * 1.0;
      s.addText(r[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.08, w: 0.6, h: 0.6, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 18, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, r[1], { x: 1.6, y: y + 0.1, w: 2.5, h: 0.5, size: 19, bold: true });
      d.text(s, r[2], { x: 4.1, y: y + 0.12, w: 8.6, h: 0.6, size: 16, color: C.muted });
    });
    s.addNotes("참고: 원 논문에서는 마지막 합성곱 층을 쓰는 이유가 '공간 정보가 남아 있으면서 의미 수준이 높기 때문'이다. 그보다 뒤(완전연결층)는 공간 정보가 사라진다.");
  }

  // ---------------- Step 3 코드 ----------------
  {
    const s = d.slide("Step 3 (NumPy): 손으로 만든 CNN 과 순전파", { tag: "1교시 · 실습 step3_gradcam_numpy.py" });
    d.code(s, [
      "K = ...   # 3×3 필터 3개: 십자(+), 네모(■), X 탐지기",
      "BIAS = [-3.5, -5.5, -2.0]    # 약한 반응은 ReLU 가 끈다",
      "",
      "def conv_relu(x):            # 합성곱 + 편향 + ReLU",
      "    xp = np.pad(x, 1); A = np.zeros((3, 10, 10))",
      "    for k in range(3):",
      "        for i in range(10):",
      "            for j in range(10):",
      "                A[k,i,j] = (xp[i:i+3, j:j+3]*K[k]).sum() + BIAS[k]",
      "    return np.maximum(A, 0)  # 특징 맵 A (3,10,10)",
      "",
      "def head(A):                 # A 이후: 2×2 평균 풀링 → FC",
      "    P = A.reshape(3,5,2,5,2).mean(axis=(2, 4))",
      "    return (Wfc * P).sum(axis=(1, 2, 3))     # 점수 y (2,)",
    ], { x: 0.6, y: 1.45, w: 7.9, h: 5.35, size: 12, title: "step3_gradcam_numpy.py" });
    d.explain(s, ["이미지에 십자(왼쪽 위), X(오른쪽 위), 네모(오른쪽 아래)를 그렸다", "필터 3개는 각각 그 모양에서만 크게 반응하도록 손으로 정했다", "ReLU 후 결과가 특징 맵 A", "클래스 0 = '십자 있음', 클래스 1 = '네모 있음'. 클래스 0 에게 X 는 반대 증거(−1)"], { x: 8.8, y: 1.45, w: 4.0, h: 5.35, size: 13 });
    s.addNotes("가중치를 학습하지 않고 손으로 정한 이유: 정답(히트맵이 어디를 가리켜야 하는지)을 우리가 알고 있으므로 구현이 맞는지 확인할 수 있다.");
  }

  {
    const s = d.slide("Step 3 (NumPy): 기울기와 Grad-CAM", { tag: "1교시 · 실습 step3_gradcam_numpy.py" });
    d.code(s, [
      "def grad_wrt_A(c):           # ∂y^c/∂A 를 손으로 미분",
      "    # y^c = Σ W·P, P = 풀링(A): A 한 칸은 W/4 의 영향",
      "    return np.repeat(np.repeat(Wfc[c] / 4.0, 2, 1), 2, 2)",
      "",
      "def grad_cam(x, c):",
      "    A = conv_relu(x)",
      "    g = grad_wrt_A(c)",
      "    alpha = g.mean(axis=(1, 2))        # 식 15.1",
      "    s = np.tensordot(alpha, A, axes=1) # Σ α_k A^k",
      "    return np.maximum(s, 0), s, alpha  # 식 15.2: ReLU",
      "",
      "# 그래디언트 검사: 손 미분 vs 수치 미분",
      "num = (head(Ap)[c] - head(Am)[c]) / (2 * e)",
      "assert abs(num - g[k, i, j]) < 1e-6",
    ], { x: 0.6, y: 1.45, w: 7.9, h: 5.35, size: 12, title: "step3_gradcam_numpy.py" });
    d.explain(s, ["식 15.1: 각 채널의 기울기를 공간 평균 (np.mean)", "식 15.2: 가중합(tensordot) 후 np.maximum 으로 ReLU", "손으로 한 미분이 맞는지 '아주 조금 흔들어 보기'(수치 미분)로 확인 — 모든 미분 코드에 쓰는 습관", "최대 오차 9.5×10⁻¹² → 검사 통과"], { x: 8.8, y: 1.45, w: 4.0, h: 5.35, size: 13 });
    s.addNotes("수치 미분: (f(x+e) − f(x−e)) / 2e. 손 미분과 거의 같으면 맞는 것이다. 이 습관은 앞서 배운 역전파 때와 같다.");
  }

  {
    const s = d.slide("Step 3 결과: 클래스에 따라 히트맵이 다른 곳을 가리킨다", { tag: "1교시 · 실습 결과" });
    img(s, "gradcam_toy", 0.6, 1.5, 12.1);
    d.bullets(s, [
      { t: "클래스 0(십자): 최대 위치 (2,2) = 십자 중심 / 클래스 1(네모): 최대 위치 (7,7) = 네모 중심", b: true },
      "X 가 있던 오른쪽 위: ReLU 전 값이 −0.75(반대 증거) → ReLU 후 0 으로 지워진다",
      "손 미분 vs 수치 미분 최대 오차 9.5×10⁻¹² (기울기 검사 통과)",
    ], { x: 0.6, y: 4.5, w: 12.1, h: 2.3, size: 17 });
    s.addNotes("그림 설명: 4번째 패널의 붉은 칸은 찬성, 푸른 칸은 반대 증거. 5번째 패널은 ReLU 로 푸른 칸을 지운 최종 히트맵.");
  }

  // ---------------- Step 4 ----------------
  {
    const s = d.slide("Step 4 (PyTorch): hook 으로 A 와 기울기 붙잡기", { tag: "1교시 · 실습 step4_gradcam_torch.py" });
    d.code(s, [
      "def grad_cam(net, x, c=None):",
      "    store = {}",
      "    def fwd_hook(module, inp, out):   # relu3 를 지날 때 호출",
      "        store['A'] = out              # ① A 저장",
      "        out.register_hook(lambda g: store.update(g=g))  # ② 기울기",
      "    h = net.relu3.register_forward_hook(fwd_hook)",
      "    net.zero_grad(); out = net(x); h.remove()",
      "    c = int(out.argmax(1)) if c is None else c",
      "    out[0, c].backward()              # ③ y^c 에서 역전파",
      "    A, g = store['A'].detach(), store['g']",
      "    alpha = g.mean(dim=(2, 3), keepdim=True)      # 식 15.1",
      "    cam = F.relu((alpha * A).sum(1, keepdim=True))  # 식 15.2",
      "    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear')",
      "    return cam / cam.max(), c",
    ], { x: 0.6, y: 1.45, w: 8.3, h: 5.35, size: 12, title: "step4_gradcam_torch.py" });
    d.explain(s, ["hook = 순전파·역전파 중간에 끼워 두는 '감시 장치' (모델 코드를 고치지 않는다)", "register_forward_hook: 층을 지날 때 출력 A 를 저장", "out.register_hook: 그 A 로 기울기가 돌아올 때 값을 저장", "나머지는 NumPy 버전과 똑같은 식 15.1, 15.2 — 업샘플링만 추가"], { x: 9.2, y: 1.45, w: 3.6, h: 5.35, size: 13 });
    s.addNotes("모델: conv(1→8) → pool → conv(8→16) → conv(16→16)+ReLU(A) → 전역 평균 풀링 → 선형. 8×8 숫자 이미지를 16×16 으로 키워 사용(히트맵을 보기 위해).");
  }

  {
    const s = d.slide("Step 4 실험: 모델이 '도장'을 보고 답한다면?", { tag: "1교시 · 지름길 실험" });
    d.card(s, { x: 0.6, y: 1.55, w: 6.0, h: 2.7, fill: C.tealL, title: "실험 설계", tColor: C.teal, size: 15, body: "· 손글씨 숫자(16×16), 약간의 잡음 추가\n· 정상 모델: 숫자 그대로 학습\n· 지름길 모델: 학습 데이터에서 7 에만 왼쪽 위에 4×4 밝은 '도장'\n→ 도장이 '7' 과 100% 함께 다닌다 (영수증의 강조 텍스트와 같은 구조)" });
    d.card(s, { x: 6.9, y: 1.55, w: 5.8, h: 2.7, fill: C.orangeL, title: "확인할 것", tColor: C.orange, size: 15, body: "① 도장을 7 이 아닌 숫자에 찍으면 7 로 오답하는가?\n② 그때 Grad-CAM 은 도장 근처를 가리키는가?\n③ 도장 없는 깨끗한 정확도는 그대로 높아 보이는가?" });
    d.bullets(s, [
      { t: "깨끗한 테스트 정확도만 보면 지름길을 눈치채기 어렵다 (정상 0.84 / 지름길 0.80)", b: true },
      "이것이 사례 속 '테스트에서는 높은 정확도'와 같은 상황이다",
    ], { x: 0.6, y: 4.6, w: 12.1, h: 1.8, size: 18 });
  }

  {
    const s = d.slide("Step 4 결과: 지름길 모델의 히트맵은 도장 쪽으로 쏠린다", { tag: "1교시 · 실습 결과" });
    img(s, "gradcam_digits", 0.6, 1.45, 6.3);
    table(s, ["측정", "정상", "지름길"], [
      ["깨끗한 테스트 정확도", "0.84", "0.80"],
      ["도장 찍은 비-7 을 7 로 오답", "5.7%", "47.7%"],
      ["히트맵 왼쪽 위 1/4 몫(면적 25%)", "0.33", "0.55"],
    ], { x: 7.2, y: 1.55, w: 5.5, colW: [3.1, 1.2, 1.2], fontSize: 13, rowH: 0.55 });
    d.bullets(s, [
      "맨 위: 도장이 찍힌 입력, 중간: 정상 모델, 아래: 지름길 모델 (붉을수록 '7' 근거)",
      { t: "지름길 모델은 모양이 아니라 도장 근처를 근거로 7 이라 답한다", b: true },
      "히트맵은 해상도가 낮다(8×8 을 키운 것): 도장 '바로 아래'를 가리켜도 도장 때문이라는 뜻",
    ], { x: 7.2, y: 3.9, w: 5.5, h: 2.9, size: 14 });
    s.addNotes("정상 모델 행의 히트맵이 위쪽에도 약하게 붙는 것은 도장이 매우 밝은 이상한 입력이기 때문. 숫자는 한 번의 학습(시드 0) 기준이며 100장 샘플의 평균이다. 시드를 바꾸면 조금 달라진다.");
  }

  {
    const s = d.slide("Grad-CAM 을 읽을 때 주의할 점 + 더 나아가기", { tag: "1교시 · 한계" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 2.5, fill: C.redL, title: "한계", tColor: C.red, size: 15, body: "· 공간 해상도가 낮다 (마지막 층 크기를 키운 것)\n· 깊은 층에서는 노이즈 많은 히트맵이 나올 수 있다\n· 'CNN 이 어디를 봤나'이지 '왜 그렇게 판단했나'는 아니다" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 2.5, fill: C.purpleL, title: "보너스: 발전된 방법들 (교재 범위 밖)", tColor: C.purple, size: 15, body: "· Grad-CAM++: 여러 곳에 같은 물체가 있을 때 더 정확\n· Integrated Gradients: 검은 기준 이미지에서 입력까지 걸으며 기울기를 평균 — 입력 칸별 기여도(합이 f(x)−f(기준)이 됨)" });
    d.code(s, ["ig = (x - base) * path.grad.mean(0)   # 입력 × 경로 평균 기울기", "# 완전성 검사 (step4): Σ속성 = 99.613  vs  f(x)−f(기준) = 99.796"], { x: 0.6, y: 4.4, w: 12.1, h: 1.0, size: 13 });
    d.text(s, "Grad-CAM 은 '가설을 만드는' 도구: 히트맵이 이상하면 데이터를 점검하고, 도장을 지우거나 바꿔 보는 개입 실험으로 확인한다.", { x: 0.6, y: 5.7, w: 12.1, h: 0.9, size: 16, color: C.muted });
  }

  {
    const s = d.slide("1교시 정리와 확인 질문", { tag: "1교시 · 정리" });
    d.bullets(s, [
      { t: "설명 가능성: 모델이 낸 출력의 '이유'를 사후에 들여다보는 방법들 (신뢰·디버깅·규제·공정성)", b: true },
      "Grad-CAM: α_k^c = 기울기의 공간 평균 → 가중합 → ReLU → 히트맵",
      "특징 맵 A 는 가중치가 아니라 활성값, ReLU 는 가중합 결과에 적용",
      "지름길(도장)에 기대는 모델은 깨끗한 정확도만으로는 알아채기 어렵고, 히트맵과 개입 실험으로 드러난다",
    ], { x: 0.6, y: 1.55, w: 12.1, h: 3.1, size: 18 });
    d.card(s, { x: 0.6, y: 4.85, w: 12.1, h: 1.9, fill: C.tealL, title: "확인 질문", tColor: C.teal, size: 15, body: "Q1. 식 15.1 에서 α 가 음수인 채널은 히트맵에 어떻게 영향을 주나?\nQ2. 왜 마지막 합성곱 층의 특징 맵을 쓰는가?\nQ3. 히트맵이 도장을 가리킨다면 다음에 무엇을 해야 하나?" });
    s.addNotes("정답 힌트. Q1: 그 채널이 켜진 칸의 합을 낮춘다(반대 증거), 합이 음수가 되면 ReLU 가 지운다. Q2: 공간 정보가 남아 있으면서 의미가 높은 층이기 때문. Q3: 데이터 점검(도장 제거·섞기)과 재학습.");
  }
};
