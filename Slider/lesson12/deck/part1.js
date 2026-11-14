// 1교시: 활성값과 비선형성, ReLU
const path = require("path");
const { C, FONT, IMG } = require("./lib");
const { img, F } = require("./helpers");

module.exports = function part1(d, pres) {
  // ---------------- 표지 ----------------
  {
    const s = pres.addSlide(); d.n += 1; s.background = { color: C.paper };
    s.addText("Lesson 12", { x: 0.9, y: 1.3, w: 11, h: 0.7, fontFace: FONT, fontSize: 28, bold: true, color: C.teal, margin: 0 });
    s.addText("Normalization and Initialization", { x: 0.9, y: 2.0, w: 11.5, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    s.addText("정규화와 초기화: 깊은 신경망이 안정적으로 배우는 비결", { x: 0.9, y: 3.1, w: 11.5, h: 0.6, fontFace: FONT, fontSize: 24, color: C.teal, bold: true, margin: 0 });
    s.addText("활성값 · ReLU · 가중치 초기화(Xavier, He) · Batch Norm · Layer Norm", { x: 0.9, y: 3.8, w: 11.5, h: 0.5, fontFace: FONT, fontSize: 18, color: C.muted, margin: 0 });
    ["평균 0", "분산 1", "안정!"].forEach((t, i) => d.box(s, t, { x: 0.9 + i * 1.9, y: 4.8, w: 1.7, h: 0.7, fill: [C.orangeL, C.tealL, C.greenL][i], size: 20, bold: true }));
    s.addText("튜링랩 프로그래밍학원\nTuringLab. Programming Academy", { x: 0.9, y: 5.9, w: 6, h: 0.7, fontFace: FONT, fontSize: 14, color: C.ink, margin: 0 });
    s.addImage({ path: path.join(IMG, "logo.png"), x: 9.5, y: 6.0, w: 3.0, h: 3.0 * 118 / 756 });
    s.addNotes("강의 소개. 오늘 3시간(50분 x 3교시)의 흐름: 1교시 활성값/ReLU, 2교시 가중치 초기화, 3교시 정규화와 종합 실험. 모든 개념은 NumPy 코드로 직접 확인한다.");
  }

  // ---------------- 로드맵 ----------------
  {
    const s = d.slide("오늘의 로드맵: 3교시 (50분 수업 + 10분 휴식) × 3", { tag: "OVERVIEW" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const cell = (t, o = {}) => ({ text: t, options: { align: "left", ...o } });
    s.addTable([
      [H("교시"), H("주제"), H("직접 해보는 것")],
      [cell("1교시", { bold: true }), cell("활성값 · 비선형성 · XOR · ReLU와 그 친구들"), cell("Step 1  선형 붕괴 확인, XOR 풀기")],
      [cell("2교시", { bold: true }), cell("가중치 초기화: 분산, Xavier, He, 신호 폭발·소실"), cell("Step 2  10층 신호 크기 실험")],
      [cell("3교시", { bold: true }), cell("미니배치 · Batch Norm · Layer Norm · 종합 실험"), cell("Step 3~4  정규화 직접 구현, 학습 비교")],
      [cell("마무리", { bold: true }), cell("실습 문제 5개 + 솔루션, 원본 자료 검토 결과"), cell("손계산 · 구현 · 예측 후 실험")],
    ], { x: 0.6, y: 1.6, w: 12.1, colW: [1.5, 6.4, 4.2], fontFace: FONT, fontSize: 16, color: C.ink, rowH: 0.7, valign: "middle", border: { type: "solid", color: C.line, pt: 1 }, fill: { color: C.white } });
    d.text(s, "준비물: Python 3.9+, NumPy, (Step 3·4) PyTorch, matplotlib   |   코드는 lesson12/code 폴더", { x: 0.6, y: 5.7, w: 12.1, h: 0.4, size: 15, color: C.muted });
  }

  // ---------------- 학습 목표 ----------------
  {
    const s = d.slide("이 강의가 끝나면 할 수 있는 일", { tag: "GOAL" });
    const items = [
      ["1", "설명하기", "활성값, 비선형성, 내부 공변량 변화, 정규화가 왜 필요한지 내 말로 설명한다"],
      ["2", "계산하기", "Xavier/He 초기화의 분산을 계산하고, 배치 정규화를 손으로 계산한다"],
      ["3", "구현하기", "BatchNorm / LayerNorm을 NumPy로 직접 만들고 PyTorch와 결과를 비교한다"],
      ["4", "고르기", "상황(활성화 함수, 배치 크기, 모델 종류)에 맞는 초기화·정규화를 선택한다"],
    ];
    items.forEach((it, i) => {
      const y = 1.6 + i * 1.2;
      s.addText(it[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 22, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, it[1], { x: 1.7, y, w: 3.2, h: 0.5, size: 20, bold: true });
      d.text(s, it[2], { x: 1.7, y: y + 0.47, w: 10.8, h: 0.5, size: 16, color: C.muted });
    });
  }

  // ---------------- 1교시 ----------------
  d.section("1교시", "활성값과 비선형성", "신경망은 왜 '구부러지는' 함수가 필요한가 · ReLU 다시 보기", "1교시 시작. 정규화와 초기화를 배우기 전에, 정규화되는 대상인 '활성값'이 무엇인지부터 확실히 한다.");

  {
    const s = d.slide("신경망 한 층이 하는 일", { tag: "1교시 · 활성값" });
    d.box(s, "입력 x", { x: 0.7, y: 1.9, w: 1.8, h: 0.8, fill: C.orangeL, size: 18, bold: true });
    d.box(s, "선형 변환\nz = Wx + b", { x: 3.3, y: 1.9, w: 2.6, h: 0.8, fill: C.tealL, size: 16, bold: true });
    d.box(s, "활성화 함수\na = φ(z)", { x: 6.7, y: 1.9, w: 2.6, h: 0.8, fill: C.purpleL, size: 16, bold: true });
    d.box(s, "활성값 a\n→ 다음 층 입력", { x: 10.1, y: 1.9, w: 2.5, h: 0.8, fill: C.greenL, size: 16, bold: true });
    d.arrow(s, 2.5, 2.3, 3.3, 2.3); d.arrow(s, 5.9, 2.3, 6.7, 2.3); d.arrow(s, 9.3, 2.3, 10.1, 2.3);
    d.bullets(s, [
      "W, b : 가중치와 편향 (학습으로 바뀌는 숫자들)",
      "φ : ReLU, Sigmoid, Tanh 같은 활성화 함수",
      "활성값(activation) a : 뉴런이 '활성화 함수를 통과한 뒤' 내놓는 값",
      "활성값은 단순한 숫자가 아니라 다음 층으로 전달되는 신호(signal)",
    ], { x: 0.7, y: 3.2, w: 11.8, h: 2.5, size: 18 });
    d.card(s, { x: 0.7, y: 5.8, w: 11.9, h: 0.95, fill: C.orangeL, title: "", body: "이번 강의의 핵심 질문: 층이 깊어질수록 이 활성값의 크기와 분포가 흐트러지는데, 어떻게 안정적으로 유지할까?", size: 16 });
    s.addNotes("z = Wx + b, a = φ(z). 활성값이 '정규화의 대상'이라는 점을 강조. 마지막 박스가 오늘의 핵심 질문.");
  }

  {
    const s = d.slide("선형(linear)과 비선형(non-linear)", { tag: "1교시 · 활성값" });
    d.card(s, { x: 0.7, y: 1.5, w: 5.9, h: 2.4, fill: C.tealL, title: "선형 함수의 조건", tColor: C.teal, size: 15, body: "f(ax + by) = a·f(x) + b·f(y)\n\n'더하기와 곱하기'만으로 이루어진 함수.\n공간을 회전·이동·늘리기만 한다." });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 2.4, fill: C.orangeL, title: "비선형 함수 = 위 성질을 만족하지 않는 함수", tColor: C.orange, size: 15, body: "ReLU: f(x) = max(0, x)\nSigmoid, Tanh\n\n공간을 접고, 자르고, 구부린다." });
    d.text(s, "선형 층만 쌓으면 어떻게 될까?", { x: 0.7, y: 4.2, w: 11.9, h: 0.4, size: 20, bold: true, color: C.teal });
    F(d, s, "y = W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx", { x: 0.7, y: 4.75, w: 7.2, h: 0.75, size: 24 });
    d.text(s, "→ 아무리 깊어도 결국 하나의 선형 변환 (얇은 층 1개와 같다)", { x: 8.1, y: 4.8, w: 4.6, h: 0.7, size: 16, bold: true, color: C.red });
    d.text(s, "활성화 함수가 없다면:  깊은 신경망 = 얕은 선형 모델", { x: 0.7, y: 5.9, w: 11.9, h: 0.5, size: 20, bold: true });
    s.addNotes("선형 함수의 정의를 말로 풀어 설명. 행렬곱은 결합법칙이 성립하므로 W3 W2 W1을 미리 곱해 하나의 W로 만들 수 있다는 점을 칠판에 직접 써 보인다.");
  }

  {
    const s = d.slide("직접 확인: 선형 2층 = 선형 1층", { tag: "1교시 · 코드 Step 1-①" });
    d.code(s, [
      "W1 = rng.normal(size=(3, 4))   # 1층 가중치 (3 -> 4)",
      "W2 = rng.normal(size=(4, 2))   # 2층 가중치 (4 -> 2)",
      "x  = rng.normal(size=(5, 3))   # 샘플 5개",
      "",
      "deep = (x @ W1) @ W2           # 두 층을 차례로 통과",
      "one  = x @ (W1 @ W2)           # 미리 곱해 둔 1층",
      "np.allclose(deep, one)         # True !",
      "",
      "relu = lambda z: np.maximum(0, z)",
      "deep_relu = relu(x @ W1) @ W2  # 중간에 ReLU를 끼움",
      "np.allclose(deep_relu, one)    # False",
    ], { x: 0.6, y: 1.5, w: 7.4, h: 4.4, size: 13, title: "step1_nonlinear.py" });
    d.explain(s, ["x @ W : 행렬곱. 한 층의 선형 변환", "(x@W1)@W2 와 x@(W1@W2)는 같다 (결합법칙)", "ReLU를 끼우면 하나로 합칠 수 없다 → 표현력이 늘어난다", "결과: True / False 로 직접 확인"], { x: 8.2, y: 1.5, w: 4.5, h: 4.4 });
    d.text(s, "실행: python step1_nonlinear.py", { x: 0.6, y: 6.1, w: 8, h: 0.4, size: 15, color: C.muted });
    s.addNotes("학생들이 직접 실행하게 한다. np.allclose는 '거의 같은가'를 확인하는 함수. True/False 결과가 핵심.");
  }

  {
    const s = d.slide("대표적인 활성화 함수들", { tag: "1교시 · 활성값" });
    img(s, "activations", 0.7, 1.5, 11.9);
    d.bullets(s, [
      "Sigmoid(0~1), Tanh(-1~1): 부드러운 S자. 양 끝에서 평평해져 기울기가 0에 가까움",
      "ReLU: 0 이하는 0, 양수는 그대로. 가장 널리 쓰임",
      "Leaky ReLU, GELU: ReLU의 약점을 보완한 후손들 (뒤에서 다룸)",
    ], { x: 0.7, y: 4.8, w: 11.9, h: 1.8, size: 17 });
    s.addNotes("모양만 눈에 익히면 된다. S자 함수는 양끝이 평평하다는 점(=기울기 거의 0)을 기억해 두라고 말한다. 2교시 초기화 실험에서 다시 나온다.");
  }

  {
    const s = d.slide("왜 비선형성이 꼭 필요한가? XOR 문제", { tag: "1교시 · 비선형성" });
    img(s, "xor", 0.6, 1.5, 7.6);
    d.card(s, { x: 8.5, y: 1.5, w: 4.2, h: 3.4, fill: C.paper, title: "XOR (배타적 논리합)", size: 14, body: "서로 다르면 1, 같으면 0\n(0,0)→0   (0,1)→1\n(1,0)→1   (1,1)→0\n\n직선 하나(선형 모델)로는 절대 두 그룹을 가를 수 없다." });
    d.card(s, { x: 0.6, y: 5.2, w: 12.1, h: 1.5, fill: C.orangeL, title: "비선형 활성화가 들어가면", tColor: C.orange, size: 15, body: "중간 표현 공간을 비틀고 → 입력을 다른 공간으로 옮겨서 → 선형으로 가를 수 있게 된다.\n현실의 문제(이미지·음성·언어)는 대부분 선형으로 표현할 수 없다." });
    s.addNotes("왼쪽: 네 점은 대각선에 같은 색이라 직선으로 못 가른다. 오른쪽: 은닉층+ReLU로 옮기면 직선이 생긴다(다음 장 코드로 확인). 단층 퍼셉트론이 XOR을 못 푼다는 역사적 이야기를 곁들여도 좋다.");
  }

  {
    const s = d.slide("비선형성 = 공간을 '구부리는' 것", { tag: "1교시 · 비선형성" });
    d.card(s, { x: 0.7, y: 1.6, w: 5.8, h: 2.6, fill: C.tealL, title: "선형 변환", tColor: C.teal, size: 17, body: "공간을 회전, 이동, 스케일링만 한다.\n\n종이를 돌리고 옮기고 늘리는 것과 같다.\n구겨지지 않는다." });
    d.card(s, { x: 6.8, y: 1.6, w: 5.9, h: 2.6, fill: C.orangeL, title: "비선형 변환", tColor: C.orange, size: 17, body: "공간을 접고, 자르고, 구부린다.\n\n종이를 접어서 서로 떨어진 점들을 한곳에 포개거나,\n겹쳐 있던 점들을 갈라놓을 수 있다." });
    d.bullets(s, ["ReLU는 '음수 부분을 0으로 접는' 가장 단순한 비선형 변환", "접힌 공간을 여러 번 겹치면 아주 복잡한 모양도 표현 가능", "활성화 함수 = 비선형성을 '의도적으로 주입'하는 장치"], { x: 0.7, y: 4.5, w: 11.9, h: 2.1, size: 18 });
    s.addNotes("종이접기 비유. 종이를 들고 직접 접어 보이면 효과적.");
  }

  {
    const s = d.slide("직접 확인: 은닉층 + ReLU로 XOR 풀기", { tag: "1교시 · 코드 Step 1-②" });
    d.code(s, [
      "X = [[0,0],[0,1],[1,0],[1,1]]   # XOR 입력 4가지",
      "Wh = [[1,1],[1,1]]; bh = [0,-1] # 은닉 뉴런 2개",
      "Wo = [1, -2]                    # 출력층",
      "",
      "h   = relu(X @ Wh + bh)  # 은닉층: (x1+x2), (x1+x2-1)",
      "out = h @ Wo             # 출력: h1 - 2*h2",
      "# out = [0, 1, 1, 0]  <- 정답과 일치!",
      "",
      "# 선형 모델(최소제곱 직선)의 최선",
      "# -> [0.5, 0.5, 0.5, 0.5]  XOR 실패",
    ], { x: 0.6, y: 1.5, w: 7.2, h: 4.4, size: 13, title: "step1_nonlinear.py" });
    d.explain(s, ["은닉층: 두 입력을 합친 값 x1+x2 와, 거기서 1을 뺀 값을 ReLU에 통과", "(0,1)과 (1,0)이 같은 점 (1,0)으로 포개진다 ← 공간이 접혔다!", "출력층의 직선 하나로 이제 분리 가능", "선형 모델은 모든 입력에 0.5 → 전혀 구분 못 함"], { x: 8.0, y: 1.5, w: 4.7, h: 4.4 });
    d.text(s, "가중치는 학습이 아니라 손으로 구함: '구조'만으로 풀 수 있음을 보여 주는 예", { x: 0.6, y: 6.1, w: 12, h: 0.4, size: 15, color: C.muted });
    s.addNotes("표로 계산: (0,0)->h=(0,0)->0, (0,1)->h=(1,0)->1, (1,0)->h=(1,0)->1, (1,1)->h=(2,1)->2-2=0. 칠판에서 한 줄씩 같이 계산한다.");
  }

  {
    const s = d.slide("ReLU 다시 보기", { tag: "1교시 · ReLU" });
    F(d, s, "ReLU(x) = max(0, x)", { x: 0.7, y: 1.5, w: 5.4, h: 0.8, size: 26 });
    d.bullets(s, ["입력이 양수면 그대로 출력", "입력이 0 이하면 0 출력", "아주 단순하지만, 이 단순함이 학습을 크게 안정시킨다"], { x: 0.7, y: 2.6, w: 5.6, h: 2.0, size: 18 });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    s.addTable([[H("입력 x"), H("ReLU(x)"), H("기울기")], ["-3", "0", "0"], ["-0.5", "0", "0"], ["0.5", "0.5", "1"], ["3", "3", "1"]].map((r, i) => i ? r.map(t => ({ text: t, options: { align: "center" } })) : r), { x: 6.8, y: 1.6, w: 5.9, colW: [1.9, 2.0, 2.0], fontFace: FONT, fontSize: 16, color: C.ink, rowH: 0.55, border: { type: "solid", color: C.line, pt: 1 } });
    d.card(s, { x: 0.7, y: 4.9, w: 12.0, h: 1.8, fill: C.paper, title: "입력 → 출력 예시", size: 16, body: "x = [-2, -1, 0, 1, 2]   →   ReLU(x) = [0, 0, 0, 1, 2]\n음수는 모두 0으로 '꺼지고', 양수만 살아남는다." });
    s.addNotes("ReLU의 정의(Rectified Linear Unit = 교정된 선형 단위). 기울기 표는 다음 장 '왜 ReLU가 중요한가'의 근거.");
  }

  {
    const s = d.slide("왜 ReLU가 중요한가? (장점 3가지)", { tag: "1교시 · ReLU" });
    d.card(s, { x: 0.6, y: 1.5, w: 3.9, h: 4.6, fill: C.tealL, title: "① 기울기 소실 완화", tColor: C.teal, size: 15, body: "sigmoid/tanh는 입력이 크면 기울기가 거의 0이 되어 깊은 층에서 학습이 안 된다.\n\nReLU는 양수 영역에서 기울기가 1이라 깊은 층까지 기울기가 잘 전달된다." });
    d.card(s, { x: 4.75, y: 1.5, w: 3.9, h: 4.6, fill: C.greenL, title: "② 계산이 매우 빠름", tColor: C.green, size: 15, body: "max 연산 하나면 끝.\n\n미분도 간단(0 또는 1)해서 GPU에서 특히 효율적이다." });
    d.card(s, { x: 8.9, y: 1.5, w: 3.8, h: 4.6, fill: C.purpleL, title: "③ 희소성(Sparsity) 유도", tColor: C.purple, size: 15, body: "입력의 절반 정도가 0이 되어 뉴런이 선택적으로 활성화된다.\n\n일종의 정규화 효과를 주어 일반화에 도움이 된다." });
    d.text(s, "※ '희소성(sparsity)': 대부분의 값이 0인 성질 (원본의 sparity는 오타)", { x: 0.6, y: 6.3, w: 12, h: 0.4, size: 14, color: C.muted });
    s.addNotes("기울기 소실은 2교시/3교시에서 다시 만난다. 희소성은 '절반이 0'이라는 점이 He 초기화의 분산이 2배인 이유와 연결된다(2교시 예고).");
  }

  {
    const s = d.slide("ReLU의 단점과 그 후손들", { tag: "1교시 · ReLU" });
    d.card(s, { x: 0.6, y: 1.5, w: 6.0, h: 2.5, fill: C.redL, title: "① Dying ReLU (죽은 ReLU)", tColor: C.red, size: 15, body: "입력이 계속 음수이면 출력이 영원히 0이 되고 기울기도 0이 되어, 그 뉴런은 더 이상 학습되지 않는다." });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 2.5, fill: C.orangeL, title: "② 출력 분포가 한쪽으로 치우침", tColor: C.orange, size: 15, body: "음수를 모두 0으로 만들어 출력 평균이 양수 쪽으로 쏠릴 수 있다.\nBatchNorm 같은 정규화와 함께 쓰면 훨씬 안정적이다." });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t, b) => ({ text: t, options: { align: "left", bold: !!b } });
    s.addTable([[H("해결책(변형)"), H("아이디어")], [c("Leaky ReLU", 1), c("음수에서도 작은 기울기 허용: f(x) = 0.01x")], [c("Parametric ReLU", 1), c("그 음수 기울기를 학습으로 정함")], [c("ELU", 1), c("음수 영역을 부드러운 곡선으로")], [c("GELU", 1), c("부드러운 ReLU. Transformer(BERT, GPT)에서 표준")]], { x: 0.6, y: 4.2, w: 12.1, colW: [3.2, 8.9], fontFace: FONT, fontSize: 15, color: C.ink, rowH: 0.46, border: { type: "solid", color: C.line, pt: 1 } });
    s.addNotes("Dying ReLU: 학습률이 너무 크거나 초기 편향이 나쁘면 뉴런이 영영 0이 되는 현상. 원본의 'Parameteric'은 'Parametric'의 오타. GELU는 최신 모델의 표준이라 최신 자료로 보강.");
  }

  {
    const s = d.slide("1교시 정리 + 생각해 보기", { tag: "1교시 · 정리" });
    d.bullets(s, [
      { t: "활성값 = 활성화 함수를 통과한 뒤 다음 층으로 전달되는 신호", b: true },
      "활성화 함수가 없으면 아무리 깊어도 선형 모델 1개와 같다 (Step 1 확인)",
      "비선형성은 공간을 접고 구부려 XOR 같은 문제를 가능하게 한다",
      "ReLU: 빠르고 기울기가 잘 흐르지만 Dying ReLU와 치우친 출력이 단점",
    ], { x: 0.7, y: 1.5, w: 12, h: 2.8, size: 19 });
    d.card(s, { x: 0.7, y: 4.5, w: 12.0, h: 2.2, fill: C.orangeL, title: "생각해 보기 (10분 휴식 전)", tColor: C.orange, size: 16, body: "Q. 층을 20개 쌓았더니, 앞쪽 층의 가중치가 조금만 바뀌어도 뒤쪽 층이 받는 입력의 '크기와 분포'가 확 달라진다면 어떤 일이 벌어질까?\n→ 다음 시간(2~3교시)의 주제로 이어집니다." });
    s.addNotes("정답을 주지 않고 호기심을 남긴 채 휴식. 답: 뒤쪽 층이 계속 움직이는 목표에 적응해야 해서 학습이 불안정/느려진다 -> 내부 공변량 변화.");
  }
};
