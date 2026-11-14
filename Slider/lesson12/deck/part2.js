// 2교시: 가중치 초기화
const { C, FONT } = require("./lib");
const { img, F } = require("./helpers");

module.exports = function part2(d, pres) {
  d.section("2교시", "가중치 초기화 (Weight Initialization)", "Xavier · He · 신호가 폭발하거나 사라지지 않게", "2교시 시작. 학습이 시작되기 전, 가중치의 '출발점'을 어떻게 정하는지가 오늘의 주제.");

  {
    const s = d.slide("왜 초기화가 중요한가?", { tag: "2교시 · 초기화" });
    d.text(s, "학습이 시작되기 전, 가중치는 보통 무작위 값으로 정한다. 그런데 '아무 무작위'면 안 된다.", { x: 0.7, y: 1.5, w: 12, h: 0.7, size: 19 });
    d.card(s, { x: 0.7, y: 2.4, w: 3.9, h: 2.8, fill: C.redL, title: "너무 크게 → 폭발", tColor: C.red, size: 15, body: "층을 지날수록 값이 기하급수로 커진다.\n활성값·기울기 폭발" });
    d.card(s, { x: 4.8, y: 2.4, w: 3.9, h: 2.8, fill: C.tealL, title: "너무 작게 → 소실", tColor: C.teal, size: 15, body: "층을 지날수록 값이 0에 가까워진다.\n활성값·기울기 소실 → 학습이 거의 안 됨" });
    d.card(s, { x: 8.9, y: 2.4, w: 3.8, h: 2.8, fill: C.greenL, title: "적절하게 → 안정", tColor: C.green, size: 15, body: "층마다 값의 크기(분산)가 비슷하게 유지되어 학습이 잘 된다." });
    d.card(s, { x: 0.7, y: 5.4, w: 12.0, h: 1.3, fill: C.orangeL, title: "", size: 16, body: "초기화의 목표: 모든 층에서 활성값과 기울기의 분산이 적절하게 통제되도록 하는 것." });
    s.addNotes("자전거 비유: 출발 방향과 속도를 잘못 잡으면 가다가 쓰러진다. 잠시 뒤 실험으로 직접 확인한다.");
  }

  {
    const s = d.slide("준비 ①: 입력 유닛 n_in, 출력 유닛 n_out", { tag: "2교시 · 초기화" });
    d.card(s, { x: 0.7, y: 1.5, w: 5.9, h: 3.3, fill: C.tealL, title: "입력 유닛 (Input Units, n_in)", tColor: C.teal, size: 15, body: "한 층이 '받아들이는' 값의 개수\n= 해당 층의 입력 벡터 차원 수\n\n예) 입력 벡터가 [x₁, x₂, x₃] → n_in = 3\n예) 이전 층의 뉴런이 128개 → 다음 층의 n_in = 128" });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 3.3, fill: C.greenL, title: "출력 유닛 (Output Units, n_out)", tColor: C.green, size: 15, body: "한 층이 '내보내는' 값의 개수\n= 해당 층의 출력 벡터 차원 수\n\n예) Dense layer(완전연결층)가 64개 뉴런 → n_out = 64\n예) 10개 클래스 분류의 마지막 층 → n_out = 10" });
    d.box(s, "입력 n_in = 4", { x: 1.0, y: 5.3, w: 3.0, h: 0.8, fill: C.orangeL, size: 17, bold: true });
    d.box(s, "Dense 층\nW : n_in × n_out", { x: 5.0, y: 5.3, w: 3.3, h: 0.8, fill: C.gray, size: 16, bold: true });
    d.box(s, "출력 n_out = 3", { x: 9.3, y: 5.3, w: 3.0, h: 0.8, fill: C.orangeL, size: 17, bold: true });
    d.arrow(s, 4.0, 5.7, 5.0, 5.7); d.arrow(s, 8.3, 5.7, 9.3, 5.7);
    s.addNotes("Dense layer = 완전연결층(fully connected layer) = 전결합층. 가중치 행렬 W의 모양이 n_in x n_out. 이 두 숫자가 초기화의 분산을 정한다.");
  }

  {
    const s = d.slide("준비 ②: 정규분포 N(0, σ²)와 분산", { tag: "2교시 · 초기화" });
    img(s, "gauss", 0.6, 1.5, 6.6);
    d.card(s, { x: 7.5, y: 1.5, w: 5.2, h: 2.5, fill: C.purpleL, title: "N(0, σ²) 읽는 법", tColor: C.purple, size: 15, body: "평균이 0이고 분산이 σ²인 정규분포.\n분산(σ²): 값이 평균에서 얼마나 퍼져 있나\n표준편차(σ) = √분산" });
    d.card(s, { x: 7.5, y: 4.2, w: 5.2, h: 2.4, fill: C.paper, title: "예: N(0, 4)", size: 15, body: "평균 = 0, 분산 = 4\n표준편차 = √4 = 2\n→ 값 대부분이 0을 중심으로 ±2 범위에 분포" });
    d.text(s, "Wᵢⱼ ~ N(0, σ²) : 'Wᵢⱼ는 이 분포에서 뽑는다'는 뜻 (~ 는 같다(=)가 아니라 '확률적으로 뽑힌다')", { x: 0.6, y: 5.3, w: 6.6, h: 1.2, size: 15, color: C.ink });
    s.addNotes("분산을 쓰는 이유: 수학적으로 다루기 편해서. 표준편차는 단위가 원래 값과 같아 직관적. 그림에서 분산이 클수록 넓게 퍼진다. W_ij = i번째 출력 뉴런과 j번째 입력 뉴런을 잇는 가중치 하나.");
  }

  {
    const s = d.slide("왜 분산이 n_in에 맞춰져야 할까? (직관 유도)", { tag: "2교시 · 초기화" });
    d.text(s, "출력 뉴런 하나는 n_in개의 입력을 곱해서 더한다:", { x: 0.7, y: 1.5, w: 12, h: 0.45, size: 18 });
    F(d, s, "y = w₁x₁ + w₂x₂ + … + w_n x_n      (n = n_in)", { x: 0.7, y: 2.05, w: 8.4, h: 0.75, size: 22 });
    d.text(s, "서로 독립이고 평균이 0이면, 분산은 항목별로 더해진다:", { x: 0.7, y: 3.05, w: 12, h: 0.45, size: 18 });
    F(d, s, "Var(y) = n_in · Var(w) · Var(x)", { x: 0.7, y: 3.6, w: 7.0, h: 0.75, size: 24, fill: C.tealL });
    d.bullets(s, [
      "층을 지나도 분산을 유지하려면 Var(y) = Var(x)",
      "→ n_in · Var(w) = 1  →  Var(w) = 1 / n_in",
      "입력이 많을수록(n_in↑) 각 가중치는 더 작아야 합이 폭주하지 않는다",
    ], { x: 0.7, y: 4.6, w: 12, h: 2.1, size: 19 });
    s.addNotes("가장 중요한 한 장. 숫자 100개를 더하면 합이 100배 가까이 커질 수 있다는 감각. 분산의 곱 규칙은 엄밀 증명 대신 '독립이면 분산이 더해진다'만 말해도 충분.");
  }

  {
    const s = d.slide("Xavier 초기화 (자비어 / Glorot 초기화)", { tag: "2교시 · 초기화" });
    F(d, s, "W_ij ~ N( 0, 2 / (n_in + n_out) )", { x: 0.7, y: 1.5, w: 8.2, h: 0.85, size: 26, fill: C.tealL });
    d.text(s, "(12.9)", { x: 9.1, y: 1.7, w: 1.5, h: 0.4, size: 14, color: C.muted });
    d.bullets(s, [
      "tanh처럼 0을 중심으로 대칭인 활성화 함수를 위해 설계",
      "순전파(앞으로 가는 신호)와 역전파(뒤로 가는 기울기)가 모두 증폭·약화되지 않도록 균형을 맞춘다",
      "순전파 조건은 1/n_in, 역전파 조건은 1/n_out → 둘을 절충(조화 평균)한 것이 2/(n_in + n_out)",
      "n_in = n_out 이면 정확히 1/n_in 과 같다",
    ], { x: 0.7, y: 2.6, w: 12, h: 2.6, size: 18 });
    d.card(s, { x: 0.7, y: 5.3, w: 12.0, h: 1.4, fill: C.paper, title: "자주 보이는 변형", size: 15, body: "정규분포형: Var(W) = 2/(n_in+n_out)   |   균등분포형: W ~ U(−a, a), a = √(6/(n_in+n_out))   |   Var(W) = 1/n_in 은 LeCun 초기화" });
    s.addNotes("원본은 Var(W)=1/n_in 또는 2/(n_in+n_out) 두 식을 함께 적었다. 둘은 목표(순전파만/순역전파 절충)가 다른 서로 다른 이름의 방식이라는 점을 짚어 준다. 균등분포 a = sqrt(6/(nin+nout))의 분산이 2/(nin+nout)이 되는 것은 U(-a,a)의 분산 a^2/3 때문.");
  }

  {
    const s = d.slide("He 초기화 (카이밍 히 / Kaiming 초기화)", { tag: "2교시 · 초기화" });
    F(d, s, "W_ij ~ N( 0, 2 / n_in )", { x: 0.7, y: 1.5, w: 7.0, h: 0.85, size: 26, fill: C.orangeL });
    d.text(s, "(12.10)", { x: 7.9, y: 1.7, w: 1.5, h: 0.4, size: 14, color: C.muted });
    d.bullets(s, [
      "ReLU처럼 음수를 0으로 만드는 비대칭 활성화 함수를 위해 설계",
      "ReLU는 출력의 절반을 0으로 만든다 → 신호의 분산이 절반으로 줄어든다",
      "그래서 Xavier(1/n_in)보다 분산을 2배(2/n_in)로 키워 보상한다",
      "ReLU가 희소성을 만들어도 순전파·역전파의 분산이 유지된다",
    ], { x: 0.7, y: 2.6, w: 12, h: 2.5, size: 18 });
    d.card(s, { x: 0.7, y: 5.2, w: 5.9, h: 1.5, fill: C.tealL, title: "Xavier : tanh 계열", tColor: C.teal, size: 15, body: "Var(W) = 2/(n_in + n_out)" });
    d.card(s, { x: 6.8, y: 5.2, w: 5.9, h: 1.5, fill: C.orangeL, title: "He : ReLU 계열", tColor: C.orange, size: 15, body: "Var(W) = 2/n_in   (Xavier보다 큼)" });
    s.addNotes("직관: 입력이 0을 중심으로 대칭이면 ReLU 후 제곱 평균이 절반이 된다. 그래서 가중치 분산을 2배로. 수식 증명은 생략하고 '절반이 0이 되니 2배로 보상'만 전달.");
  }

  {
    const s = d.slide("실험 준비: 10층 신경망에 신호를 흘려 보내자", { tag: "2교시 · 코드 Step 2" });
    d.code(s, [
      "def forward_stds(act, std_fn, layers=10, width=512, n=1000):",
      "    a = rng.normal(size=(n, width))     # 입력: 평균 0, 표준편차 1",
      "    stds = [a.std()]",
      "    for _ in range(layers):",
      "        std = std_fn(width, width)       # 초기화 규칙으로 표준편차 결정",
      "        W = rng.normal(0, std, size=(width, width))   # W ~ N(0, std²)",
      "        a = act(a @ W)                   # 선형 변환 + 활성화 함수",
      "        stds.append(a.std())             # 이 층 활성값의 표준편차 기록",
      "    return stds",
      "",
      "xavier = lambda n_in, n_out: np.sqrt(2 / (n_in + n_out))",
      "he     = lambda n_in, n_out: np.sqrt(2 / n_in)",
    ], { x: 0.5, y: 1.5, w: 8.2, h: 4.6, size: 12, title: "step2_init.py" });
    d.explain(s, ["폭 512인 층 10개. 입력은 표준 정규분포", "가중치는 N(0, std²)에서 뽑는다 (std = 초기화 규칙)", "층을 지날 때마다 활성값의 표준편차를 기록", "std_fn만 바꿔서 '초기화 방식'을 비교한다", "표준편차가 1 근처 유지 = 안정, 0이나 큰 값으로 가면 소실/폭발"], { x: 8.9, y: 1.5, w: 3.9, h: 4.6, size: 12 });
    d.text(s, "실행: python step2_init.py", { x: 0.5, y: 6.3, w: 8, h: 0.4, size: 15, color: C.muted });
    s.addNotes("규칙을 함수로 주고받는 구조(std_fn)를 설명: '초기화 방식도 하나의 함수'.");
  }

  {
    const s = d.slide("실험 결과 ①: 층마다 활성값의 크기 (로그 눈금)", { tag: "2교시 · 실험" });
    img(s, "act_std", 0.5, 1.45, 7.6);
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const r = (a, b) => [{ text: a, options: { align: "left" } }, { text: b, options: { align: "center" } }];
    s.addTable([[H("설정"), H("10층 후 std")], r("tanh + std 0.01", "0.0000003"), r("tanh + std 1.0", "0.98 (포화)"), r("tanh + Xavier", "0.22"), r("ReLU + std 0.1", "79.7 (폭발)"), r("ReLU + Xavier", "0.023 (소실)"), r("ReLU + He", "0.73 (안정)")], { x: 8.3, y: 1.5, w: 4.5, colW: [2.6, 1.9], fontFace: FONT, fontSize: 13, color: C.ink, rowH: 0.45, border: { type: "solid", color: C.line, pt: 1 } });
    d.card(s, { x: 8.3, y: 4.85, w: 4.5, h: 1.85, fill: C.orangeL, title: "읽는 법", tColor: C.orange, size: 13, body: "로그 눈금이라 세로 한 칸이 10배! 너무 작은 초기값은 몇 층 만에 값이 사라지고, ReLU에는 He가 가장 안정적이다." });
    d.text(s, "(실제 실행 결과: 씨앗 0, 폭 512, 10층)", { x: 0.5, y: 5.75, w: 7.6, h: 0.4, size: 13, color: C.muted });
    s.addNotes("꼭 짚을 것 3가지: (1) 0.01은 3층 만에 거의 0. (2) ReLU에 Xavier를 쓰면 서서히 줄어든다 -> He가 필요한 이유. (3) tanh에서 std=1은 '폭발'이 아니라 +-1에 달라붙는 '포화' -> 기울기 소실. 원본의 '큰 분산=폭발'은 활성화 함수에 따라 폭발 또는 포화로 나타난다.");
  }

  {
    const s = d.slide("실험 결과 ②: tanh 활성값의 분포 (히스토그램)", { tag: "2교시 · 실험" });
    img(s, "act_hist", 0.5, 1.45, 7.8);
    d.card(s, { x: 8.5, y: 1.5, w: 4.3, h: 1.6, fill: C.tealL, title: "std = 0.01", tColor: C.teal, size: 14, body: "층이 깊어질수록 전부 0 한 점으로 모인다 → 소실" });
    d.card(s, { x: 8.5, y: 3.25, w: 4.3, h: 1.6, fill: C.redL, title: "std = 1.0", tColor: C.red, size: 14, body: "값이 ±1 끝에 달라붙는다(포화) → tanh 기울기 ≈ 0" });
    d.card(s, { x: 8.5, y: 5.0, w: 4.3, h: 1.6, fill: C.greenL, title: "Xavier", tColor: C.green, size: 14, body: "종 모양 분포가 유지된다 → 학습에 좋은 상태" });
    s.addNotes("분포를 눈으로 보는 것이 포인트. 포화된 뉴런은 기울기가 0에 가까워 역전파 신호가 끊긴다.");
  }

  {
    const s = d.slide("순전파만이 아니라 역전파도: 기울기의 분산", { tag: "2교시 · 초기화" });
    d.text(s, "이상적인 초기화는 두 조건을 모두 만족해야 한다:", { x: 0.7, y: 1.5, w: 12, h: 0.45, size: 19, bold: true });
    F(d, s, "Var(x⁽ˡ⁾) ≈ Var(x⁽ˡ⁺¹⁾)      (순전파 활성값의 분산이 층을 지나도 유지)", { x: 0.7, y: 2.1, w: 11.9, h: 0.7, size: 20 });
    F(d, s, "Var(∇x⁽ˡ⁾) ≈ Var(∇x⁽ˡ⁺¹⁾)    (역전파 기울기의 분산도 층을 지나도 유지)", { x: 0.7, y: 2.95, w: 11.9, h: 0.7, size: 20 });
    d.bullets(s, [
      "x⁽ˡ⁺¹⁾ : (l+1)번째 층의 활성값    ∇x⁽ˡ⁺¹⁾ : 그 활성값에 대한 손실함수의 기울기",
      "Var(∇x) : 역전파 중 기울기의 크기가 너무 커지거나 작아지지 않는지 보는 지표",
      "분산이 작으면 기울기 소실(앞쪽 층이 학습 안 됨), 크면 기울기 폭발(학습 발산)",
      "Xavier와 He는 이 두 조건을 만족시키려고 등장했다",
    ], { x: 0.7, y: 3.9, w: 12, h: 2.8, size: 17 });
    s.addNotes("순전파=앞으로 흐르는 신호, 역전파=뒤로 흐르는 학습 신호(기울기). 둘 다 건강해야 학습이 된다. n_in과 n_out이 같지 않으면 두 조건을 동시에 만족할 수 없어 Xavier는 절충식을 쓴다.");
  }

  {
    const s = d.slide("실전: 어떤 초기화를 쓸까? (PyTorch)", { tag: "2교시 · 정리" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t, b) => ({ text: t, options: { align: "left", bold: !!b } });
    s.addTable([
      [H("활성화 함수"), H("권장 초기화"), H("PyTorch 코드")],
      [c("tanh, (sigmoid)", 1), c("Xavier"), c("nn.init.xavier_normal_(layer.weight)")],
      [c("ReLU, Leaky ReLU", 1), c("He (Kaiming)"), c("nn.init.kaiming_normal_(w, nonlinearity='relu')")],
      [c("GELU 등", 1), c("He 계열 또는 기본값"), c("Transformer는 보통 N(0, 0.02²) 사용")],
    ], { x: 0.6, y: 1.5, w: 12.1, colW: [2.8, 3.0, 6.3], fontFace: FONT, fontSize: 15, color: C.ink, rowH: 0.62, valign: "middle", border: { type: "solid", color: C.line, pt: 1 } });
    d.bullets(s, [
      "편향(bias)은 보통 0으로 초기화한다",
      "PyTorch의 nn.Linear 기본 초기화는 Kaiming uniform 변형이라, 소규모 실험에서는 별도 설정이 없어도 잘 동작하는 편",
      "깊은 네트워크를 직접 만들거나 활성화 함수를 바꿀 때는 초기화도 함께 점검",
      "※ sigmoid는 출력 평균이 0.5라서 엄밀히는 '0 대칭'이 아님 → Xavier의 가정에서 조금 벗어남",
    ], { x: 0.6, y: 4.2, w: 12.1, h: 2.5, size: 16 });
    s.addNotes("최신 자료 반영: 현대 프레임워크는 대부분 합리적인 기본값을 제공하지만 원리를 알아야 이상 상황을 진단할 수 있다. GPT 계열은 표준편차 0.02 정규분포를 사용(소규모 설정).");
  }

  {
    const s = d.slide("2교시 정리", { tag: "2교시 · 정리" });
    d.bullets(s, [
      { t: "초기화 목표: 모든 층에서 활성값과 기울기의 분산을 일정하게 유지", b: true },
      "분산이 너무 크면 폭발/포화, 너무 작으면 소실 → 깊을수록 효과가 기하급수로 누적",
      "Var(y) = n_in · Var(w) · Var(x)  →  Var(w) ≈ 1/n_in 이 기본 원리",
      "Xavier: 2/(n_in+n_out), tanh 계열     |     He: 2/n_in, ReLU 계열",
      "실험으로 확인: ReLU + He만 10층 뒤에도 신호가 건강했다",
    ], { x: 0.7, y: 1.5, w: 12, h: 3.5, size: 19 });
    d.card(s, { x: 0.7, y: 5.2, w: 12, h: 1.5, fill: C.orangeL, title: "다음 시간 예고", tColor: C.orange, size: 16, body: "초기화는 '출발'만 안정시킨다. 학습 도중 층 입력 분포가 계속 흔들리는 문제는? → 정규화(Batch Norm, Layer Norm)" });
    s.addNotes("2교시 마무리 후 휴식.");
  }
};
