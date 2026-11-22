// 1교시: 파라미터, 경사 하강, 학습률, ∇, 지수이동평균(EMA), 모멘텀
const path = require("path");
const { C, FONT, IMG } = require("./lib");
const { img, F, table } = require("./helpers");

module.exports = function part1(d, pres) {
  // ---------------- 표지 ----------------
  {
    const s = pres.addSlide(); d.n += 1; s.background = { color: C.paper };
    s.addText("Lesson 14", { x: 0.9, y: 1.3, w: 11, h: 0.7, fontFace: FONT, fontSize: 28, bold: true, color: C.teal, margin: 0 });
    s.addText("Advanced Optimization", { x: 0.9, y: 2.0, w: 11.5, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    s.addText("개선된 최적화: 모델의 '조향 장치' 이해하기", { x: 0.9, y: 3.1, w: 11.5, h: 0.6, fontFace: FONT, fontSize: 24, color: C.teal, bold: true, margin: 0 });
    s.addText("경사 하강 · 모멘텀 · Adam · AdamW · Lion · Adafactor · 학습률 스케줄 · Sharp/Flat Minima", { x: 0.9, y: 3.8, w: 11.5, h: 0.5, fontFace: FONT, fontSize: 18, color: C.muted, margin: 0 });
    ["방향(모멘텀)", "보폭(적응형)", "메모리(효율)"].forEach((t, i) => d.box(s, t, { x: 0.9 + i * 2.7, y: 4.8, w: 2.5, h: 0.7, fill: [C.orangeL, C.tealL, C.greenL][i], size: 20, bold: true }));
    s.addText("튜링랩 프로그래밍학원\nTuringLab. Programming Academy", { x: 0.9, y: 5.9, w: 6, h: 0.7, fontFace: FONT, fontSize: 14, color: C.ink, margin: 0 });
    s.addImage({ path: path.join(IMG, "logo.png"), x: 9.5, y: 6.0, w: 3.0, h: 3.0 * 118 / 756 });
    s.addNotes("강의 소개. 오늘 3시간(50분 x 3교시)의 흐름: 1교시 경사 하강·학습률·지수이동평균·모멘텀(기초), 2교시 Adam과 AdamW, 3교시 Lion·Adafactor·학습률 스케줄·sharp/flat minima·종합 비교. 모든 숫자는 NumPy/PyTorch로 직접 돌려서 얻은 값이다.");
  }

  // ---------------- 로드맵 ----------------
  {
    const s = d.slide("오늘의 로드맵: 3교시 (50분 수업 + 10분 휴식) × 3", { tag: "OVERVIEW" });
    table(s, ["교시", "주제", "직접 해보는 것"], [
      [{ text: "1교시", options: { bold: true } }, "파라미터 · 경사 하강 · 학습률 · ∇ · 지수이동평균(EMA) · 모멘텀", "Step 1  GD 손계산, EMA, 협곡에서 SGD vs 모멘텀"],
      [{ text: "2교시", options: { bold: true } }, "Adam (1·2차 모멘트, 편향 보정) · AdamW (분리된 가중치 감쇠)", "Step 2  Adam 구현 + torch.optim 과 비교"],
      [{ text: "3교시", options: { bold: true } }, "Lion · Adafactor · 학습률 스케줄 · sharp/flat · 종합 비교", "Step 3~6  NumPy 구현 후 PyTorch 비교 실험"],
      [{ text: "마무리", options: { bold: true } }, "원본 자료 검토 결과 · 실습 문제 5개 + 솔루션", "손계산 · 구현 · 메모리 계산"],
    ], { x: 0.6, y: 1.6, w: 12.1, colW: [1.5, 6.4, 4.2], fontSize: 16, rowH: 0.75 });
    d.text(s, "준비물: Python 3.9+, NumPy, scikit-learn(손글씨 데이터), PyTorch 2.5+ (torch.optim.Adafactor), matplotlib   |   코드는 lesson14/code 폴더 (NumPy 버전 먼저, PyTorch 버전은 그 다음)", { x: 0.6, y: 5.9, w: 12.1, h: 0.6, size: 15, color: C.muted });
  }

  // ---------------- 학습 목표 ----------------
  {
    const s = d.slide("이 강의가 끝나면 할 수 있는 일", { tag: "GOAL" });
    const items = [
      ["1", "설명하기", "경사 하강, 모멘텀, 적응형 학습률, 가중치 감쇠가 각각 무엇을 해결하는지 내 말로 설명한다"],
      ["2", "계산하기", "EMA·Adam·Lion·코사인 스케줄의 한 걸음을 손으로 계산한다 (초심자용 숫자 예제)"],
      ["3", "구현하기", "SGD·Adam·AdamW·Lion·Adafactor 를 NumPy 로 만들고 torch.optim 과 같은 값이 나오는지 확인한다"],
      ["4", "선택하기", "메모리·잡음·모델 종류에 따라 옵티마이저와 학습률 스케줄을 고르고, 원본 자료의 오류를 알아챈다"],
    ];
    items.forEach((it, i) => {
      const y = 1.6 + i * 1.2;
      s.addText(it[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 22, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, it[1], { x: 1.7, y, w: 3.2, h: 0.5, size: 20, bold: true });
      d.text(s, it[2], { x: 1.7, y: y + 0.47, w: 10.8, h: 0.5, size: 16, color: C.muted });
    });
  }

  // ---------------- 1교시 ----------------
  d.section("1교시", "경사 하강부터 모멘텀까지", "파라미터 · 경사 하강 · 학습률 · 델(∇) · 지수이동평균(EMA) · 모멘텀", "1교시 시작. 지난 시간(Lesson 13)에는 데이터 쪽에서 일반화를 높이는 증강을 배웠다. 오늘은 '파라미터를 어떻게 업데이트하는가', 즉 옵티마이저를 다룬다.");

  {
    const s = d.slide("옵티마이저 = 모델 학습의 '조향 장치'", { tag: "1교시 · 큰 그림" });
    d.card(s, { x: 0.6, y: 1.55, w: 3.9, h: 2.1, fill: C.tealL, title: "지형 = 손실 표면", tColor: C.teal, size: 15, body: "파라미터를 바꿀 때마다 '틀린 정도(손실)'가 달라진다. 이것을 높낮이로 그린 것이 손실 표면." });
    d.card(s, { x: 4.7, y: 1.55, w: 3.9, h: 2.1, fill: C.orangeL, title: "옵티마이저 = 운전자", tColor: C.orange, size: 15, body: "기울기(경사)를 보고 어느 방향으로, 얼마나 갈지 정한다. 오늘의 주인공!" });
    d.card(s, { x: 8.8, y: 1.55, w: 3.9, h: 2.1, fill: C.greenL, title: "목적지 = 낮은 손실", tColor: C.green, size: 15, body: "손실이 가장 낮은 골짜기에 안전하게 도착하는 것이 목표." });
    d.bullets(s, [
      { t: "현대 딥러닝은 효율적이고 안정적인 최적화 알고리즘에 의존한다", b: true },
      "신경망의 본질은 수백만 개의 파라미터를 신중하게, 반복적으로 업데이트하는 것",
      "오늘 배울 최신 기법: AdamW, Lion, Adafactor (그리고 이들의 뿌리인 SGD·모멘텀·Adam)",
    ], { x: 0.6, y: 4.0, w: 12.1, h: 2.4, size: 18 });
    s.addNotes("자동차 비유: 지형(손실 표면), 차량(모델), 운전 방식(옵티마이저), 목적지(최소 손실). 운전 방식에 따라 같은 지형도 빨리 갈 수도, 헤맬 수도 있다.");
  }

  {
    const s = d.slide("📌 매개변수(Parameter)란? — 학습으로 스스로 조정되는 값", { tag: "1교시 · 용어" });
    F(d, s, "y = w · x + b", { x: 0.6, y: 1.5, w: 12.1, h: 0.85, size: 30, fill: C.tealL });
    d.card(s, { x: 0.6, y: 2.65, w: 5.9, h: 2.3, fill: C.orangeL, title: "① 가중치 (Weight) w", tColor: C.orange, size: 16, body: "입력값에 곱해지는 값.\n입력 x 가 결과에 얼마나 영향을 주는지 결정한다.\n학습하면서 계속 바뀐다." });
    d.card(s, { x: 6.8, y: 2.65, w: 5.9, h: 2.3, fill: C.purpleL, title: "② 편향 (Bias) b", tColor: C.purple, size: 16, body: "결과를 통째로 위·아래로 옮기는 값.\n(예: 키에 상관없이 기본 점수를 +5점)\n이것도 학습으로 조정된다." });
    d.text(s, "신경망의 한 층 = 입력 → (가중치 × 입력 + 편향) → 활성화 함수 → 출력.  이것을 여러 층 반복한다. 모든 층의 w 와 b 를 모아 놓은 것이 모델의 '파라미터 집합 θ' 이다.", { x: 0.6, y: 5.25, w: 12.1, h: 1.2, size: 17 });
  }

  {
    const s = d.slide("파라미터는 몇 개일까? — m × (n + 1)", { tag: "1교시 · 계산" });
    F(d, s, "입력 n 개 → 출력 m 개 :  가중치 n×m 개 + 편향 m 개  =  m × (n + 1) 개", { x: 0.6, y: 1.5, w: 12.1, h: 0.8, size: 24, fill: C.tealL });
    table(s, ["층", "n (입력)", "m (출력)", "m × (n+1)", "비고"], [
      ["우리 실습 1층", "64", "64", "64 × 65 = 4,160", "8×8 손글씨 → 은닉 64"],
      ["우리 실습 2층", "64", "10", "10 × 65 = 650", "은닉 64 → 숫자 10종"],
      [{ text: "실습 모델 합계", options: { bold: true } }, "", "", { text: "4,810", options: { bold: true } }, "step6 에서 torch 로 센 값과 일치 (assert)"],
      ["MNIST 28×28 → 128", "784", "128", "128 × 785 = 100,480", "층 하나가 10만 개"],
      ["4096 → 4096 (대형 모델의 한 행렬)", "4096", "4096", "4096 × 4097 ≈ 1,678만", "이런 행렬이 수십~수백 개"],
    ], { x: 0.6, y: 2.6, w: 12.1, colW: [3.3, 1.4, 1.4, 3.0, 3.0], fontSize: 14, rowH: 0.55 });
    d.text(s, "→ 파라미터가 수백만~수십억 개이므로, 한 걸음마다 '모든 파라미터를 어떻게 바꿀지'를 싸고 안정적으로 정하는 알고리즘이 필요하다.", { x: 0.6, y: 6.2, w: 12.1, h: 0.7, size: 15, color: C.muted });
  }

  {
    const s = d.slide("역전파에서 최적화까지: 한 번의 훈련 단계", { tag: "1교시 · 흐름" });
    const names = ["입력 x\n+ 정답 y", "순전파\n예측 ŷ", "손실 L\n(틀린 정도)", "역전파\n기울기 ∇L", "옵티마이저\n(오늘의 주제)", "파라미터\nθ 갱신"];
    const fills = [C.gray, C.tealL, C.tealL, C.orangeL, C.purpleL, C.greenL];
    names.forEach((t, i) => {
      d.box(s, t, { x: 0.6 + i * 2.05, y: 1.8, w: 1.75, h: 1.0, fill: fills[i], size: 14, bold: i === 4, line: i === 4 ? C.purple : null });
      if (i < 5) d.arrow(s, 2.37 + i * 2.05, 2.3, 2.62 + i * 2.05, 2.3);
    });
    d.bullets(s, [
      "기울기 ∇L 은 손실이 '가장 가파르게 커지는 방향'을 가리킨다 → 손실을 줄이려면 반대로 가야 한다",
      { t: "최적화 = 기울기를 '의미 있는 파라미터 업데이트'로 바꾸는 알고리즘", b: true },
      "옵티마이저가 결정하는 것 두 가지: ① 어느 방향으로  ② 얼마나(보폭)",
      "이 과정을 수천~수백만 번 반복하는 것이 '학습'이다",
    ], { x: 0.6, y: 3.3, w: 12.1, h: 3.3, size: 18 });
    s.addNotes("역전파(Lesson 5~6 내용)까지가 '기울기 계산', 그 다음이 오늘 배우는 '기울기로 어떻게 움직일까'. 같은 기울기라도 옵티마이저에 따라 움직임이 달라진다.");
  }

  {
    const s = d.slide("경사 하강(Gradient Descent): 기본으로 돌아가기", { tag: "1교시 · 식 14.1" });
    F(d, s, "θ(k+1)  =  θ(k)  −  η · ∇L( θ(k) )          (식 14.1)", { x: 0.6, y: 1.5, w: 12.1, h: 0.9, size: 28, fill: C.tealL });
    table(s, ["기호", "뜻", "쉬운 말"], [
      ["θ", "모든 파라미터를 모은 벡터", "지금 모델이 가진 모든 w, b"],
      ["θ(k)", "k 번째 걸음에서의 파라미터 값", "현재 위치"],
      ["L( θ(k) )", "현재 파라미터에서의 손실", "얼마나 틀렸나 (높이)"],
      ["∇L( θ(k) )", "손실을 θ 로 미분한 값 (기울기)", "가장 가파르게 '오르막'인 방향"],
      ["η (에타)", "학습률 (learning rate)", "한 번에 가는 보폭"],
    ], { x: 0.6, y: 2.7, w: 12.1, colW: [2.4, 5.2, 4.5], fontSize: 15, rowH: 0.5 });
    d.text(s, "한 줄 요약: \"현재 위치에서 기울기를 구하고, 그 반대 방향으로 학습률만큼 이동한다.\"  (이 식은 예전에 본 가중치 갱신식에서 w 를 θ 로 바꾼 것일 뿐, 완전히 같다.)", { x: 0.6, y: 5.75, w: 12.1, h: 0.9, size: 16, color: C.muted });
    s.addNotes("η 는 '이타'/'에타'. 원본의 [!NOTE]처럼 예전에 본 w ← w − η·dL/dw 와 같은 식임을 강조한다.");
  }

  {
    const s = d.slide("값을 '빼는' 이유와 손계산 한 걸음", { tag: "1교시 · 손계산" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 2.4, fill: C.orangeL, title: "왜 마이너스(−)?", tColor: C.orange, size: 16, body: "기울기는 손실이 가장 빨리 '증가'하는 방향.\n손실을 줄이려면 반대로 가야 하니까 η·기울기를 빼 준다.\n(산에서 내려오려면 오르막의 반대 방향!)" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 2.4, fill: C.tealL, title: "예제: L(θ) = θ²", tColor: C.teal, size: 16, body: "기울기 = dL/dθ = 2θ\n시작 θ = 3, 학습률 η = 0.1\n기울기 = 2×3 = 6 (양수: 오른쪽이 오르막)" });
    F(d, s, "θ(새) = 3 − 0.1 × 6 = 2.4          손실  9 → 5.76", { x: 0.6, y: 4.25, w: 12.1, h: 0.8, size: 26, fill: C.greenL });
    d.text(s, "한 걸음 더: θ=2.4 → 기울기 4.8 → θ = 2.4 − 0.48 = 1.92 → …  매번 θ에 (1 − 2η) = 0.8 을 곱하게 되어 0 으로 다가간다.", { x: 0.6, y: 5.35, w: 12.1, h: 0.8, size: 17 });
    d.text(s, "코드: step1_gd_ema.py  (assert 로 2.4 확인)", { x: 0.6, y: 6.3, w: 12.1, h: 0.4, size: 14, color: C.muted });
  }

  {
    const s = d.slide("학습률 η: 너무 작아도, 너무 커도 문제", { tag: "1교시 · 학습률" });
    img(s, "lr_effect", 1.35, 1.4, 10.6);
    table(s, ["η", "곱해지는 값 (1−2η)", "20걸음 뒤 θ (실측)", "해석"], [
      ["0.01", "+0.98", "2.003", "너무 느리다 (아직 멀었다)"],
      ["0.1", "+0.80", "0.0346", "적당: 0 으로 부드럽게 접근"],
      ["0.9", "−0.80", "0.0346", "부호가 매번 뒤집히며 진동하지만 결국 수렴"],
      ["1.05", "−1.10", "20.18", "|값|>1 → 점점 커짐 = 발산"],
    ], { x: 0.6, y: 4.65, w: 12.1, colW: [1.2, 3.2, 3.2, 4.5], fontSize: 13, rowH: 0.4 });
    s.addNotes("학습률의 핵심: 너무 크면 발산, 너무 작으면 학습이 느리다. 수치는 step1_gd_ema.py 의 출력이다.");
  }

  {
    const s = d.slide("∇ (델, nabla) 연산자: 딥러닝에서는 '기울기'만 기억하자", { tag: "1교시 · 수학 보충" });
    F(d, s, "∇ = ( ∂/∂x ,  ∂/∂y ,  ∂/∂z )      편미분들을 모아 놓은 벡터 연산자", { x: 0.6, y: 1.5, w: 12.1, h: 0.8, size: 22, fill: C.tealL });
    table(s, ["형태", "이름", "뜻", "딥러닝 학습에서?"], [
      ["∇f", "Gradient (기울기)", "f 가 가장 빠르게 증가하는 방향과 그 증가율", { text: "매 걸음 사용 ★", options: { bold: true, color: C.teal } }],
      ["∇ · F", "Divergence (발산)", "벡터장이 한 점에서 퍼져 나가는 정도 (물이 새는 정도)", "직접 쓰지 않음 (물리·유체)"],
      ["∇ × F", "Curl (회전)", "벡터장이 도는 정도 (소용돌이 세기)", "직접 쓰지 않음 (물리·전자기)"],
    ], { x: 0.6, y: 2.5, w: 12.1, colW: [1.3, 2.6, 5.4, 2.8], fontSize: 13, rowH: 0.55 });
    d.card(s, { x: 0.6, y: 5.0, w: 12.1, h: 1.75, fill: C.orangeL, title: "∂ (편미분)은 '너만 빼고 다 그대로!'", tColor: C.orange, size: 15, body: "변수가 여러 개일 때, 하나만 아주 조금 움직이고 나머지는 고정한 채 결과가 얼마나 변하는지 본다.\n예: 온도 f(x,y,z) 에서 ∂f/∂x = 'y, z 는 그대로 두고 x 만 움직일 때의 변화율'. ∇f 는 이 값들을 한 줄로 모은 것이다." });
    s.addNotes("원본에는 발산/회전도 나오지만 오늘 필요한 것은 gradient 뿐이라는 점을 분명히 한다. 정보 과다를 막기 위한 정리 슬라이드.");
  }

  {
    const s = d.slide("현실의 손실 표면은 매끄러운 언덕이 아니다", { tag: "1교시 · 문제" });
    d.card(s, { x: 0.6, y: 1.55, w: 3.9, h: 2.5, fill: C.redL, title: "① 잡음(noisy)", tColor: C.red, size: 15, body: "전체 데이터가 아니라 미니배치(일부)로 기울기를 계산한다 → 기울기가 매번 조금씩 다르다 (확률적, SGD)." });
    d.card(s, { x: 4.7, y: 1.55, w: 3.9, h: 2.5, fill: C.orangeL, title: "② 고차원", tColor: C.orange, size: 15, body: "파라미터가 수백만 개 → 방향도 수백만 개. 어떤 방향은 아주 가파르고 어떤 방향은 거의 평평하다." });
    d.card(s, { x: 8.8, y: 1.55, w: 3.9, h: 2.5, fill: C.purpleL, title: "③ 협곡(ravine)", tColor: C.purple, size: 15, body: "한 방향은 급경사, 다른 방향은 완만한 좁고 긴 골짜기. 단순 경사 하강은 벽에 부딪히며 지그재그로 간다." });
    d.bullets(s, [
      { t: "단순한 경사 하강은 이 복잡성에 고전한다 → 더 개선된 최적화가 필요한 이유", b: true },
      "고쳐야 할 점 ①: 지그재그와 잡음 → 과거 기울기를 평균 내 '방향'을 안정시키자 (모멘텀)",
      "고쳐야 할 점 ②: 방향마다 가파름이 다르다 → 파라미터마다 '보폭'을 따로 정하자 (적응형 학습률)",
      "고쳐야 할 점 ③: 메모리·일반화·학습률 변화 → Lion · Adafactor · 스케줄 · 가중치 감쇠",
    ], { x: 0.6, y: 4.35, w: 12.1, h: 2.4, size: 16 });
    s.addNotes("원본의 [그림 14.1]은 공이 복잡한 손실 표면을 굴러가는 그림이다. 칠판에 봉우리와 골짜기를 그리며 '지역 최소값을 벗어나는 것이 어렵다'는 설명을 덧붙인다.");
  }

  {
    const s = d.slide("지수이동평균(EMA): 최근 것을 더 중요하게 평균내기", { tag: "1교시 · EMA" });
    d.card(s, { x: 0.6, y: 1.5, w: 5.9, h: 1.9, fill: C.gray, title: "단순 이동 평균 (SMA)", tColor: C.muted, size: 15, body: "최근 k 개만 똑같은 비중으로 평균, 나머지는 완전히 버린다.\n(x(t) + x(t−1) + … + x(t−k+1)) / k" });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 1.9, fill: C.tealL, title: "지수 이동 평균 (EMA)", tColor: C.teal, size: 15, body: "모든 과거를 쓰되, 오래될수록 비중이 지수적으로 작아진다.\nm(k) = β · m(k−1) + (1−β) · g(k)" });
    img(s, "ema", 0.6, 3.55, 8.2);
    d.text(s, "β=0.9 의 비중\n현재  0.1\n1단계 전  0.09\n2단계 전  0.081\n3단계 전  0.0729\n⋮ 점점 빠르게 작아짐", { x: 9.5, y: 3.65, w: 3.2, h: 2.4, size: 15 });
    d.text(s, "β 가 1에 가까울수록 과거를 오래 기억(매끄럽지만 느림), 작을수록 최근에 민감.", { x: 0.6, y: 6.3, w: 12.1, h: 0.5, size: 14, color: C.muted });
  }

  {
    const s = d.slide("EMA 를 직접 펼쳐 보자 (초심자용 손계산)", { tag: "1교시 · EMA 전개" });
    d.text(s, "m(0) = 0,  β = 0.9,  기울기 g 가 계속 1 이라면?", { x: 0.6, y: 1.45, w: 12.1, h: 0.45, size: 20, bold: true, color: C.teal });
    table(s, ["걸음 k", "계산", "m(k)", "설명"], [
      ["1", "0.9 × 0 + 0.1 × 1", "0.100", "진짜 값은 1인데 0 에서 출발해서 작게 시작"],
      ["2", "0.9 × 0.1 + 0.1 × 1", "0.190", "조금씩 따라 올라간다"],
      ["3", "0.9 × 0.19 + 0.1 × 1", "0.271", "…"],
      ["∞", "", "→ 1", "충분히 지나면 진짜 평균 1 에 도달"],
    ], { x: 0.6, y: 2.0, w: 12.1, colW: [1.3, 3.8, 1.6, 5.4], fontSize: 15, rowH: 0.5 });
    F(d, s, "m(k) = (1−β)·g(k) + β(1−β)·g(k−1) + β²(1−β)·g(k−2) + …", { x: 0.6, y: 4.5, w: 12.1, h: 0.8, size: 22, fill: C.tealL });
    d.bullets(s, [
      "식을 계속 대입해 펼치면 위처럼 되고, 오래된 항일수록 β 가 여러 번 곱해져 작아진다 (0<β<1)",
      "비중을 전부 더하면 (1−β)(1+β+β²+…) = 1 → 진짜 '평균'이다",
      "※ 위 표의 '작게 시작'하는 문제가 2교시의 '편향 보정'으로 이어진다",
    ], { x: 0.6, y: 5.5, w: 12.1, h: 1.3, size: 15 });
    s.addNotes("학생이 직접 계산기로 0.1, 0.19, 0.271 을 확인하게 한다. 코드: step1_gd_ema.py (assert 로 0.1/0.09/0.081 확인, 실측 잡음 감소: 표준편차 2.07 → 0.44).");
  }

  {
    const s = d.slide("모멘텀(Momentum): 굴러가는 공의 관성", { tag: "1교시 · 모멘텀" });
    F(d, s, "v ← μ·v + g ,     θ ← θ − η·v          (μ ≈ 0.9)", { x: 0.6, y: 1.5, w: 12.1, h: 0.85, size: 28, fill: C.tealL });
    d.card(s, { x: 0.6, y: 2.65, w: 5.9, h: 2.2, fill: C.greenL, title: "같은 방향이 계속될 때", tColor: C.green, size: 15, body: "속도 v 가 점점 쌓여 빨라진다 (최대 약 1/(1−μ) = 10배까지)." });
    d.card(s, { x: 6.8, y: 2.65, w: 5.9, h: 2.2, fill: C.orangeL, title: "방향이 +, −, +, − 로 흔들릴 때", tColor: C.orange, size: 15, body: "서로 상쇄되어 진동이 줄어든다 → 협곡에서 지그재그 감소." });
    d.bullets(s, [
      "기본 SGD: 매 걸음 그 순간의 기울기만 본다 → 방향이 휙휙 바뀐다",
      "모멘텀: 이전 방향을 일정 비율 유지 = '관성' → 더 부드럽게 이동",
      "Adam 의 1차 모멘트 m 도 같은 아이디어 (EMA 형태) — 내일의 Adam 으로 가는 징검다리",
    ], { x: 0.6, y: 5.1, w: 12.1, h: 1.7, size: 16 });
    s.addNotes("주의: PyTorch 의 SGD(momentum) 는 v ← μv + g 형태이고, 원본 Adam 설명의 m 은 (1−β) 가중 EMA 형태이다. 두 표기의 차이는 학습률 스케일뿐이다.");
  }

  {
    const s = d.slide("협곡 실험: L = ½(x² + 100y²)  —  SGD vs 모멘텀 vs Adam", { tag: "1교시 · 실험 (실측)" });
    img(s, "valley", 1.5, 1.4, 10.3);
    d.bullets(s, [
      "SGD: y 방향이 100배 가파르므로 학습률을 0.02(=2/100) 아래로 눌러야 한다 → 100걸음 중 y 부호가 100번 뒤집힘(지그재그), 손실 6.9e-1",
      "모멘텀: 같은 100걸음 뒤 손실 3.1e-4 (약 2000배 낮음), 지그재그 35번",
      "Adam(3.2e-4): 방향마다 보폭을 따로 정해 협곡을 부드럽게 따라 내려간다 — 2교시에서 원리를 배운다",
    ], { x: 0.6, y: 4.85, w: 12.1, h: 1.95, size: 14 });
    s.addNotes("수치는 step1_gd_ema.py / step2_adam.py 의 실측값이다. 모멘텀 경로는 x=0 근처에서 약간 지나쳤다가 되돌아온다(오버슈트) — 관성의 부작용.");
  }

  {
    const s = d.slide("Step 1 코드: GD 한 걸음 · EMA · 모멘텀 (NumPy)", { tag: "1교시 · 코드" });
    d.code(s, [
      "theta, lr = 3.0, 0.1",
      "grad = 2 * theta            # L=θ² 의 기울기 = 6",
      "theta = theta - lr * grad   # 3 - 0.1*6 = 2.4",
      "",
      "beta, m = 0.9, 0.0",
      "for x in g:                 # g: 잡음 낀 기울기들",
      "    m = beta*m + (1-beta)*x # 지수이동평균",
      "",
      "class SGD:                  # optim_np.py",
      "    def step(self, params, grads):",
      "        for p, g, v in zip(params, grads, self.v):",
      "            v *= self.mu; v += g   # 속도: 관성 + 새 기울기",
      "            p -= self.lr * v       # 속도 방향으로 이동",
    ], { x: 0.6, y: 1.5, w: 7.7, h: 5.2, size: 12, title: "step1_gd_ema.py · optim_np.py" });
    d.explain(s, [
      "기울기의 반대 방향으로 η 만큼: 손실이 줄어든다",
      "EMA: 과거(β) + 새 값(1−β) 의 가중 평균",
      "SGD 클래스: μ=0 이면 순수 경사 하강, μ=0.9 면 모멘텀 SGD",
      "p -= … 는 배열을 '제자리에서' 바꾼다 (파라미터가 바로 갱신됨)",
      "실행하면 assert 가 2.4, 0.1/0.09/0.081 을 검사한다",
    ], { x: 8.5, y: 1.5, w: 4.2, h: 5.2, size: 13 });
  }

  {
    const s = d.slide("1교시 정리 · 확인 질문", { tag: "1교시 · 정리" });
    d.bullets(s, [
      "파라미터 = 학습으로 조정되는 w, b.  n 입력 m 출력 한 층의 개수는 m(n+1)",
      "경사 하강: θ(새) = θ − η∇L.  기울기의 반대 방향으로 η 만큼 이동",
      "학습률: 너무 작으면 느리고, 너무 크면 진동·발산",
      "EMA: 최근 값을 더 크게 반영하는 평균 (β=0.9 → 0.1, 0.09, 0.081, …)",
      "모멘텀: 기울기의 EMA 로 방향을 안정화 → 협곡에서 지그재그 감소",
    ], { x: 0.6, y: 1.5, w: 12.1, h: 3.0, size: 17 });
    d.card(s, { x: 0.6, y: 4.65, w: 12.1, h: 2.1, fill: C.tealL, title: "확인 질문 (정답은 노트)", tColor: C.teal, size: 15, body: "Q1. 기울기가 −2, η=0.5, θ=1 이면 새 θ 는?     Q2. β=0.5 일 때 현재·1단계 전·2단계 전의 비중은?\nQ3. 모멘텀이 지그재그를 줄이는 이유를 한 문장으로.    Q4. 학습률을 올렸더니 손실이 점점 커진다. 무슨 일일까?" });
    s.addNotes("정답: Q1 θ = 1 − 0.5×(−2) = 2.  Q2 0.5, 0.25, 0.125.  Q3 +/−로 흔들리는 기울기가 평균에서 상쇄되기 때문.  Q4 학습률이 너무 커서 발산(|1−ηλ|>1).");
  }
};
