// 3교시: Lion, Adafactor, 학습률 스케줄, sharp/flat minima, 종합 비교
const { C, FONT } = require("./lib");
const { img, F, table } = require("./helpers");

module.exports = function part3(d, pres) {
  d.section("3교시", "Lion · Adafactor · 학습률 스케줄", "부호 기반 모멘텀 · 메모리 효율 · 웜업/코사인 · sharp vs flat · 종합 비교", "3교시 시작. 1~2교시는 '방향'과 '보폭'을 다뤘다. 3교시는 메모리를 줄이는 방법(Lion, Adafactor), 학습률을 시간에 따라 바꾸는 방법, 그리고 일반화와의 관계를 본다.");

  {
    const s = d.slide("Lion: 기울기의 '크기'를 버리고 '부호'만 쓴다", { tag: "3교시 · Lion" });
    F(d, s, "c = sign( β1·m + (1−β1)·g ) ,    θ ← θ − η·( c + λθ ) ,    m ← β2·m + (1−β2)·g", { x: 0.6, y: 1.45, w: 12.1, h: 0.8, size: 20, fill: C.purpleL });
    d.bullets(s, [
      { t: "2023년 Google 연구진이 제안 (진화적 탐색으로 발견된 옵티마이저)", b: true },
      "업데이트에 쓰는 값은 sign(·) → 항상 −1, 0, +1.  모든 파라미터가 똑같은 크기 η 로 이동",
      "2차 모멘트 v 가 없다 → 상태가 m 하나뿐: 메모리·계산이 Adam 보다 가볍다",
      "ViT 같은 비전 모델, GPU 메모리가 빠듯한 환경에서 AdamW 보다 나은 보고가 있다 (일부 설정)",
      "논문의 안내: AdamW 보다 학습률은 3~10배 작게, 가중치 감쇠(λ)는 3~10배 크게",
    ], { x: 0.6, y: 2.55, w: 7.4, h: 4.2, size: 15 });
    d.card(s, { x: 8.3, y: 2.55, w: 4.4, h: 4.2, fill: C.orangeL, title: "원본 식과 달라진 점", tColor: C.orange, size: 13, body: "원본 식 14.7 은 sign(β2·m(k+1) + (1−β2)·g(k)) 로 적혀 있다.\n① 아직 계산 전인 m(k+1) 을 쓸 수 없고\n② β1/β2 의 역할이 뒤바뀌었다.\n논문(Chen et al., 2023): 업데이트 방향에 β1, 모멘텀 갱신에 β2.\n③ 가중치 감쇠 λθ 항이 빠져 있다." });
    s.addNotes("원본 검토 항목. 우리 구현(optim_np.Lion)은 논문 순서를 따른다: 부호 계산 → 파라미터 갱신 → 모멘텀 갱신. 기본값 β1=0.9, β2=0.99.");
  }

  {
    const s = d.slide("Lion 코드와 실측: 기울기가 100만 배 달라도 걸음은 같다", { tag: "3교시 · Lion 코드" });
    d.code(s, [
      "class Lion:      # optim_np.py",
      "  def step(self, params, grads):",
      "    for p, g, m in zip(params, grads, self.m):",
      "      c = np.sign(b1*m + (1-b1)*g)    # 부호만 사용",
      "      p -= lr * (c + wd * p)          # 모두 같은 크기 η",
      "      m *= b2; m += (1-b2) * g        # 모멘텀 갱신(β2)",
      "",
      "# PyTorch (직접 만든 Optimizer 서브클래스)",
      "c = (b1*m + (1-b1)*p.grad).sign()",
      "p.add_(c + wd * p, alpha=-lr)",
      "m.mul_(b2).add_(p.grad, alpha=1-b2)",
    ], { x: 0.6, y: 1.5, w: 7.2, h: 3.9, size: 12, title: "Lion (NumPy · PyTorch)" });
    table(s, ["기울기 크기", "Lion 이동량 (η=0.01)"], [
      ["× 0.001", "[0.01 0.01 0.01 0.01]"],
      ["× 1", "[0.01 0.01 0.01 0.01]"],
      ["× 1000", "[0.01 0.01 0.01 0.01]"],
    ], { x: 8.1, y: 1.5, w: 4.6, colW: [1.7, 2.9], fontSize: 13, rowH: 0.5 });
    d.text(s, "NumPy Lion 과 PyTorch Lion 은 30걸음 뒤 최대 차이 1.1e-16 으로 일치 (step6, assert).", { x: 8.1, y: 3.6, w: 4.6, h: 1.0, size: 13, color: C.muted });
    d.bullets(s, [
      "torch.optim 에는 Lion 이 없다 → torch.optim.Optimizer 를 상속해 직접 만든다 (30줄 안팎)",
      "걸음 크기가 늘 η 라서 끝까지 η 를 유지하면 바닥에서 흔들린다 → 학습률 감소(코사인 등)와 함께 쓰는 것이 좋다",
    ], { x: 0.6, y: 5.6, w: 12.1, h: 1.2, size: 15 });
  }

  {
    const s = d.slide("옵티마이저는 메모리도 먹는다: 파라미터 하나당 추가 숫자", { tag: "3교시 · 메모리" });
    table(s, ["옵티마이저", "파라미터당 추가 상태", "70억 파라미터, float32 로 상태만", "비고"], [
      ["SGD", "0 개", "0 GB", "모멘텀을 쓰면 1개 (속도 v)"],
      ["SGD + 모멘텀", "1 개 (v)", "28 GB", ""],
      ["Adam / AdamW", "2 개 (m, v)", "56 GB", "모델 가중치(28 GB)와 별도!"],
      ["Lion", "1 개 (m)", "28 GB", "Adam 의 절반"],
      ["Adafactor", "행+열 합만 (≈ 0)", "수 MB 이하", "행렬 n×m 에 n+m 개"],
    ], { x: 0.6, y: 1.45, w: 12.1, colW: [2.6, 2.9, 3.6, 3.0], fontSize: 14, rowH: 0.5 });
    d.bullets(s, [
      "학습할 때는 가중치 + 기울기 + 옵티마이저 상태 (+ 활성값) 가 모두 GPU 메모리에 올라간다",
      "실측(step6, 우리 작은 모델 4,810 파라미터): AdamW 상태 9,620개, Lion 4,810개, Adafactor 276개",
      "그래서 큰 모델일수록 '메모리 효율적인 옵티마이저'가 중요해진다",
    ], { x: 0.6, y: 5.0, w: 12.1, h: 1.8, size: 15 });
    s.addNotes("실습 문제 4번과 연결된다. 70억 파라미터 × 4바이트 = 28 GB, Adam 은 m,v 두 개라 56 GB.");
  }

  {
    const s = d.slide("Adafactor: 2차 모멘트를 '행 벡터 × 열 벡터'로 압축", { tag: "3교시 · Adafactor" });
    d.text(s, "Adam 은 v 를 가중치 행렬과 같은 크기(n×m)로 저장한다. Adafactor 는 v 대신 행 합 R(길이 n)과 열 합 C(길이 m)만 저장하고 필요할 때 곱해서 복원한다.", { x: 0.6, y: 1.45, w: 12.1, h: 0.8, size: 16 });
    d.text(s, "Adam : v  (n × m 개)", { x: 0.9, y: 2.45, w: 4, h: 0.4, size: 15, bold: true, color: C.purple });
    d.grid(s, [[1, 2, 3, 2], [2, 4, 6, 4], [3, 6, 9, 6]], { x: 0.9, y: 2.9, cw: 0.7, ch: 0.5, fill: C.purpleL, size: 14 });
    d.text(s, "Adafactor : R (n개) , C (m개)", { x: 4.6, y: 2.45, w: 4.5, h: 0.4, size: 15, bold: true, color: C.green });
    d.grid(s, [[6], [12], [18]], { x: 4.6, y: 2.9, cw: 0.8, ch: 0.5, fill: C.greenL, size: 14 });
    d.grid(s, [[6, 12, 18, 12]], { x: 5.6, y: 2.9, cw: 0.7, ch: 0.5, fill: C.greenL, size: 14 });
    d.text(s, "R = 행 합,  C = 열 합\n(위 예: v 가 '행 패턴 × 열 패턴'이면 정확히 복원)", { x: 4.6, y: 4.5, w: 4.5, h: 0.8, size: 13, color: C.muted });
    F(d, s, "v̂ ≈ R · Cᵀ / ΣR", { x: 9.3, y: 3.0, w: 3.4, h: 0.8, size: 24, fill: C.tealL });
    d.bullets(s, [
      { t: "메모리: n×m  →  n+m  (원본의 'O(n²) → O(n)' 은 정사각 행렬일 때만 맞는 표현; 일반식은 O(nm) → O(n+m))", b: true },
      "T5 같은 아주 큰 언어 모델을 제한된 하드웨어에서 학습 가능. 1차원 파라미터(편향)는 일반 v 를 사용",
      "업데이트가 너무 크지 않도록 RMS 클리핑, 시간에 따라 변하는 β2(t) = 1 − t^(−0.8) 도 사용",
    ], { x: 0.6, y: 5.35, w: 12.1, h: 1.45, size: 14 });
  }

  {
    const s = d.slide("Adafactor 코드와 검증 (실측)", { tag: "3교시 · Adafactor 코드" });
    d.code(s, [
      "R *= b2; R += (1-b2) * g2.sum(1)    # 행 합의 이동평균 (n개)",
      "C *= b2; C += (1-b2) * g2.sum(0)    # 열 합의 이동평균 (m개)",
      "vhat = np.outer(R, C) / R.sum()     # 두 벡터로 n×m 복원",
      "U = g / np.sqrt(vhat)               # 적응형 업데이트",
      "U /= max(1.0, rms(U) / d)           # RMS 클리핑",
      "p -= lr * U",
    ], { x: 0.6, y: 1.45, w: 7.7, h: 2.4, size: 12, title: "optim_np.Adafactor (핵심 6줄)" });
    d.explain(s, [
      "g2 = g² (아주 작은 수를 더해 0 방지)",
      "복원 결과는 ‘랭크 1’ 근사",
      "클리핑으로 한 걸음이 너무 커지는 것을 방지",
    ], { x: 8.5, y: 1.5, w: 4.2, h: 2.5, size: 13 });
    table(s, ["확인 항목 (step3_lion_adafactor.py, step6)", "결과"], [
      ["랭크 1 행렬 복원 오차", "2.2e-16 (정확히 복원)"],
      ["일반 랜덤 행렬의 복원", "근사 (평균 상대 오차 0.54): 정보가 일부 손실"],
      ["4096×4096 한 행렬의 2차 모멘트", "Adam 16,777,216개  vs  Adafactor 8,192개 (2048배 적음)"],
      ["torch.optim.Adafactor, 4,810 파라미터 모델의 상태", "276개  (AdamW 는 9,620개)"],
    ], { x: 0.6, y: 4.1, w: 12.1, colW: [6.0, 6.1], fontSize: 13, rowH: 0.4 });
    d.text(s, "우리 구현은 모멘텀과 상대 스텝 크기를 뺀 간단 버전이다 (torch.optim.Adafactor 와 값이 100% 같다고 주장하지 않는다).", { x: 0.6, y: 6.5, w: 12.1, h: 0.4, size: 12, color: C.muted });
  }

  {
    const s = d.slide("학습률 스케줄: 학습률은 고정하지 말고 시간에 따라 바꾸자", { tag: "3교시 · 스케줄" });
    d.card(s, { x: 0.6, y: 1.5, w: 5.9, h: 2.5, fill: C.greenL, title: "초반: 큰 학습률", tColor: C.green, size: 15, body: "빠르게 넓게 탐색. 하지만 너무 처음부터 크면 불안정 → 웜업(warm-up)으로 낮게 시작해 서서히 올린다." });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 2.5, fill: C.orangeL, title: "후반: 작은 학습률", tColor: C.orange, size: 15, body: "골짜기 바닥에서 미세 조정. 보폭이 크면 바닥 주변에서 계속 튄다." });
    d.bullets(s, [
      { t: "일반적 전략 ① Warmup + Decay: 낮게 시작 → 올리고 → 시간이 지나며 감소", b: true },
      "웜업이 필요한 이유(대표 설명): 학습 초반에는 v 같은 통계가 아직 부정확해 큰 학습률이 불안정을 부른다",
      { t: "일반적 전략 ② Cosine Annealing: 코사인 곡선으로 부드럽게 감소, 보통 웜업과 함께", b: true },
      "AdamW·Lion 은 코사인 스케줄과 함께 자주 쓰이고, Adafactor 는 파라미터 노름에 기반한 적응형(상대 스텝) 스케줄과 함께 쓰이는 경우가 많다",
    ], { x: 0.6, y: 4.3, w: 12.1, h: 2.5, size: 15 });
  }

  {
    const s = d.slide("코사인 감쇠: 식과 손계산", { tag: "3교시 · 코사인" });
    F(d, s, "η(t) = η_min + ½ (η_max − η_min) ( 1 + cos( π · t / T ) )", { x: 0.6, y: 1.45, w: 12.1, h: 0.8, size: 24, fill: C.tealL });
    table(s, ["t / T", "cos(π t/T)", "η(t) / η_max (η_min=0)", "선형 감소"], [
      ["0", "1", "½(1+1) = 1.000", "1.000"],
      ["0.25", "0.707", "½(1+0.707) = 0.854", "0.750"],
      ["0.5", "0", "½(1+0) = 0.500", "0.500"],
      ["0.75", "−0.707", "½(1−0.707) = 0.146", "0.250"],
      ["1", "−1", "½(1−1) = 0.000", "0.000"],
    ], { x: 0.6, y: 2.5, w: 7.3, colW: [1.2, 1.7, 2.9, 1.5], fontSize: 14, rowH: 0.5 });
    d.bullets(s, [
      "시작·끝에서 완만, 중간에서 빠르게 감소",
      "급격한 변화가 없어 학습이 안정적",
      "η_min 을 0 대신 작은 값(예: η_max의 1/10)으로 두기도 한다",
    ], { x: 8.2, y: 2.5, w: 4.5, h: 3.0, size: 15 });
    d.text(s, "그림 ①: 선형 감소와 비교 (다음 슬라이드). 코드: optim_np.cosine — assert 로 t=0, T/2, T 값을 검사.", { x: 0.6, y: 5.6, w: 12.1, h: 0.6, size: 14, color: C.muted });
  }

  {
    const s = d.slide("웜업 + 코사인, 그리고 재시작(SGDR)", { tag: "3교시 · 스케줄 그림" });
    img(s, "schedules", 1.45, 1.4, 10.4);
    d.bullets(s, [
      "② 웜업: 처음 100걸음은 0 에서 η_max 까지 직선 상승(첫 값 1e-5 = η_max/100) 후 코사인 감소",
      "③ SGDR (Warm Restarts): 주기가 끝나면 학습률을 다시 크게 올리고 코사인 감소를 반복. 재시작 직전 t=249 에는 3.9e-8, 250 에서 1e-3 으로 점프",
      "재시작은 지역 최소점에서 벗어날 기회를 주고 일반화가 좋아진 사례가 보고되었다 (항상은 아님)",
    ], { x: 0.6, y: 4.8, w: 12.1, h: 2.0, size: 14 });
  }

  {
    const s = d.slide("용어 정리: Cosine Decay vs Cosine Annealing", { tag: "3교시 · 용어 교정" });
    table(s, ["용어", "원본의 구분", "실제 사용 (PyTorch 등)"], [
      ["Cosine Decay", "'한 번만 감소'", "같은 코사인 곡선. 한 번 감소하는 것을 가리킴"],
      ["Cosine Annealing", "'보통 주기적 감소(restart 가능)'", "CosineAnnealingLR = 한 주기의 감소 (식이 같다). 재시작은 별도 CosineAnnealingWarmRestarts"],
      ["SGDR", "Warm Restart 를 제안한 방법", "Loshchilov & Hutter (2017). 코사인 + 주기적 재시작"],
    ], { x: 0.6, y: 1.5, w: 12.1, colW: [2.4, 4.1, 5.6], fontSize: 14, rowH: 0.7 });
    d.card(s, { x: 0.6, y: 4.6, w: 12.1, h: 2.1, fill: C.orangeL, title: "정리", tColor: C.orange, size: 15, body: "두 이름은 식이 똑같다. '감쇠 vs 어닐링'을 '단발 vs 주기'로 가르는 것은 표준적인 구분이 아니므로, 수업에서는 '코사인 스케줄 (+ 재시작 여부)'로 말하자.\nAnnealing(풀림, 담금질): 금속을 가열한 뒤 천천히 식혀 안정시키는 과정 → '처음엔 크게 탐색, 나중엔 안정적으로 수렴'." });
  }

  {
    const s = d.slide("스케줄 코드 (NumPy → PyTorch)", { tag: "3교시 · 코드" });
    d.code(s, [
      "def warmup_cosine(t, T, warm, lr_max, lr_min=0.0):",
      "    if t < warm:                        # 웜업: 직선 상승",
      "        return lr_max * (t + 1) / warm",
      "    return cosine(t - warm, T - warm, lr_max, lr_min)",
      "",
      "# PyTorch: 같은 모양을 LambdaLR 로",
      "f = lambda t: (t+1)/warm if t < warm else \\",
      "    0.5*(1 + math.cos(math.pi*(t-warm)/(T-warm)))",
      "sch = torch.optim.lr_scheduler.LambdaLR(opt, f)",
      "# 재시작은 내장 스케줄러",
      "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=250)",
      "for step in range(T): ...; opt.step(); sch.step()",
    ], { x: 0.6, y: 1.5, w: 8.3, h: 5.2, size: 12, title: "스케줄" });
    d.explain(s, [
      "t 는 '지금까지의 걸음 수'",
      "LambdaLR: 기준 학습률 × f(t)",
      "sch.step() 은 optimizer.step() 다음에 매 걸음(또는 매 에포크) 호출",
      "학습률 스케줄은 옵티마이저와 독립적인 레고 블록이다",
    ], { x: 9.1, y: 1.5, w: 3.6, h: 5.2, size: 13 });
  }

  {
    const s = d.slide("안정성과 일반화: Sharp Minima vs Flat Minima", { tag: "3교시 · 일반화" });
    img(s, "sharp_flat", 1.25, 1.4, 9.6);
    table(s, ["(장난감 1차원 손실, 실측)", "학습 손실", "시험 손실 (표면 0.25 이동)", "가중치 잡음(σ=0.2) 평균 손실", "곡률 L″"], [
      ["Sharp (θ=−1.5, 깊고 좁다)", "−1.001", "−0.008", "−0.372", "156.2"],
      ["Flat (θ=+1.5, 얕고 넓다)", "−0.800", "−0.762", "−0.776", "1.25"],
    ], { x: 0.6, y: 4.6, w: 12.1, colW: [3.6, 1.6, 2.6, 2.9, 1.4], fontSize: 12, rowH: 0.4 });
    d.text(s, "학습 손실은 sharp 가 더 낮지만, 데이터가 조금 달라지거나(표면 이동) 가중치에 잡음이 끼면 이득이 거의 사라진다. 곡률(Hessian 의 큰 고윳값 ↔ 2차 미분)이 큰 곳이 sharp.", { x: 0.6, y: 6.0, w: 12.1, h: 0.85, size: 13, color: C.muted });
  }

  {
    const s = d.slide("옵티마이저와 일반화: 경향이지 법칙은 아니다", { tag: "3교시 · 일반화 (주의)" });
    d.card(s, { x: 0.6, y: 1.5, w: 5.9, h: 2.5, fill: C.tealL, title: "원본의 설명", tColor: C.teal, size: 15, body: "· Adam/Lion 같은 적응형은 빠르게 수렴하지만 sharp minima 로 갈 수 있다\n· 모멘텀 SGD 는 느리지만 flat minima 를 찾아 테스트 정확도가 높다\n· 잡음이 큰 환경: AdamW 안정적 / 파인튜닝: SGD 로 전환하면 강건성 ↑" });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 2.5, fill: C.orangeL, title: "덧붙일 점 (보강)", tColor: C.orange, size: 15, body: "· 이는 '일부 연구에서 관찰된 경향'이다. 비전 모델의 SGD 우위 보고가 많지만 트랜스포머에서는 AdamW 가 표준\n· 'sharpness' 는 파라미터를 다시 스케일링하면 값이 바뀔 수 있어 일반화의 확정 지표가 아니다\n· 학습률·배치 크기·정규화·스케줄이 더 큰 영향을 줄 수 있다" });
    d.bullets(s, [
      "결론: 'Adam 은 나쁘고 SGD 는 좋다'가 아니라, 일반화 성능은 옵티마이저 + 하이퍼파라미터 + 정규화 + 데이터의 합이다",
      "실무 팁(원본): 모델의 부분마다 다른 옵티마이저를 쓸 수 있다 — 예: 임베딩 AdamW, 분류 헤드 SGD (PyTorch 에서는 옵티마이저를 둘 만들어 각각 step())",
    ], { x: 0.6, y: 4.3, w: 12.1, h: 2.5, size: 16 });
    s.addNotes("원본 검토 항목: 단정적 서술을 '경향'으로 완화. 학생이 '옵티마이저만 바꾸면 일반화가 해결된다'고 오해하지 않도록 한다.");
  }

  {
    const s = d.slide("특징 비교 요약 (표 14.1, 보강판)", { tag: "3교시 · 비교표" });
    table(s, ["옵티마이저", "메모리\n(추가 상태)", "적응형 학습률", "모멘텀", "사용 사례"], [
      ["SGD", "Low (0~1배)", "No", "Yes (선택)", "노이즈가 적은 환경, 파인튜닝, 강건성"],
      ["AdamW", "Medium (2배)", "Yes (v)", "Yes (m)", "대부분의 모델 · 트랜스포머의 기본값"],
      ["Lion", "Low (1배)", "No (부호만; 보폭 균일)", "Sign", "ViT, 메모리가 부족한 환경"],
      ["Adafactor", "Very Low (n+m)", "Yes (분해된 v)", "No (기본값)", "초대형 언어 모델 (T5 등)"],
    ], { x: 0.6, y: 1.5, w: 12.1, colW: [2.0, 2.3, 2.6, 1.8, 3.4], fontSize: 14, rowH: 0.62 });
    d.text(s, "[표 14.1] 현대 최적화 방법 정리. '메모리'는 파라미터 수 대비 추가 상태의 크기, 'Lion 의 적응형'은 부호 때문에 파라미터별 크기 보정이 없다는 뜻(원본 표 그대로 No).", { x: 0.6, y: 5.1, w: 12.1, h: 0.8, size: 14, color: C.muted });
    d.text(s, "Adafactor 의 모멘텀 'No' 는 기본 설정이다 (β1 을 켤 수 있지만 그러면 메모리 이점이 줄어든다).", { x: 0.6, y: 5.95, w: 12.1, h: 0.6, size: 14, color: C.muted });
  }

  {
    const s = d.slide("PyTorch 종합 실험 (실측, 5 seeds)", { tag: "3교시 · 종합 실험" });
    img(s, "compare", 0.6, 1.4, 8.0);
    table(s, ["옵티마이저", "학습률", "시험 정확도", "상태 수"], [
      ["SGD+모멘텀", "0.05", "97.1 ± 0.6%", "4,810"],
      ["AdamW", "3e-3", "97.6 ± 0.6%", "9,620"],
      ["Lion", "1e-3", "97.1 ± 0.3%", "4,810"],
      ["Adafactor", "1e-2", "96.8 ± 0.7%", "276"],
    ], { x: 8.8, y: 1.5, w: 3.9, colW: [1.2, 0.7, 1.2, 0.8], fontSize: 11, rowH: 0.5 });
    d.bullets(s, [
      "같은 모델(64→64→10), 같은 데이터, 40 에포크. 각 옵티마이저에 맞는 학습률을 따로 사용",
      "정확도 차이는 표준편차(±0.3~0.7)와 비슷하다 → 이 작은 문제에서는 '누가 더 낫다'고 말할 수 없다",
      "학습 손실은 Lion 이 가장 낮고 Adafactor 가 가장 높다 → 학습 손실이 낮다고 시험 성능이 높은 것은 아니다 (과적합과 일반화)",
      "Adafactor 는 메모리를 거의 쓰지 않으면서도 비슷한 정확도를 낸다",
    ], { x: 0.6, y: 4.35, w: 12.1, h: 2.45, size: 14 });
    s.addNotes("정직한 결론: 이 장난감 실험은 알고리즘의 '작동'을 확인하는 용도이며 순위를 가리는 용도가 아니다. 대규모 모델에서의 결과는 논문들을 참고하도록 안내한다.");
  }

  {
    const s = d.slide("스케줄 실험과 파라미터 그룹 (실측)", { tag: "3교시 · 실험" });
    table(s, ["AdamW (lr 1e-2) 스케줄", "학습 손실 (40 ep)", "시험 정확도 (5 seeds)"], [
      ["고정 학습률", "0.0054", "97.3 ± 0.4%"],
      ["웜업 + 코사인", "0.0152", "97.5 ± 0.3%"],
      ["코사인 재시작 (SGDR, 주기 10 에포크)", "0.0147", "97.3 ± 0.2%"],
    ], { x: 0.6, y: 1.5, w: 12.1, colW: [5.1, 3.2, 3.8], fontSize: 14, rowH: 0.5 });
    d.text(s, "학습 손실은 고정 학습률이 가장 낮지만(과하게 외움), 시험 정확도는 스케줄을 쓴 쪽이 근소하게 높거나 비슷하다 — 차이는 오차 범위 안이다.", { x: 0.6, y: 3.7, w: 12.1, h: 0.7, size: 15 });
    d.code(s, [
      "# 파라미터 그룹: 층마다 다른 옵티마이저 (원본의 '실용적 팁')",
      "opts = [torch.optim.AdamW(weights, 3e-3, weight_decay=0.01),   # 가중치 → AdamW",
      "        torch.optim.SGD(biases, 0.05)]                         # 편향 → SGD",
      "loss.backward()",
      "for o in opts: o.step()          # 둘 다 step() 하면 된다",
    ], { x: 0.6, y: 4.6, w: 12.1, h: 2.2, size: 12, title: "step6_torch_compare.py" });
  }

  {
    const s = d.slide("실용적 선택 가이드", { tag: "3교시 · 실용 팁" });
    d.card(s, { x: 0.6, y: 1.5, w: 3.9, h: 3.0, fill: C.tealL, title: "잡음·불안정 → AdamW", tColor: C.teal, size: 14, body: "노이즈가 많은 데이터, 기울기가 불안정할 때 안전하고 효과적. 모르겠으면 AdamW + 웜업 + 코사인에서 시작." });
    d.card(s, { x: 4.7, y: 1.5, w: 3.9, h: 3.0, fill: C.purpleL, title: "자원 제한·ViT → Lion", tColor: C.purple, size: 14, body: "GPU 메모리가 빠듯하거나 비전 트랜스포머를 학습할 때 처리량 이점. 학습률은 AdamW 의 1/3~1/10, 감쇠는 더 크게." });
    d.card(s, { x: 8.8, y: 1.5, w: 3.9, h: 3.0, fill: C.greenL, title: "초대형 모델 → Adafactor", tColor: C.green, size: 14, body: "수십억 파라미터에서는 메모리 효율이 결정적. (원본: '사실상 필수'는 과장 — 큰 모델도 AdamW + 상태 분산 저장으로 학습한다.)" });
    d.bullets(s, [
      "SGD(+모멘텀): 비전 모델, 잡음이 적은 환경, 파인튜닝 단계에서 여전히 강력한 기본기",
      "학습률 스케줄은 옵티마이저와 독립적으로 선택한다 (수렴 속도와 최종 성능에 큰 영향)",
      "항상: 학습률부터 튜닝 → 그 다음 감쇠·β 순서로. 바꾸기 전에 기준 실험(baseline)을 먼저 저장!",
    ], { x: 0.6, y: 4.8, w: 12.1, h: 2.0, size: 15 });
  }

  {
    const s = d.slide("요약", { tag: "3교시 · 마무리" });
    d.bullets(s, [
      "현대의 옵티마이저는 모멘텀 · 적응형 학습률 · 메모리 효율 · 정규화를 통합한다",
      { t: "AdamW: 안정성과 성능의 균형 → 가장 널리 쓰이는 기본 선택", b: true },
      { t: "Lion: 부호만 쓰는 가벼운 모멘텀 (상태 1개) → 자원 제한 환경", b: true },
      { t: "Adafactor: 2차 모멘트를 행·열로 분해 (n×m → n+m) → 초대형 모델", b: true },
      "SGD: 기본이지만 노이즈가 적은 환경·파인튜닝에서 여전히 효과적",
      "학습률 스케줄(웜업 + 코사인)은 옵티마이저만큼 중요하다",
      "옵티마이저는 단순한 수학 기법이 아니라 '조향 장치' — 지형(문제), 차량(모델), 목적지(목표)에 맞춰 고른다",
    ], { x: 0.6, y: 1.5, w: 12.1, h: 5.2, size: 18 });
    s.addNotes("마지막으로 원본의 비유를 다시 말한다. 다음 시간(Lesson 15)은 설명 가능성(Grad-CAM, SHAP, 어텐션)이다.");
  }
};
