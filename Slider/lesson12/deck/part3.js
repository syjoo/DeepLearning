// 3교시: 정규화 (내부 공변량 변화, 미니배치, BatchNorm, LayerNorm, 종합 실험)
const { C, FONT } = require("./lib");
const { img, F } = require("./helpers");

module.exports = function part3(d, pres) {
  d.section("3교시", "정규화 (Normalization)", "내부 공변량 변화 · 미니배치 · Batch Norm · Layer Norm", "3교시 시작. 초기화는 출발만 안정시킨다. 학습 도중의 불안정을 정규화가 막아 준다.");

  {
    const s = d.slide("정규화를 하는 이유", { tag: "3교시 · 정규화" });
    d.text(s, "학습 중 활성값은 신경망을 따라 전파된다. 네트워크가 깊을수록 활성값의 분포는 층(layer) 간 또는 배치(batch) 간에 크게 달라질 수 있다.", { x: 0.7, y: 1.5, w: 12, h: 0.9, size: 18 });
    d.card(s, { x: 0.7, y: 2.6, w: 5.9, h: 2.2, fill: C.redL, title: "이 현상 = 내부 공변량 변화", tColor: C.red, size: 15, body: "(Internal Covariate Shift)\n학습을 불안정하게 만들고 기울기의 크기가 크게 흔들리게 한다." });
    d.card(s, { x: 6.8, y: 2.6, w: 5.9, h: 2.2, fill: C.greenL, title: "해결: 정규화 기법", tColor: C.green, size: 15, body: "Batch Normalization (배치 정규화)\nLayer Normalization (레이어 정규화)\n→ 층 입력의 분포를 안정시킨다." });
    d.card(s, { x: 0.7, y: 5.0, w: 12.0, h: 1.7, fill: C.paper, title: "한 줄 요약", size: 16, body: "정규화 = 각 층으로 들어오는 값의 평균을 0 근처로, 분산을 1 근처로 유지하여 '입력 분포의 변화 폭'을 줄여 주는 것" });
    s.addNotes("원본의 '개요'와 '정규화를 하는 이유' 두 단락은 같은 문장이 중복되어 있어 하나로 합쳤다.");
  }

  {
    const s = d.slide("내부 공변량(Internal Covariate)이란?", { tag: "3교시 · 정규화" });
    d.text(s, "공변량(covariate) = 모델의 입력으로 사용되는 변수.  신경망 '안'에서는 이전 층의 출력이 다음 층의 입력이다.", { x: 0.7, y: 1.5, w: 12, h: 0.8, size: 18 });
    d.box(s, "데이터 x", { x: 0.7, y: 2.7, w: 1.9, h: 0.8, fill: C.orangeL, size: 17, bold: true });
    d.box(s, "1층", { x: 3.3, y: 2.7, w: 1.4, h: 0.8, fill: C.tealL, size: 17, bold: true });
    d.box(s, "h₁", { x: 5.4, y: 2.7, w: 1.2, h: 0.8, fill: C.purpleL, size: 18, bold: true });
    d.box(s, "2층", { x: 7.3, y: 2.7, w: 1.4, h: 0.8, fill: C.tealL, size: 17, bold: true });
    d.box(s, "h₂", { x: 9.4, y: 2.7, w: 1.2, h: 0.8, fill: C.purpleL, size: 18, bold: true });
    d.box(s, "3층 …", { x: 11.2, y: 2.7, w: 1.5, h: 0.8, fill: C.tealL, size: 17, bold: true });
    [[2.6, 3.3], [4.7, 5.4], [6.6, 7.3], [8.7, 9.4], [10.6, 11.2]].forEach(([a, b]) => d.arrow(s, a, 3.1, b, 3.1));
    d.bullets(s, ["데이터 x → 첫 번째 층의 입력", "첫 번째 층의 출력 h₁ → 두 번째 층의 '내부 공변량'", "두 번째 층의 출력 h₂ → 세 번째 층의 '내부 공변량'", "즉, 네트워크 내부에서 각 층이 입력으로 받는 값들을 내부 공변량이라고 부른다"], { x: 0.7, y: 4.0, w: 12, h: 2.7, size: 18 });
    s.addNotes("원본 제목의 'Internal Covariants'는 'Covariates'의 오타. 용어: 공변량=입력 변수.");
  }

  {
    const s = d.slide("내부 공변량 변화(Internal Covariate Shift)와 문제점", { tag: "3교시 · 정규화" });
    d.card(s, { x: 0.6, y: 1.5, w: 6.0, h: 3.0, fill: C.tealL, title: "정의", tColor: C.teal, size: 15, body: "학습이 진행되면서 앞쪽 층의 파라미터(가중치)가 업데이트되면, 그 결과로 뒤쪽 층이 받는 입력 분포가 계속 변하는 현상.\n\n한 층이 어떤 입력 분포(평균·분산)에 맞춰 학습 중인데 → 이전 층 가중치가 바뀌면서 → 같은 층의 입력 분포가 매 iteration마다 달라진다." });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 3.0, fill: C.redL, title: "왜 문제인가", tColor: C.red, size: 15, body: "· 각 층이 계속 바뀌는 분포에 적응해야 해서 학습이 느려짐\n· 활성값이 너무 커지거나 작아져 기울기 폭주/소실 가능\n· 학습률을 크게 쓰기 어려워짐\n· 초기화에 민감해짐\n→ 결과적으로 학습이 불안정해진다" });
    d.card(s, { x: 0.6, y: 4.7, w: 12.1, h: 1.95, fill: C.greenL, title: "정규화가 하는 일", tColor: C.green, size: 16, body: "각 층으로 들어오는 내부 공변량의 평균을 0 근처로, 분산을 1 근처로 유지하도록 만들어 입력 분포의 변화 폭을 줄여 준다.\n비유: 매번 눈높이가 바뀌는 교실보다, 책상 높이가 일정한 교실에서 공부가 잘 된다." });
    s.addNotes("비유: 시험 범위가 매일 바뀌면 공부가 안 되는 것과 같다. 학생이 앞 학생(이전 층)의 숙제 결과에 의존하는 릴레이로 설명해도 좋다.");
  }

  {
    const s = d.slide("[최신 자료] 정말로 'Internal Covariate Shift' 때문일까?", { tag: "3교시 · 최신 연구" });
    d.bullets(s, [
      { t: "원논문(Ioffe & Szegedy, 2015)의 설명: BN이 내부 공변량 변화를 줄여서 학습이 빨라진다", b: true },
      "후속 연구(Santurkar 외, 2018 'How Does Batch Normalization Help Optimization?')는 BN의 진짜 효과가 ICS 감소보다 손실 지형(loss landscape)을 매끄럽게 만들어 기울기를 예측 가능하게 하는 데 있다고 보고",
      "즉, '분포가 안정되어서'보다 '최적화가 쉬운 지형이 되어서'가 더 설득력 있는 설명",
      "그래도 실용적 결론은 같다: 정규화를 쓰면 학습이 빠르고 안정적이며 큰 학습률을 쓸 수 있다",
    ], { x: 0.7, y: 1.5, w: 12, h: 3.7, size: 18 });
    d.card(s, { x: 0.7, y: 5.4, w: 12, h: 1.3, fill: C.orangeL, title: "강의 방침", tColor: C.orange, size: 15, body: "교재의 ICS 설명은 직관을 얻는 데 유용하므로 그대로 배우되, 이 개념이 '유일한 정답'은 아니라는 점을 알고 있자." });
    s.addNotes("교재 내용과 달라진 최신 연구 반영. 학생들에게 '교과서 설명도 계속 수정된다'는 연구 태도를 전한다. 2015 원논문과 2018 후속 논문 이름만 소개.");
  }

  {
    const s = d.slide("미니배치(mini-batch)란?", { tag: "3교시 · 미니배치" });
    d.text(s, "전체 학습 데이터 중에서 한 번의 파라미터 업데이트에 사용되는 작은 데이터 묶음", { x: 0.7, y: 1.45, w: 12, h: 0.5, size: 19, bold: true });
    d.card(s, { x: 0.6, y: 2.2, w: 3.9, h: 3.3, fill: C.tealL, title: "① 배치 학습 (Batch)", tColor: C.teal, size: 14, body: "전체 데이터셋을 한 번에 사용\n\n장점: 기울기가 정확함\n단점: 메모리 부담 큼, 느림" });
    d.card(s, { x: 4.7, y: 2.2, w: 3.9, h: 3.3, fill: C.purpleL, title: "② 확률적 경사하강법 (SGD)", tColor: C.purple, size: 14, body: "데이터 1개씩 사용\n\n장점: 빠른 업데이트\n단점: 기울기 노이즈 큼, 불안정" });
    d.card(s, { x: 8.8, y: 2.2, w: 3.9, h: 3.3, fill: C.greenL, title: "③ 미니배치 ★ 가장 많이 사용", tColor: C.green, size: 14, body: "일부(예: 32, 64, 128개)만 사용\n\n속도·안정성·메모리의 균형이 좋음" });
    d.text(s, "데이터셋 크기: N,  미니배치 크기: m  →  한 스텝: {(x₁,y₁), …, (x_m,y_m)} 으로 손실 계산 → 기울기 계산 → 가중치 업데이트", { x: 0.6, y: 5.8, w: 12.1, h: 0.8, size: 16 });
    s.addNotes("용어 주의: 일상에서 'batch'는 보통 미니배치를 가리킨다. 'batch size = 64'는 미니배치 크기. 비유: 시험 문제집을 한 번에 다 풀기(배치) / 한 문제씩 채점(SGD) / 한 쪽씩 풀고 채점(미니배치).");
  }

  {
    const s = d.slide("Batch Normalization (배치 정규화) ①: 수식", { tag: "3교시 · BatchNorm" });
    d.text(s, "특정 뉴런(또는 채널)의 활성값 미니배치 {x₁, x₂, …, x_m} 을 '배치 차원 기준'으로 정규화", { x: 0.6, y: 1.45, w: 12.1, h: 0.45, size: 17 });
    F(d, s, "μ_B = (1/m) Σᵢ xᵢ", { x: 0.6, y: 2.05, w: 6.5, h: 0.65, size: 21 }); d.text(s, "(12.1) 미니배치 평균", { x: 7.3, y: 2.2, w: 5, h: 0.4, size: 15, color: C.muted });
    F(d, s, "σ²_B = (1/m) Σᵢ (xᵢ − μ_B)²", { x: 0.6, y: 2.85, w: 6.5, h: 0.65, size: 21 }); d.text(s, "(12.2) 미니배치 분산", { x: 7.3, y: 3.0, w: 5, h: 0.4, size: 15, color: C.muted });
    F(d, s, "x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε)", { x: 0.6, y: 3.65, w: 6.5, h: 0.65, size: 21, fill: C.tealL }); d.text(s, "(12.3) 정규화: 평균 0, 분산 1", { x: 7.3, y: 3.8, w: 5.4, h: 0.4, size: 15, color: C.muted });
    F(d, s, "yᵢ = γ·x̂ᵢ + β", { x: 0.6, y: 4.45, w: 6.5, h: 0.65, size: 21, fill: C.orangeL }); d.text(s, "(12.4) 배율(γ)·이동(β)으로 복원", { x: 7.3, y: 4.6, w: 5.4, h: 0.4, size: 15, color: C.muted });
    d.bullets(s, ["m : 미니배치 안의 샘플 수.  μ_B, σ²_B : 그 미니배치에서 계산한 평균과 분산", "ε(엡실론) : 0으로 나누는 것을 막는 아주 작은 수 (예: 10⁻⁵)", "γ, β : 학습되는 파라미터 (초기값 γ=1, β=0)"], { x: 0.6, y: 5.35, w: 12.1, h: 1.4, size: 15 });
    s.addNotes("네 개의 식을 위에서 아래로 한 줄씩 읽는다: 평균 구하기 -> 퍼진 정도 구하기 -> 평균 빼고 표준편차로 나누기 -> 다시 키우고 옮기기.");
  }

  {
    const s = d.slide("BN 손계산: 값 [2, 4, 6, 8] 한 특징, 배치 크기 4", { tag: "3교시 · BatchNorm" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const ctr = (t) => ({ text: t, options: { align: "center" } });
    s.addTable([[H("단계"), H("계산"), H("결과")], [ctr("① 평균"), ctr("(2+4+6+8)/4"), ctr("μ = 5")], [ctr("② 분산"), ctr("((−3)²+(−1)²+1²+3²)/4 = 20/4"), ctr("σ² = 5")], [ctr("③ 표준편차"), ctr("√5"), ctr("≈ 2.236")], [ctr("④ 정규화"), ctr("(x − 5)/2.236"), ctr("−1.342, −0.447, 0.447, 1.342")]], { x: 0.6, y: 1.55, w: 12.1, colW: [2.0, 5.0, 5.1], fontFace: FONT, fontSize: 16, color: C.ink, rowH: 0.6, valign: "middle", border: { type: "solid", color: C.line, pt: 1 } });
    d.card(s, { x: 0.6, y: 4.8, w: 5.9, h: 1.9, fill: C.greenL, title: "확인", tColor: C.green, size: 15, body: "정규화된 값의 평균 = 0\n정규화된 값의 분산 = 1\n(코드 step3_norm.py 첫 줄 출력과 동일)" });
    d.card(s, { x: 6.8, y: 4.8, w: 5.9, h: 1.9, fill: C.orangeL, title: "γ=2, β=1 을 적용하면", tColor: C.orange, size: 15, body: "y = 2·x̂ + 1\n= −1.68, 0.11, 1.89, 3.68\n평균 1, 표준편차 2 로 '되돌린' 분포" });
    s.addNotes("학생이 직접 계산기로 해 보게 한다. 모집단 분산(m으로 나눔)을 쓴다는 점 주의: 학습 때 BN은 1/m 분산을 쓴다.");
  }

  {
    const s = d.slide("γ와 β는 왜 필요한가?", { tag: "3교시 · BatchNorm" });
    img(s, "bn_effect", 2.1, 1.4, 9.0);
    d.bullets(s, [
      "정규화만 하면 모든 값이 평균 0, 분산 1 이 되어 표현력이 제한된다",
      "그래서 모델이 필요하면 다시 스케일을 조절할 수 있도록 γ(배율)와 β(이동)를 '학습 가능한 파라미터'로 둔다",
      "BN은 학습 안정성을 위해 정규화하고, γ·β는 표현력 회복을 위해 재조정한다 → 이 두 가지가 합쳐져 BatchNorm이 강력해진다",
    ], { x: 0.6, y: 4.6, w: 12.1, h: 2.1, size: 16 });
    s.addNotes("특히 γ와 β를 학습하면 모델이 '정규화를 되돌리는 것'까지 스스로 선택할 수 있다는 점을 강조.");
  }

  {
    const s = d.slide("BN을 어디에 넣을까? 그리고 미니배치 크기", { tag: "3교시 · BatchNorm" });
    d.text(s, "일반적으로 선형 변환 이후, 비선형 활성화 이전에 삽입한다:", { x: 0.7, y: 1.5, w: 12, h: 0.45, size: 18 });
    ["Linear", "BatchNorm", "ReLU"].forEach((t, i) => d.box(s, t, { x: 1.2 + i * 3.8, y: 2.1, w: 2.6, h: 0.85, fill: [C.tealL, C.orangeL, C.purpleL][i], size: 20, bold: true }));
    d.arrow(s, 3.8, 2.5, 5.0, 2.5); d.arrow(s, 7.6, 2.5, 8.8, 2.5);
    d.card(s, { x: 0.7, y: 3.3, w: 5.9, h: 3.3, fill: C.greenL, title: "효과", tColor: C.green, size: 15, body: "기울기 흐름을 개선\n더 빠른 학습, 더 높은 학습률 사용 가능\n(초기화에도 덜 민감 → 실험에서 확인)" });
    d.card(s, { x: 6.8, y: 3.3, w: 5.9, h: 3.3, fill: C.redL, title: "BN과 미니배치 크기", tColor: C.red, size: 15, body: "BN은 미니배치 내부의 평균·분산을 사용한다.\n· 미니배치가 너무 작으면 통계량이 불안정\n· 미니배치가 1이면 BN이 의미 없음\n→ 작은 배치에서는 LayerNorm, GroupNorm이 선호된다" });
    s.addNotes("실전에서는 'Conv/Linear -> BN -> ReLU' 순서가 가장 흔하다. 순서를 바꾼 변형도 있으나 입문에서는 이 순서를 외운다.");
  }

  {
    const s = d.slide("[보강] 학습할 때와 추론할 때의 BN은 다르다", { tag: "3교시 · BatchNorm" });
    d.card(s, { x: 0.6, y: 1.5, w: 5.9, h: 2.6, fill: C.tealL, title: "학습(training) 모드", tColor: C.teal, size: 15, body: "현재 미니배치의 μ_B, σ²_B 사용\n+ 이동 평균(running mean/var)을 조금씩 갱신\nrun = 0.9·run + 0.1·(이번 배치 통계)" });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 2.6, fill: C.orangeL, title: "추론(eval) 모드", tColor: C.orange, size: 15, body: "배치가 없거나 1개여도 동작해야 한다\n→ 학습 중 모아 둔 이동 평균 통계를 고정해서 사용\n(PyTorch: model.eval())" });
    d.bullets(s, ["원본 자료에는 빠져 있지만 실전에서 반드시 알아야 하는 내용", "model.eval()을 잊으면 추론 결과가 배치 구성에 따라 흔들리는 대표적인 버그가 생긴다", "LayerNorm은 샘플 하나 안에서 계산하므로 학습/추론 구분이 없다"], { x: 0.6, y: 4.4, w: 12.1, h: 2.3, size: 17 });
    s.addNotes("원본 누락 보충. 코드에서 BatchNorm 클래스의 training 인자와 run_mean/run_var가 이것을 구현한다.");
  }

  {
    const s = d.slide("직접 구현: BatchNorm (NumPy)", { tag: "3교시 · 코드 Step 3-①" });
    d.code(s, [
      "class BatchNorm:",
      "    def __init__(self, dim, momentum=0.1):",
      "        self.gamma = np.ones(dim)     # 배율 γ (학습됨)",
      "        self.beta  = np.zeros(dim)    # 이동 β (학습됨)",
      "        self.run_mean = np.zeros(dim) # 추론용 이동 평균",
      "        self.run_var  = np.ones(dim)",
      "        self.momentum = momentum",
      "    def __call__(self, x, training=True):",
      "        if training:",
      "            mu  = x.mean(axis=0)      # (12.1) 배치 방향 평균",
      "            var = x.var(axis=0)       # (12.2) 배치 방향 분산",
      "            # 이동 평균 갱신 (추론에서 사용)",
      "            self.run_mean = (1-self.momentum)*self.run_mean + self.momentum*mu",
      "            self.run_var  = (1-self.momentum)*self.run_var  + self.momentum*var",
      "        else:",
      "            mu, var = self.run_mean, self.run_var",
      "        x_hat = (x - mu) / np.sqrt(var + EPS)   # (12.3)",
      "        return self.gamma * x_hat + self.beta   # (12.4)",
    ], { x: 0.4, y: 1.4, w: 8.4, h: 5.3, size: 11, title: "step3_norm.py" });
    d.explain(s, ["x 의 모양: (배치, 특징). axis=0 은 '배치 방향' → 특징마다 따로 평균/분산", "γ=1, β=0 으로 시작: 처음에는 순수 정규화", "training=True : 이번 배치 통계 사용 + 이동 평균 갱신", "training=False : 저장된 통계 사용 (배치 1개도 OK)", "EPS=1e-5 : 0으로 나누기 방지"], { x: 9.0, y: 1.4, w: 3.8, h: 5.3, size: 12 });
    s.addNotes("axis=0이 핵심. '세로로(배치 방향) 평균'을 낸다. 다음에 나오는 LayerNorm은 axis=-1.");
  }

  {
    const s = d.slide("Layer Normalization (레이어 정규화)", { tag: "3교시 · LayerNorm" });
    d.text(s, "작은 배치를 쓰거나 시퀀스 모델(NLP, Transformer)에서는 배치 통계가 불안정 → '샘플 하나 안'에서 특징(feature) 차원으로 정규화", { x: 0.6, y: 1.45, w: 12.1, h: 0.8, size: 17 });
    F(d, s, "μ = (1/d) Σᵢ xᵢ", { x: 0.6, y: 2.4, w: 5.8, h: 0.6, size: 20 }); d.text(s, "(12.5)", { x: 6.6, y: 2.5, w: 1.5, h: 0.4, size: 14, color: C.muted });
    F(d, s, "σ² = (1/d) Σᵢ (xᵢ − μ)²", { x: 0.6, y: 3.15, w: 5.8, h: 0.6, size: 20 }); d.text(s, "(12.6)", { x: 6.6, y: 3.25, w: 1.5, h: 0.4, size: 14, color: C.muted });
    F(d, s, "x̂ᵢ = (xᵢ − μ) / √(σ² + ε)", { x: 0.6, y: 3.9, w: 5.8, h: 0.6, size: 20, fill: C.tealL }); d.text(s, "(12.7)", { x: 6.6, y: 4.0, w: 1.5, h: 0.4, size: 14, color: C.muted });
    F(d, s, "yᵢ = γ·x̂ᵢ + β", { x: 0.6, y: 4.65, w: 5.8, h: 0.6, size: 20, fill: C.orangeL }); d.text(s, "(12.8)", { x: 6.6, y: 4.75, w: 1.5, h: 0.4, size: 14, color: C.muted });
    d.card(s, { x: 7.8, y: 2.4, w: 4.9, h: 2.85, fill: C.redL, title: "원본 자료 정정", tColor: C.red, size: 13, body: "원본의 식 (12.7)은 μ_B, σ²_B(배치 통계 기호)를 썼다. 레이어 정규화는 샘플 하나의 μ, σ²를 쓰므로 위처럼 바로잡았다." });
    d.bullets(s, ["d : 샘플 하나의 특징 개수 (벡터 x ∈ ℝᵈ)", "배치 크기와 무관 → 배치 1개도 OK, 학습/추론이 같다", "RNN, Transformer 같은 순차 모델에 특히 효과적"], { x: 0.6, y: 5.5, w: 12.1, h: 1.3, size: 15 });
    s.addNotes("식 12.5~12.8은 12.1~12.4와 같은 모양인데 '평균을 내는 방향'만 다르다. 원본 식 12.7의 아래첨자 B는 오기.");
  }

  {
    const s = d.slide("Batch Norm vs Layer Norm: 어느 방향으로 평균을 낼까?", { tag: "3교시 · 비교" });
    img(s, "bn_ln_table", 0.5, 1.45, 7.2);
    d.text(s, "[그림 12.1 변형] 표 한 장 = 샘플(N) × 특징(C). 파란 부분이 한 번에 평균·분산을 구하는 묶음.", { x: 0.5, y: 4.4, w: 7.2, h: 0.55, size: 12, color: C.muted });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t, b) => ({ text: t, options: { align: "left", bold: !!b } });
    s.addTable([[H(""), H("BatchNorm"), H("LayerNorm")], [c("정규화 기준", 1), c("배치 방향(N)"), c("특징 방향(C,H,W)")], [c("통계 계산", 1), c("특징마다 따로"), c("샘플마다 따로")], [c("배치 크기 의존", 1), c("크다(작으면 불안정)"), c("없다")], [c("학습/추론", 1), c("다름(이동평균)"), c("같음")], [c("주 사용처", 1), c("CNN(이미지)"), c("Transformer, RNN, NLP")]], { x: 7.9, y: 1.5, w: 4.9, colW: [1.5, 1.7, 1.7], fontFace: FONT, fontSize: 12, color: C.ink, rowH: 0.6, valign: "middle", border: { type: "solid", color: C.line, pt: 1 } });
    d.card(s, { x: 0.5, y: 5.1, w: 7.2, h: 1.65, fill: C.orangeL, title: "한 줄 기억법", tColor: C.orange, size: 15, body: "BatchNorm = 같은 특징을 '여러 샘플에 걸쳐' 맞춘다 (세로줄)\nLayerNorm = 한 샘플을 '자기 특징들끼리' 맞춘다 (가로줄)" });
    s.addNotes("그림 12.1(원본)은 3차원 큐브로 같은 개념을 보여 준다. 핵심은 파란 부분이 세로냐 가로냐.");
  }

  {
    const s = d.slide("직접 구현: LayerNorm + PyTorch로 검증", { tag: "3교시 · 코드 Step 3-②" });
    d.code(s, [
      "class LayerNorm:",
      "    def __init__(self, dim):",
      "        self.gamma = np.ones(dim); self.beta = np.zeros(dim)",
      "    def __call__(self, x):",
      "        mu  = x.mean(axis=-1, keepdims=True)  # (12.5) 샘플별 평균",
      "        var = x.var(axis=-1, keepdims=True)   # (12.6) 샘플별 분산",
      "        x_hat = (x - mu) / np.sqrt(var + EPS) # (12.7)",
      "        return self.gamma * x_hat + self.beta # (12.8)",
      "",
      "# --- PyTorch 내장 구현과 비교 ---",
      "ref_b = torch.nn.BatchNorm1d(4, eps=EPS)(t)",
      "ref_l = torch.nn.LayerNorm(4, eps=EPS)(t)",
      "# 출력: BN 일치: True   LN 일치: True",
    ], { x: 0.4, y: 1.4, w: 8.2, h: 4.5, size: 12, title: "step3_norm.py" });
    d.explain(s, ["axis=-1 : 마지막 축 = '특징 방향' → 샘플 하나 안에서 계산", "keepdims=True : 모양을 유지해야 (x - mu) 브로드캐스팅이 맞음", "BatchNorm과 거의 같은 코드, 평균 내는 방향만 다르다!", "내장 구현과 결과가 일치 → 구현이 맞다는 증거"], { x: 8.8, y: 1.4, w: 4.0, h: 4.5, size: 12 });
    d.card(s, { x: 0.4, y: 6.05, w: 12.4, h: 0.7, fill: C.tealL, title: "", size: 14, body: "출력 확인: BN 출력의 열별 평균 ≈ 0, 표준편차 ≈ 1 / LN 출력의 행별 평균 ≈ 0, 표준편차 ≈ 1" });
    s.addNotes("두 클래스를 나란히 보여 주며 axis만 다름을 강조. 마지막에 python step3_norm.py를 실행해 True/True를 직접 확인.");
  }

  {
    const s = d.slide("종합 실험: 초기화와 정규화가 학습에 주는 효과", { tag: "3교시 · 코드 Step 4" });
    d.code(s, [
      "# 8층 MLP, 나선형 3-클래스 분류, SGD(momentum 0.9)",
      "def build(init, norm, depth=8, width=64):",
      "    for _ in range(depth):",
      "        lin = nn.Linear(d_in, width)",
      "        if init == 'small':  nn.init.normal_(lin.weight, 0, 0.01)",
      "        elif init == 'xavier': nn.init.xavier_normal_(lin.weight)",
      "        else: nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')",
      "        layers.append(lin)            # Linear -> (Norm) -> ReLU",
      "        if norm == 'bn': layers.append(nn.BatchNorm1d(width))",
      "        if norm == 'ln': layers.append(nn.LayerNorm(width))",
      "        layers.append(nn.ReLU())",
    ], { x: 0.4, y: 1.4, w: 8.4, h: 4.2, size: 11, title: "step4_compare.py" });
    d.explain(s, ["같은 모델·같은 데이터에서 초기화(small/Xavier/He)와 정규화(없음/BN/LN)만 바꿔 비교", "순서: Linear → Norm → ReLU", "학습 손실 곡선과 최종 정확도를 비교"], { x: 9.0, y: 1.4, w: 3.8, h: 4.2, size: 12 });
    d.card(s, { x: 0.4, y: 5.8, w: 12.4, h: 0.95, fill: C.orangeL, title: "", size: 15, body: "예측해 보자: 6가지 설정 중 가장 빨리 배우는 것은? 아예 못 배우는 것은? (다음 장에서 확인)" });
    s.addNotes("학생들이 먼저 예측하게 한 뒤 결과를 보여 준다. 실행: python step4_compare.py (CPU로 1~2분).");
  }

  {
    const s = d.slide("종합 실험 결과", { tag: "3교시 · 실험" });
    img(s, "train_compare", 0.4, 1.4, 7.3);
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const r = (a, b, c) => [{ text: a, options: { align: "left" } }, { text: b, options: { align: "center" } }, { text: c, options: { align: "center" } }];
    s.addTable([[H("설정"), H("10에폭 손실"), H("최종 정확도")], r("작은 초기값(0.01)", "1.099", "33.3%"), r("Xavier", "0.818", "100%"), r("He", "0.320", "100%"), r("He + BatchNorm", "0.040", "100%"), r("He + LayerNorm", "0.828", "100%"), r("작은 초기값 + BN", "0.032", "99.8%")], { x: 7.9, y: 1.45, w: 4.9, colW: [2.3, 1.3, 1.3], fontFace: FONT, fontSize: 12, color: C.ink, rowH: 0.42, border: { type: "solid", color: C.line, pt: 1 } });
    d.bullets(s, ["작은 초기값만 쓰면 학습이 아예 시작되지 않는다 (손실 1.099 = ln3, 찍기 수준)", "He가 Xavier보다 ReLU에서 빠르다", "BatchNorm을 넣으면 초기화가 나빠도(작은 초기값) 거뜬히 학습 → 초기화 민감도 감소"], { x: 7.9, y: 4.6, w: 4.9, h: 2.2, size: 13 });
    d.text(s, "※ 씨앗 0, 한 번 실행한 결과. 수치는 환경/난수에 따라 달라질 수 있음", { x: 0.4, y: 5.7, w: 7.3, h: 0.4, size: 12, color: C.muted });
    s.addNotes("가장 극적인 대비: 파란 선(작은 초기값)은 평평, 갈색 선(작은 초기값+BN)은 바로 학습. 이것이 '정규화가 초기화에 덜 민감하게 만든다'의 실증. LayerNorm은 이 문제에서 BN만큼 빠르지는 않았다 -> 이미지/MLP 형 데이터에는 BN이 유리한 경우가 많으나 모델·데이터에 따라 다름.");
  }

  {
    const s = d.slide("[최신 자료] Transformer 시대의 정규화", { tag: "3교시 · 최신" });
    d.bullets(s, [
      { t: "Transformer(GPT, BERT, LLaMA 등)는 BatchNorm이 아니라 LayerNorm 계열을 사용: 시퀀스 길이가 달라지고 배치가 작아도 안정적이기 때문", b: true },
      "RMSNorm : 평균을 빼지 않고 제곱평균제곱근(RMS)으로만 나누는 단순화 버전. LLaMA 등 최신 대형 언어 모델에서 널리 사용 (실습 문제 3)",
      "Pre-LN : 정규화를 서브층(어텐션/FFN) '앞'에 두는 배치. 깊은 모델이 더 안정적으로 학습됨",
      "활성화 함수도 ReLU → GELU / SiLU(Swish) 쪽으로 이동",
      "이미지 모델은 여전히 BatchNorm을 많이 사용하고, 작은 배치에서는 GroupNorm을 쓴다",
    ], { x: 0.7, y: 1.5, w: 12, h: 4.0, size: 17 });
    d.card(s, { x: 0.7, y: 5.6, w: 12, h: 1.1, fill: C.tealL, title: "", size: 15, body: "앞서 배운 Transformer 강의의 LayerNorm이 바로 이 수업의 식 (12.5)~(12.8) 입니다!" });
    s.addNotes("앞 강의(Transformer)와 연결. 원본 자료에 없는 최신 동향을 덧붙인 것이라는 점을 밝힌다.");
  }

  {
    const s = d.slide("Lesson 12 전체 요약", { tag: "SUMMARY" });
    d.bullets(s, [
      "심층 신경망은 활성값과 기울기의 스케일·분포에 매우 민감하다",
      "정규화(BatchNorm, LayerNorm): 특징 값을 표준화해 불안정을 완화하고 층 사이 정보 흐름을 원활하게 한다 (γ, β로 표현력 회복)",
      "적절한 초기화: 학습 초기에 기울기가 너무 크거나 너무 작아지는 것을 방지한다",
      "Xavier는 대칭 활성화(tanh) 계열, He는 ReLU 계열에 맞춘 초기화",
      "정규화와 초기화는 함께 작동하여 현대 딥러닝의 안정적이고 빠르며 견고한 학습의 핵심 기반을 이룬다",
      "ReLU: 기울기 소실을 줄이고 계산이 빠르지만 Dying ReLU에 주의 → He 초기화·정규화와 함께 쓴다",
    ], { x: 0.7, y: 1.5, w: 12, h: 5.2, size: 18 });
    s.addNotes("원본 Summary를 그대로 정리. 3시간 강의 마무리. 이어서 실습 문제를 푼다(슬라이드 뒤쪽에 솔루션).");
  }
};
