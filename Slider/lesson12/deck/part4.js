// 실습 문제, 솔루션, 원본 자료 검토 결과
const { C, FONT } = require("./lib");
const { F } = require("./helpers");

module.exports = function part4(d, pres) {
  d.section("실습", "실습 문제 5개", "손계산 · 구현 · 예측 후 실험  (솔루션은 문제 뒤에 이어집니다)", "문제를 먼저 풀 시간을 충분히 준다(약 20분). 솔루션 슬라이드는 풀이 후에 연다.");

  {
    const s = d.slide("실습 문제 ①~③", { tag: "실습 문제" });
    d.card(s, { x: 0.6, y: 1.5, w: 12.1, h: 1.55, fill: C.tealL, title: "문제 1. 손으로 배치 정규화 (γ=1, β=0, ε 무시)", tColor: C.teal, size: 15, body: "한 특징의 값 x = [1, 3, 5, 7, 9] (배치 크기 5)를 정규화하라. 평균, 분산, 정규화된 5개 값을 구하라." });
    d.card(s, { x: 0.6, y: 3.2, w: 12.1, h: 1.55, fill: C.orangeL, title: "문제 2. 초기화 표준편차 계산", tColor: C.orange, size: 15, body: "입력 256개, 출력 128개인 Dense 층에서 Xavier(정규분포형)와 He 초기화의 분산과 표준편차를 각각 구하라." });
    d.card(s, { x: 0.6, y: 4.9, w: 12.1, h: 1.8, fill: C.purpleL, title: "문제 3. RMSNorm 구현", tColor: C.purple, size: 15, body: "평균을 빼지 않고 y = γ · x / √(mean(x²) + ε) 로 정규화하는 RMSNorm 클래스를 NumPy로 구현하라. x=[[1,2,3,4]]일 때 결과는? LayerNorm과 어떻게 다른가?" });
    s.addNotes("5분씩 풀게 한다.");
  }
  {
    const s = d.slide("실습 문제 ④~⑤", { tag: "실습 문제" });
    d.card(s, { x: 0.6, y: 1.5, w: 12.1, h: 2.3, fill: C.greenL, title: "문제 4. 예측 후 실험 (코드: step2_init.py)", tColor: C.green, size: 15, body: "ReLU 신경망에서 He 대신 분산 1/n_in (std = √(1/n_in))을 쓰면 10층 뒤 활성값의 표준편차는 He의 결과(≈0.725)와 비교해 어떻게 될까?\n먼저 이유와 함께 예측한 뒤, forward_stds로 확인하라." });
    d.card(s, { x: 0.6, y: 4.0, w: 12.1, h: 2.5, fill: C.redL, title: "문제 5. 배치 크기 1의 BatchNorm", tColor: C.red, size: 15, body: "step3_norm.py의 BatchNorm(3)에 β=[0.5, 0.5, 0.5]를 설정하고, 샘플 하나 [[10, -3, 7]]을 training=True 로 넣으면 출력은? 왜 그런가? 같은 입력을 LayerNorm(3)에 넣으면 어떻게 다른가?" });
    s.addNotes("문제 5의 포인트: 배치 1이면 평균 = 자기 자신이라 x - mu = 0.");
  }

  d.section("솔루션", "실습 문제 풀이", "코드: lesson12/code/practice_solutions.py", "풀이를 하나씩 공개한다.");

  {
    const s = d.slide("솔루션 ①: 손으로 배치 정규화", { tag: "솔루션" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const ctr = (t) => ({ text: t, options: { align: "center" } });
    s.addTable([[H("단계"), H("계산"), H("결과")], [ctr("평균"), ctr("(1+3+5+7+9)/5"), ctr("5")], [ctr("분산"), ctr("((−4)²+(−2)²+0²+2²+4²)/5 = 40/5"), ctr("8")], [ctr("표준편차"), ctr("√8"), ctr("≈ 2.828")], [ctr("정규화 x̂"), ctr("(x−5)/2.828"), ctr("−1.414, −0.707, 0, 0.707, 1.414")]], { x: 0.6, y: 1.6, w: 12.1, colW: [2.0, 5.3, 4.8], fontFace: FONT, fontSize: 16, color: C.ink, rowH: 0.65, valign: "middle", border: { type: "solid", color: C.line, pt: 1 } });
    d.card(s, { x: 0.6, y: 5.2, w: 12.1, h: 1.4, fill: C.greenL, title: "검산", tColor: C.green, size: 15, body: "새 평균 = (−1.414 −0.707 + 0 + 0.707 + 1.414)/5 = 0,  새 분산 = (2+0.5+0+0.5+2)/5 = 1  ✔" });
    s.addNotes("실행 결과와 일치: 정규화 [-1.414 -0.707 0. 0.707 1.414].");
  }
  {
    const s = d.slide("솔루션 ②: 초기화 표준편차", { tag: "솔루션" });
    F(d, s, "Xavier : Var = 2/(256+128) = 2/384 ≈ 0.00521   →   std = √0.00521 ≈ 0.0722", { x: 0.6, y: 1.7, w: 12.1, h: 0.9, size: 22, fill: C.tealL });
    F(d, s, "He : Var = 2/256 = 0.0078125   →   std = √0.0078125 ≈ 0.0884", { x: 0.6, y: 2.9, w: 12.1, h: 0.9, size: 22, fill: C.orangeL });
    d.bullets(s, ["He의 분산이 더 크다 (ReLU가 절반을 죽이는 것을 보상)", "코드: np.sqrt(2/(n_in+n_out)),  np.sqrt(2/n_in)", "이 std로 W = rng.normal(0, std, (n_in, n_out)) 를 만들면 초기화 완료"], { x: 0.6, y: 4.2, w: 12.1, h: 2.4, size: 18 });
  }
  {
    const s = d.slide("솔루션 ③: RMSNorm", { tag: "솔루션" });
    d.code(s, [
      "class RMSNorm:",
      "    def __init__(self, dim):",
      "        self.gamma = np.ones(dim)",
      "    def __call__(self, x):",
      "        # 제곱의 평균에 루트 = RMS (평균은 빼지 않는다)",
      "        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)",
      "        return self.gamma * x / rms",
      "",
      "z = np.array([[1., 2, 3, 4]])",
      "# RMSNorm  -> [0.365 0.730 1.095 1.461]",
      "# LayerNorm-> [-1.342 -0.447 0.447 1.342]",
    ], { x: 0.5, y: 1.5, w: 7.4, h: 4.2, size: 13, title: "practice_solutions.py" });
    d.explain(s, ["RMS = √(제곱의 평균) : 값들의 '평균적인 크기'", "평균을 빼지 않으므로 부호와 상대 크기가 그대로 보존 (모두 양수)", "LayerNorm은 평균을 빼서 0 중심으로 옮기지만 RMSNorm은 크기만 맞춘다", "계산이 더 간단하고 빨라서 LLaMA 등에서 사용"], { x: 8.1, y: 1.5, w: 4.6, h: 4.2, size: 13 });
    s.addNotes("검산: mean(x^2)=(1+4+9+16)/4=7.5, sqrt=2.739. 1/2.739=0.365.");
  }
  {
    const s = d.slide("솔루션 ④: 분산 1/n_in 을 쓰면?", { tag: "솔루션" });
    d.bullets(s, [
      { t: "예측: ReLU가 출력의 절반을 0으로 만들어 층마다 신호의 분산이 약 1/2(표준편차 약 0.7배)로 줄어든다", b: true },
      "결과(10층 뒤 표준편차):  He ≈ 0.725   vs   1/n_in ≈ 0.0227   → He가 약 32배 크다",
      "층마다 0.7배씩 줄어들며 신호가 점점 소실 → 깊을수록 격차가 커진다",
      "이것이 ReLU에 He(2/n_in)가 필요한 이유이며, 그림 '실험 결과 ①'의 ReLU + Xavier 선과 같은 모양이다",
    ], { x: 0.6, y: 1.6, w: 12.1, h: 3.8, size: 19 });
    d.code(s, ["half = lambda n_in, n_out: np.sqrt(1 / n_in)", "forward_stds(relu, half)[-1]   # 0.0227"], { x: 0.6, y: 5.5, w: 12.1, h: 1.2, size: 14 });
  }
  {
    const s = d.slide("솔루션 ⑤: 배치 크기 1의 BatchNorm", { tag: "솔루션" });
    d.code(s, [
      "bn = BatchNorm(3); bn.beta[:] = [0.5, 0.5, 0.5]",
      "one = np.array([[10., -3, 7]])",
      "bn(one, training=True)    # [[0.5 0.5 0.5]]  <- β만 남는다!",
      "LayerNorm(3)(one)         # [[ 0.96 -1.379  0.42 ]]",
    ], { x: 0.6, y: 1.5, w: 12.1, h: 1.9, size: 15, title: "practice_solutions.py" });
    d.bullets(s, [
      "배치에 샘플이 1개면 '배치 평균 = 자기 자신' → x − μ = 0 → 정규화 값이 0 → 출력은 β만 남는다 (입력 정보가 전부 사라짐)",
      "그래서 BatchNorm은 충분히 큰 배치가 필요하다",
      "LayerNorm은 샘플 안의 특징끼리 계산하므로 샘플 1개여도 정상 동작한다",
      "추론 때(training=False)는 저장된 이동 평균 통계를 쓰므로 배치 1개여도 괜찮다",
    ], { x: 0.6, y: 3.7, w: 12.1, h: 3.0, size: 18 });
  }

  // ---------------- 원본 자료 검토 ----------------
  {
    const s = d.slide("원본 자료 검토 결과 (강사용): 바로잡은 내용", { tag: "APPENDIX" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t, b) => ({ text: t, options: { align: "left", bold: !!b } });
    s.addTable([
      [H("원본 위치"), H("문제"), H("이 강의에서의 처리")],
      [c("BN 식 설명 (12.3)"), c("'평균을 빼는 것이 분산을 줄인다'고 설명 — 평균을 빼도 분산은 변하지 않음 (분산을 줄이는 것은 표준편차로 나누는 단계)"), c("평균 빼기=0 중심 이동, 표준편차로 나누기=스케일 1로 구분해 설명")],
      [c("LN 식 (12.7)"), c("레이어 정규화 식에 배치 통계 기호 μ_B, σ²_B 사용"), c("μ, σ²로 정정")],
      [c("Xavier 설명"), c("1/n_in 과 2/(n_in+n_out) 를 같은 방법처럼 병기 / sigmoid를 '대칭'으로 설명"), c("LeCun·Glorot 구분, sigmoid 주의 추가")],
      [c("'분산이 크면 폭발'"), c("tanh에서는 폭발이 아니라 포화(±1)로 나타남"), c("실험으로 포화·폭발 구분(ReLU)")],
      [c("BN 설명 전반"), c("추론 시 이동 평균 사용 설명 누락, ICS 설명이 최신 연구와 다름"), c("[보강]·[최신 자료] 슬라이드 추가")],
      [c("표기·중복"), c("Internal Covariants, Intialization, sparity, Parameteric 오타. 개요와 '정규화를 하는 이유' 문단 중복"), c("바로잡음 / 통합")],
    ], { x: 0.4, y: 1.45, w: 12.5, colW: [2.2, 6.0, 4.3], fontFace: FONT, fontSize: 11.5, color: C.ink, rowH: 0.66, valign: "middle", border: { type: "solid", color: C.line, pt: 1 } });
    d.text(s, "※ Lesson 13(데이터 증강)은 이 강의 범위가 아니라 제외했다.", { x: 0.4, y: 6.4, w: 12, h: 0.4, size: 13, color: C.muted });
    s.addNotes("교재 오류 검사 결과를 투명하게 공유. 학생에게는 필요할 때만 언급.");
  }

  {
    const s = pres.addSlide(); d.n += 1; s.background = { color: C.paper };
    s.addText("수고하셨습니다!", { x: 0.9, y: 2.4, w: 11, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    s.addText("다음 시간: Lesson 13. Data Augmentation (데이터 증강)", { x: 0.9, y: 3.6, w: 11, h: 0.6, fontFace: FONT, fontSize: 22, color: C.teal, bold: true, margin: 0 });
    s.addText("튜링랩 프로그래밍학원  |  TuringLab. Programming Academy", { x: 0.9, y: 5.2, w: 11, h: 0.4, fontFace: FONT, fontSize: 16, color: C.muted, margin: 0 });
  }
};
