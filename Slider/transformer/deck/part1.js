// 1회차 (1~3교시): 개념 + 부품별 NumPy 구현
const path = require("path");
const { C, FONT, MONO, IMG } = require("./lib");

module.exports = function part1(d, pres) {
  // ============================== 표지 ==============================
  {
    const s = pres.addSlide(); d.n += 1;
    s.background = { color: C.paper };
    s.addText("Transformer 입문", { x: 0.9, y: 1.6, w: 11, h: 1.1, fontFace: FONT, fontSize: 54, bold: true, color: C.ink, margin: 0 });
    s.addText("기초 개념부터 NumPy · PyTorch 직접 구현까지", { x: 0.9, y: 2.8, w: 11, h: 0.6, fontFace: FONT, fontSize: 24, color: C.teal, bold: true, margin: 0 });
    s.addText("ChatGPT의 심장을 처음부터 끝까지 내 손으로 만들어 봅니다", { x: 0.9, y: 3.5, w: 11, h: 0.5, fontFace: FONT, fontSize: 18, color: C.muted, margin: 0 });
    const toks = [["나는", C.orangeL], ["AI를", C.tealL], ["배운다", C.purpleL]];
    toks.forEach((t, i) => d.box(s, t[0], { x: 0.9 + i * 1.7, y: 4.6, w: 1.5, h: 0.7, fill: t[1], size: 20, bold: true }));
    d.box(s, "→ [0.2, -1.1, 0.7, ...]  숫자 벡터", { x: 6.0, y: 4.6, w: 5.2, h: 0.7, fill: C.gray, size: 18 });
    s.addText("튜링랩 프로그래밍학원\nTuringLab. Programming Academy", { x: 0.9, y: 5.9, w: 6, h: 0.7, fontFace: FONT, fontSize: 14, color: C.ink, margin: 0 });
    s.addImage({ path: path.join(IMG, "logo.png"), x: 9.5, y: 6.0, w: 3.0, h: 3.0 * 118 / 756 });
    s.addNotes("강의 소개. 오늘부터 2회(회당 3시간)에 걸쳐 트랜스포머를 이론 -> NumPy 구현 -> PyTorch 구현 순서로 배운다고 안내.");
  }

  // ============================== 로드맵 ==============================
  {
    const s = d.slide("강의 로드맵: 6교시 (3시간 × 2회)", { tag: "OVERVIEW" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const cell = (t, o = {}) => ({ text: t, options: { align: "left", ...o } });
    s.addTable([
      [H("회차"), H("교시"), H("주제"), H("직접 만드는 것")],
      [cell("1회차", { bold: true }), cell("1교시"), cell("글자를 숫자로: 토큰화·임베딩, 트랜스포머 구조 한눈에"), cell("Step 1  토큰·임베딩")],
      ["", cell("2교시"), cell("순서와 어텐션: 위치 인코딩, Q·K·V, softmax, 마스크"), cell("Step 2~3  위치 인코딩·어텐션")],
      ["", cell("3교시"), cell("멀티헤드, FFN, 잔차연결, LayerNorm, 블록 조립"), cell("Step 4  멀티헤드 어텐션")],
      [cell("2회차", { bold: true }), cell("4교시"), cell("학습 원리: 손실, 역전파, 자동미분 엔진 만들기"), cell("Step 5  미니 Autograd")],
      ["", cell("5교시"), cell("NumPy만으로 GPT 조립·학습·글 생성"), cell("Step 6  완성된 트랜스포머")],
      ["", cell("6교시"), cell("PyTorch로 실전 버전 + 검증 + 실습 문제"), cell("PyTorch Step 1~6")],
    ], { x: 0.6, y: 1.5, w: 12.1, colW: [1.3, 1.1, 6.4, 3.3], fontFace: FONT, fontSize: 15, color: C.ink, rowH: 0.62, valign: "middle", border: { type: "solid", color: C.line, pt: 1 }, fill: { color: C.white } });
    d.text(s, "매 교시: 50분 수업 + 10분 휴식   |   준비물: Python 3.9+, NumPy, (6교시) PyTorch", { x: 0.6, y: 6.1, w: 12.1, h: 0.4, size: 15, color: C.muted });
  }

  // ============================== 학습 목표 ==============================
  {
    const s = d.slide("이 강의가 끝나면 할 수 있는 일", { tag: "GOAL" });
    const items = [
      ["1", "설명하기", "토큰화 → 임베딩 → 어텐션 → 블록 → 출력의 흐름을 내 말로 설명한다"],
      ["2", "계산하기", "어텐션 가중치를 손으로 계산하고 shape을 추적한다"],
      ["3", "만들기 (NumPy)", "PyTorch 없이 자동미분 엔진부터 트랜스포머까지 직접 만든다"],
      ["4", "만들기 (PyTorch)", "같은 모델을 실전 방식으로 구현하고 내장 부품과 비교 검증한다"],
    ];
    items.forEach((it, i) => {
      const y = 1.6 + i * 1.18;
      s.addText(it[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 22, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, it[1], { x: 1.7, y: y, w: 3.2, h: 0.5, size: 20, bold: true });
      d.text(s, it[2], { x: 1.7, y: y + 0.47, w: 10.5, h: 0.5, size: 16, color: C.muted });
    });
  }

  // ============================== 1교시 ==============================
  d.section("1교시", "글자를 숫자로 바꾸기", "토큰화 · 임베딩 · 트랜스포머는 왜 필요한가", "1교시 시작. 컴퓨터는 숫자만 이해한다는 출발점.");

  {
    const s = d.slide("컴퓨터는 글자를 모른다, 숫자만 안다", { tag: "1교시 · 기초 개념" });
    d.box(s, "\"나는 AI를 배운다\"", { x: 0.8, y: 2.0, w: 3.6, h: 0.9, fill: C.orangeL, size: 20, bold: true });
    d.arrow(s, 4.5, 2.45, 5.6, 2.45);
    d.box(s, "???", { x: 5.7, y: 2.0, w: 1.6, h: 0.9, fill: C.gray, size: 24, bold: true });
    d.arrow(s, 7.4, 2.45, 8.5, 2.45);
    d.box(s, "[0.2, -1.1, 0.7, ...]\n숫자 벡터", { x: 8.6, y: 1.9, w: 3.8, h: 1.1, fill: C.tealL, size: 17, bold: true });
    d.card(s, { x: 0.8, y: 3.6, w: 5.8, h: 2.8, title: "문제", body: "신경망의 계산(곱하기·더하기)은\n숫자에만 할 수 있다.\n\n글자 \"사과\"를 어떻게 곱하고 더할까?", fill: C.redL, tColor: C.red, size: 16 });
    d.card(s, { x: 6.9, y: 3.6, w: 5.5, h: 2.8, title: "해결: 두 단계 변환", body: "① 토큰화  글자 → 정수 번호\n② 임베딩  번호 → 숫자 벡터\n\n이 두 단계가 모든 언어모델의 출발점", fill: C.greenL, tColor: C.green, size: 16 });
  }

  {
    const s = d.slide("① 토큰화(Tokenization): 문장을 조각내어 번호 붙이기", { tag: "1교시 · 기초 개념" });
    d.text(s, "토큰(token) = 모델이 다루는 가장 작은 조각. 문자·단어·단어 조각(subword) 등 방식이 여러 가지다.", { x: 0.7, y: 1.45, w: 12, h: 0.5, size: 16, color: C.muted });
    const rows = [
      ["문자 단위", ["h", "e", "l", "l", "o"], "단순, 어휘가 작다 (이번 실습에서 사용)"],
      ["단어 단위", ["I", "love", "AI"], "직관적이지만 어휘가 매우 커진다"],
      ["서브워드(BPE)", ["un", "believ", "able"], "ChatGPT 등 실제 모델이 사용"],
    ];
    rows.forEach((r, i) => {
      const y = 2.2 + i * 1.25;
      d.text(s, r[0], { x: 0.7, y: y + 0.12, w: 2.4, h: 0.5, size: 18, bold: true, color: C.teal });
      r[1].forEach((t, j) => d.box(s, t, { x: 3.2 + j * 1.15, y, w: 1.0, h: 0.7, fill: [C.orangeL, C.tealL, C.purpleL][i], size: 17, bold: true }));
      d.text(s, r[2], { x: 8.9, y: y + 0.12, w: 3.9, h: 0.6, size: 14, color: C.muted });
    });
    d.card(s, { x: 0.7, y: 6.0, w: 12.0, h: 0.8, fill: C.gray, body: "핵심: 어떤 방식이든 \"토큰 → 정수 번호\"로 바뀐다. 번호는 단지 이름표일 뿐, 크기에 의미가 없다.", size: 15 });
  }

  {
    const s = d.slide("어휘집(vocab)과 인코딩 / 디코딩", { tag: "1교시 · 기초 개념" });
    d.text(s, "텍스트 \"hello\"에서 쓰인 글자를 모아 정렬한 뒤, 순서대로 번호를 매긴다.", { x: 0.7, y: 1.45, w: 12, h: 0.45, size: 16, color: C.muted });
    const letters = ["e", "h", "l", "o"];
    letters.forEach((ch, i) => {
      d.box(s, ch, { x: 1.2 + i * 1.3, y: 2.2, w: 1.05, h: 0.7, fill: C.tealL, size: 22, bold: true });
      d.box(s, String(i), { x: 1.2 + i * 1.3, y: 3.0, w: 1.05, h: 0.55, fill: C.gray, size: 18, color: C.teal, bold: true });
    });
    d.text(s, "글자\n번호", { x: 0.3, y: 2.35, w: 0.8, h: 1.2, size: 13, color: C.muted });
    d.text(s, "encode(\"hello\")", { x: 0.7, y: 4.1, w: 4, h: 0.4, size: 17, bold: true });
    ["h", "e", "l", "l", "o"].forEach((ch, i) => d.box(s, ch, { x: 0.7 + i * 0.85, y: 4.6, w: 0.75, h: 0.55, fill: C.orangeL, size: 16, bold: true }));
    d.arrow(s, 3.0, 5.2, 3.0, 5.6);
    ["1", "0", "2", "2", "3"].forEach((n, i) => d.box(s, n, { x: 0.7 + i * 0.85, y: 5.7, w: 0.75, h: 0.55, fill: C.gray, size: 16, bold: true, color: C.teal }));
    d.card(s, { x: 6.5, y: 2.1, w: 6.2, h: 2.1, title: "stoi / itos 두 개의 사전", body: "stoi : 글자 → 번호   {'e':0, 'h':1, ...}\nitos : 번호 → 글자   {0:'e', 1:'h', ...}\n\n학습할 때는 stoi, 글을 만들 때는 itos", size: 15 });
    d.card(s, { x: 6.5, y: 4.4, w: 6.2, h: 1.9, title: "vocab_size (어휘 크기)", body: "서로 다른 토큰의 개수 = V\n출력층이 \"V개 중 다음 토큰은?\"을 맞히게 된다.\n(ChatGPT류는 보통 수만 ~ 수십만)", fill: C.orangeL, tColor: C.orange, size: 15 });
  }

  {
    const s = d.slide("② 임베딩(Embedding): 번호를 의미 있는 숫자 벡터로", { tag: "1교시 · 기초 개념" });
    d.text(s, "번호 3은 단지 이름표다. 그래서 토큰마다 \"성격을 나타내는 숫자 묶음(벡터)\"을 하나씩 준다.", { x: 0.7, y: 1.45, w: 12, h: 0.45, size: 16, color: C.muted });
    d.text(s, "임베딩 표 E  (V × d_model)", { x: 0.7, y: 2.1, w: 5, h: 0.4, size: 16, bold: true, color: C.teal });
    const E = [["번호", "d1", "d2", "d3", "d4"], [0, 0.3, -1.2, 0.8, 0.1], [1, -0.5, 0.4, 0.2, 1.1], [2, 1.4, 0.3, -0.7, 0.6], [3, 0.2, 0.9, 0.5, -0.4]];
    d.grid(s, E, { x: 0.7, y: 2.6, cw: 0.95, ch: 0.5, size: 14, hi: (i, j) => (i === 0 || j === 0 ? C.gray : (i === 3 ? C.orangeL : C.tealL)) });
    d.text(s, "← 2번 행", { x: 5.5, y: 4.2, w: 1.2, h: 0.4, size: 13, color: C.orange, bold: true });
    d.card(s, { x: 0.7, y: 5.3, w: 5.4, h: 1.4, fill: C.gray, body: "임베딩 조회 = E[번호]\n처음엔 랜덤 값이지만 학습하며 의미가 담긴다.\nd_model: 벡터 길이 (GPT-3는 12,288)", size: 14 });
    d.card(s, { x: 6.9, y: 2.1, w: 5.8, h: 4.6, title: "알아 두기", body: "• 임베딩 표는 \"학습되는 파라미터\"다.\n\n• \"one-hot 벡터 × 표\" 행렬곱과 같은 결과인데, 훨씬 빠른 조회 방식이다.\n\n• 같은 토큰은 위치와 상관없이 처음엔 같은 벡터를 받는다. (위치 정보는 2교시에서 따로 더한다)\n\n• 출력 shape: (T, d_model)\n   T = 토큰 개수", size: 15 });
  }

  {
    const s = d.slide("임베딩 공간: 비슷한 의미는 가까이", { tag: "1교시 · 기초 개념" });
    // 2D 산점도 도식
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 1.6, w: 6.6, h: 4.9, fill: { color: C.paper }, line: { color: C.line, width: 1 }, rectRadius: 0.1 });
    const pts = [["강아지", 2.0, 2.3, C.orange], ["고양이", 2.9, 2.0, C.orange], ["호랑이", 3.4, 2.9, C.orange], ["사과", 5.3, 4.9, C.green], ["바나나", 6.0, 4.4, C.green], ["포도", 5.6, 5.6, C.green], ["자동차", 2.4, 5.3, C.purple]];
    pts.forEach(p => {
      s.addShape(pres.shapes.OVAL, { x: p[1] - 0.09, y: p[2] - 0.09, w: 0.18, h: 0.18, fill: { color: p[3] }, line: { color: p[3], width: 0 } });
      d.text(s, p[0], { x: p[1] - 0.45, y: p[2] + 0.12, w: 0.9, h: 0.3, size: 13, bold: true, color: p[3], align: "center" });
    });
    d.text(s, "(실제는 수백~수만 차원이라 그림은 2차원 단순화)", { x: 0.9, y: 6.15, w: 6.4, h: 0.3, size: 11, color: C.muted });
    d.card(s, { x: 7.8, y: 1.6, w: 4.9, h: 2.3, title: "거리 = 의미의 차이", body: "벡터가 가까울수록 의미가 비슷하다.\n학습이 끝난 임베딩에서는\n동물끼리, 과일끼리 모인다.", size: 15 });
    d.card(s, { x: 7.8, y: 4.1, w: 4.9, h: 2.4, title: "유명한 예: 단어 산술", body: "왕 − 남자 + 여자 ≈ 여왕\n\n벡터 계산으로 의미 관계가\n표현되기도 한다.", fill: C.orangeL, tColor: C.orange, size: 15 });
  }

  {
    const s = d.slide("[코드 Step 1] 토큰화와 임베딩 (NumPy)", { tag: "1교시 · 실습", notes: "파일: numpy_version/step1_embedding.py. 학생들이 직접 실행해 보게 한다." });
    d.code(s, [
      "text = \"hello transformer\"",
      "chars = sorted(set(text))        # 중복 제거 + 정렬",
      "vocab_size = len(chars)          # 어휘 크기 V",
      "stoi = {ch: i for i, ch in enumerate(chars)}",
      "itos = {i: ch for ch, i in stoi.items()}",
      "",
      "def encode(s): return [stoi[c] for c in s]",
      "def decode(ids): return \"\".join(itos[i] for i in ids)",
      "",
      "d_model = 8",
      "rng = np.random.default_rng(0)   # 시드 고정 = 재현 가능",
      "E = rng.normal(0, 1, (vocab_size, d_model))  # 임베딩 표",
      "",
      "ids = np.array(encode(\"hello\"))  # 정수 번호",
      "x = E[ids]                       # 표에서 행 꺼내기",
      "# x.shape = (5, 8) = (토큰 수 T, d_model)",
    ], { x: 0.6, y: 1.5, w: 7.7, h: 5.2, size: 12, title: "numpy_version/step1_embedding.py" });
    d.explain(s, [
      "set → sorted: 글자마다 항상 같은 번호가 나오도록 정렬한다.",
      "stoi / itos: 글자↔번호 변환용 사전 두 개.",
      "E는 (V, d_model) 크기의 학습 대상 표다.",
      "E[ids]: 번호를 행 번호로 써서 벡터를 한꺼번에 꺼낸다.",
      "이 결과 x가 다음 단계(위치 인코딩, 어텐션)의 입력이 된다.",
    ], { x: 8.6, y: 1.5, w: 4.1, h: 5.2, size: 13 });
  }

  {
    const s = d.slide("왜 트랜스포머인가? 기존 방식(RNN)의 한계", { tag: "1교시 · 기초 개념" });
    d.card(s, { x: 0.6, y: 1.5, w: 5.9, h: 4.9, fill: C.redL, title: "RNN: 한 단어씩 순서대로", tColor: C.red });
    ["나는", "AI를", "배운다"].forEach((t, i) => {
      d.box(s, t, { x: 0.9 + i * 1.75, y: 2.5, w: 1.3, h: 0.7, fill: C.white, size: 16, bold: true });
      if (i < 2) d.arrow(s, 2.25 + i * 1.75, 2.85, 2.6 + i * 1.75, 2.85, C.red);
    });
    d.bullets(s, ["앞 단어를 처리해야 다음 단어 가능 → 병렬화 어려움, 느림", "문장이 길면 앞부분을 점점 잊어버림", "멀리 떨어진 단어 사이 관계 파악이 어려움"], { x: 0.9, y: 3.6, w: 5.3, h: 2.7, size: 15 });
    d.card(s, { x: 6.8, y: 1.5, w: 5.9, h: 4.9, fill: C.greenL, title: "Transformer: 모두를 한꺼번에", tColor: C.green });
    const xs = [7.1, 8.9, 10.7];
    xs.forEach((x, i) => d.box(s, ["나는", "AI를", "배운다"][i], { x, y: 2.5, w: 1.3, h: 0.7, fill: C.white, size: 16, bold: true }));
    d.arrow(s, 7.75, 3.25, 11.35, 3.25, C.green, 1.5);
    d.arrow(s, 11.35, 3.4, 7.75, 3.4, C.green, 1.5);
    d.bullets(s, ["모든 단어가 모든 단어를 직접 참조(어텐션)", "한꺼번에 계산 → GPU 병렬화에 최적, 빠름", "거리와 상관없이 관계를 바로 연결"], { x: 7.1, y: 3.6, w: 5.3, h: 2.7, size: 15 });
  }

  {
    const s = d.slide("트랜스포머 연대기", { tag: "1교시 · 기초 개념" });
    const ev = [
      ["2017", "Transformer 등장", "논문 \"Attention Is All You Need\" (Google)", C.teal],
      ["2018", "BERT · GPT-1", "이해(BERT)와 생성(GPT) 두 갈래 시작", C.purple],
      ["2020", "GPT-3", "모델을 키울수록 능력이 크게 향상", C.orange],
      ["2022", "ChatGPT", "대화형 서비스로 대중화", C.green],
      ["2023~", "멀티모달·에이전트", "이미지·음성·코드·도구 사용으로 확장", C.red],
    ];
    ev.forEach((e, i) => {
      const x = 0.6 + i * 2.45;
      d.box(s, e[0], { x, y: 1.9, w: 2.2, h: 0.7, fill: e[3], color: C.white, size: 22, bold: true });
      d.card(s, { x, y: 2.8, w: 2.2, h: 2.5, fill: C.paper, title: e[1], tColor: e[3], body: e[2], size: 13, tSize: 15 });
    });
    d.card(s, { x: 0.6, y: 5.8, w: 12.1, h: 0.9, fill: C.gray, body: "오늘 만드는 구조 = GPT 계열이 쓰는 \"디코더 트랜스포머\"의 축소판. 원리는 ChatGPT와 같고 크기만 다르다.", size: 15 });
  }

  {
    const s = d.slide("트랜스포머 전체 구조 한눈에 (오늘의 지도)", { tag: "1교시 · 기초 개념", notes: "이 지도를 매 교시 시작할 때 다시 보여주며 현재 위치를 표시한다." });
    const stack = [
      ["다음 토큰 확률 (softmax)", C.greenL],
      ["Linear: 어휘 크기 V개의 점수", C.greenL],
      ["LayerNorm", C.gray],
      ["트랜스포머 블록 × N  (어텐션 + FFN)", C.orangeL],
      ["임베딩 + 위치 인코딩", C.tealL],
      ["토큰화 (글자 → 번호)", C.tealL],
      ["입력 문장", C.paper],
    ];
    stack.forEach((b, i) => {
      const y = 1.5 + i * 0.78;
      d.box(s, b[0], { x: 0.8, y, w: 5.6, h: 0.58, fill: b[1], size: 15, bold: i === 3, line: i === 3 ? C.orange : null });
      if (i < 6) d.arrow(s, 3.6, y + 0.78, 3.6, y + 0.6, C.muted, 1.5);
    });
    const notes = [
      [1, "5교시: 출력층·학습"],
      [3, "2~3교시: 어텐션, FFN, 잔차, 정규화"],
      [4, "1~2교시: Step 1, 2"],
    ];
    d.card(s, { x: 7.0, y: 1.5, w: 5.7, h: 1.4, title: "블록 안에서 일어나는 일", body: "x = x + 어텐션(LayerNorm(x))\nx = x + FFN(LayerNorm(x))", fill: C.orangeL, tColor: C.orange, size: 14 });
    d.card(s, { x: 7.0, y: 3.1, w: 5.7, h: 1.6, title: "아래에서 위로 한 방향", body: "입력이 층을 통과할 때마다 단어 벡터가\n문맥을 담아 점점 똑똑해진다.", size: 14 });
    d.card(s, { x: 7.0, y: 4.9, w: 5.7, h: 1.6, title: "마지막 한 줄", body: "마지막 위치의 출력 점수 → softmax →\n\"다음에 올 토큰\" 확률 분포", fill: C.greenL, tColor: C.green, size: 14 });
  }

  {
    const s = d.slide("1교시 정리 퀴즈", { tag: "1교시 · 정리", notes: "정답: 1) 숫자만 계산할 수 있어서 2) V x d_model 3) 병렬 처리 가능, 먼 단어 관계 직접 연결" });
    const qs = [
      ["Q1", "컴퓨터에 글자를 넣기 전에 왜 숫자로 바꿔야 할까요?"],
      ["Q2", "어휘 크기가 28, d_model이 32라면 임베딩 표의 shape은?"],
      ["Q3", "RNN 대비 트랜스포머의 장점 두 가지는?"],
    ];
    qs.forEach((q, i) => {
      const y = 1.7 + i * 1.5;
      d.box(s, q[0], { x: 0.8, y, w: 1.0, h: 0.9, fill: C.teal, color: C.white, size: 22, bold: true });
      d.card(s, { x: 2.0, y, w: 10.6, h: 0.9, fill: C.paper, body: q[1], size: 18 });
    });
    d.text(s, "→ 5분 쉬고 2교시에서 \"순서\"와 \"어텐션\"을 배웁니다.", { x: 0.8, y: 6.3, w: 11, h: 0.4, size: 16, color: C.muted });
  }

  // ============================== 2교시 ==============================
  d.section("2교시", "순서와 어텐션", "위치 인코딩 · Query / Key / Value · softmax · 마스크", "2교시 시작. 오늘의 핵심 개념인 어텐션을 천천히 다룬다.");

  {
    const s = d.slide("문제: 어텐션은 단어의 순서를 모른다", { tag: "2교시 · 위치 인코딩" });
    d.box(s, "개가 사람을 물었다", { x: 0.8, y: 1.8, w: 5.2, h: 0.8, fill: C.tealL, size: 20, bold: true });
    d.box(s, "사람이 개를 물었다", { x: 7.0, y: 1.8, w: 5.2, h: 0.8, fill: C.orangeL, size: 20, bold: true });
    d.text(s, "뜻이 완전히 다르다!", { x: 4.4, y: 2.8, w: 4.6, h: 0.5, size: 20, bold: true, color: C.red, align: "center" });
    d.card(s, { x: 0.8, y: 3.7, w: 5.8, h: 2.8, title: "왜 문제일까?", body: "어텐션은 단어 집합을 한꺼번에 처리한다.\n단어가 같으면 순서를 바꿔도\n같은 계산이 되어 버린다.\n(RNN은 순서대로 읽어서 자동으로 알았다)", fill: C.redL, tColor: C.red, size: 15 });
    d.card(s, { x: 6.9, y: 3.7, w: 5.8, h: 2.8, title: "해결: 위치 표시를 더해 준다", body: "각 토큰 벡터에 \"나는 몇 번째\"를\n나타내는 위치 벡터를 더한다.\n\n입력 = 단어 임베딩 + 위치 인코딩", fill: C.greenL, tColor: C.green, size: 15 });
  }

  {
    const s = d.slide("위치 인코딩 아이디어: 시계의 바늘", { tag: "2교시 · 위치 인코딩" });
    d.text(s, "시침·분침·초침은 서로 다른 속도로 돈다. 세 바늘의 각도 조합으로 하루 어느 시각이든 유일하게 표현할 수 있다.", { x: 0.7, y: 1.45, w: 12, h: 0.6, size: 16, color: C.muted });
    [["빠르게 도는 파형", "차원 0, 1", "위치의 \"세밀한\" 차이"], ["중간 속도 파형", "차원 2, 3", "위치의 \"중간\" 차이"], ["느리게 도는 파형", "차원 4, 5...", "위치의 \"큰\" 차이"]].forEach((w, i) => {
      d.card(s, { x: 0.8 + i * 4.1, y: 2.4, w: 3.8, h: 1.9, fill: [C.orangeL, C.tealL, C.purpleL][i], title: w[0], tColor: [C.orange, C.teal, C.purple][i], body: w[1] + "\n" + w[2], size: 15 });
    });
    d.card(s, { x: 0.8, y: 4.6, w: 11.9, h: 1.9, title: "사인·코사인을 쓰는 이유", body: "• 값이 항상 −1 ~ 1 사이라 임베딩을 망가뜨리지 않는다.\n• 학습이 필요 없고, 훈련 때 못 본 긴 문장에도 계산할 수 있다.\n• 가까운 위치끼리는 비슷한 벡터가 된다.", size: 15 });
  }

  {
    const s = d.slide("위치 인코딩 공식과 모양", { tag: "2교시 · 위치 인코딩" });
    d.card(s, { x: 0.7, y: 1.5, w: 6.0, h: 1.9, fill: C.gray, title: "공식 (pos = 위치, i = 차원 쌍 번호)", body: "PE(pos, 2i)   = sin( pos / 10000^(2i/d) )\nPE(pos, 2i+1) = cos( pos / 10000^(2i/d) )", size: 15 });
    d.bullets(s, ["짝수 차원은 sin, 홀수 차원은 cos", "분모 10000^(2i/d): 차원이 커질수록 파형이 느려진다", "임베딩에 더한다 (이어 붙이지 않는다)", "x_in = 임베딩 + PE[:T]  → shape 그대로 (T, d)"], { x: 0.7, y: 3.6, w: 6.0, h: 3.0, size: 15 });
    s.addImage({ path: path.join(IMG, "pe_heatmap.png"), x: 7.0, y: 1.55, w: 5.7, h: 5.7 * 3.6 / 5.2 });
    d.text(s, "가로: 차원 / 세로: 위치. 왼쪽 차원일수록 빠르게 변한다.", { x: 7.0, y: 5.6, w: 5.7, h: 0.4, size: 12, color: C.muted });
  }

  {
    const s = d.slide("[코드 Step 2] 위치 인코딩 (NumPy)", { tag: "2교시 · 실습", notes: "파일: numpy_version/step2_positional_encoding.py" });
    d.code(s, [
      "def positional_encoding(max_len, d_model):",
      "    pos = np.arange(max_len)[:, None]      # (max_len, 1)",
      "    i = np.arange(0, d_model, 2)[None, :]  # (1, d/2)",
      "    angle = pos / np.power(10000.0, i / d_model)",
      "    pe = np.zeros((max_len, d_model))",
      "    pe[:, 0::2] = np.sin(angle)            # 짝수 열",
      "    pe[:, 1::2] = np.cos(angle)            # 홀수 열",
      "    return pe",
      "",
      "pe = positional_encoding(16, 8)",
      "x_in = x + pe[:x.shape[0]]   # 임베딩에 '더하기'",
      "",
      "# 가까운 위치일수록 내적이 크다 (검산)",
      "pe = positional_encoding(64, 32)",
      "pe[0] @ pe[1]    # 15.3  (거리 1)",
      "pe[0] @ pe[48]   # 5.3   (거리 48)",
    ], { x: 0.6, y: 1.5, w: 7.7, h: 5.2, size: 12, title: "numpy_version/step2_positional_encoding.py" });
    d.explain(s, [
      "[:, None] / [None, :] 로 축을 만들어 (위치 × 차원) 표를 한 번에 계산 (브로드캐스팅).",
      "0::2 는 0,2,4... 번째 열, 1::2 는 1,3,5... 번째 열.",
      "pe 표는 학습하지 않는 상수다. 한 번 만들어 계속 쓴다.",
      "x + pe[:T] : 같은 shape이어서 원소별 덧셈이 가능하다.",
      "마지막 두 줄: 거리가 멀어질수록 내적이 줄어드는 것을 확인한다.",
    ], { x: 8.6, y: 1.5, w: 4.1, h: 5.2, size: 13 });
  }

  {
    const s = d.slide("셀프 어텐션 직관: 도서관에서 책 찾기", { tag: "2교시 · 어텐션" });
    d.card(s, { x: 0.7, y: 1.6, w: 3.9, h: 3.0, fill: C.orangeL, title: "Query (질문)", tColor: C.orange, body: "\"AI 입문서 있나요?\"\n\n내가 지금 찾고 싶은 것", size: 15 });
    d.card(s, { x: 4.8, y: 1.6, w: 3.9, h: 3.0, fill: C.tealL, title: "Key (색인·이름표)", tColor: C.teal, body: "책등에 적힌 제목·분류\n\n\"나는 이런 정보를 가졌어요\"", size: 15 });
    d.card(s, { x: 8.9, y: 1.6, w: 3.8, h: 3.0, fill: C.purpleL, title: "Value (책의 내용)", tColor: C.purple, body: "실제로 가져올 내용\n\n선택된 책에서 얻는 정보", size: 15 });
    d.text(s, "질문(Q)과 이름표(K)가 잘 맞을수록 → 그 책의 내용(V)을 많이 가져온다", { x: 0.7, y: 4.9, w: 12, h: 0.5, size: 18, bold: true, color: C.ink });
    d.card(s, { x: 0.7, y: 5.6, w: 12, h: 1.1, fill: C.gray, body: "문장 속 모든 단어가 각자 Q, K, V를 가진다. 모든 단어가 \"나는 누구를 얼마나 참고할까?\"를 동시에 계산한다.", size: 15 });
  }

  {
    const s = d.slide("예: \"그것\"은 무엇을 가리킬까?", { tag: "2교시 · 어텐션" });
    d.text(s, "\"강아지가 길을 건너지 못했다. 그것은 너무 피곤했다.\"", { x: 0.7, y: 1.5, w: 12, h: 0.5, size: 20, bold: true });
    d.text(s, "사람은 \"그것 = 강아지\"를 바로 안다. 어텐션은 \"그것\"이 다른 단어를 얼마나 참고할지를 숫자로 계산한다.", { x: 0.7, y: 2.1, w: 12, h: 0.6, size: 15, color: C.muted });
    const bars = [["강아지", 0.62, C.orange], ["길", 0.15, C.teal], ["건너지", 0.08, C.teal], ["못했다", 0.05, C.teal], ["그것", 0.10, C.teal]];
    d.text(s, "\"그것\"이 각 단어에 주는 어텐션 가중치 (예시)", { x: 0.9, y: 3.0, w: 8, h: 0.4, size: 16, bold: true, color: C.teal });
    bars.forEach((b, i) => {
      const y = 3.6 + i * 0.6;
      d.text(s, b[0], { x: 0.9, y: y + 0.05, w: 1.5, h: 0.4, size: 15, bold: true });
      s.addShape(pres.shapes.RECTANGLE, { x: 2.5, y, w: b[1] * 8.0, h: 0.45, fill: { color: b[2] }, line: { color: b[2], width: 0 } });
      d.text(s, b[1].toFixed(2), { x: 2.6 + b[1] * 8.0, y: y + 0.05, w: 0.8, h: 0.4, size: 14 });
    });
    d.card(s, { x: 9.6, y: 3.6, w: 3.1, h: 2.9, fill: C.greenL, title: "핵심", tColor: C.green, body: "합계 = 1.00\n(비율이 된다)\n\n큰 가중치의 단어 정보를 많이 섞는다.", size: 14 });
  }

  {
    const s = d.slide("1단계: 입력 x에서 Q, K, V 만들기", { tag: "2교시 · 어텐션" });
    d.box(s, "X\n(T × d_model)", { x: 0.8, y: 3.0, w: 2.0, h: 1.2, fill: C.gray, size: 16, bold: true });
    [["Q = X · Wq", C.orangeL, 1.9], ["K = X · Wk", C.tealL, 3.1], ["V = X · Wv", C.purpleL, 4.3]].forEach(r => {
      d.arrow(s, 2.9, 3.6, 4.4, r[2] + 0.4, C.muted, 1.5);
      d.box(s, r[0], { x: 4.5, y: r[2], w: 3.2, h: 0.8, fill: r[1], size: 18, bold: true });
    });
    d.card(s, { x: 8.2, y: 1.6, w: 4.5, h: 4.9, title: "알아 둘 점", body: "• Wq, Wk, Wv는 학습으로 바뀌는 가중치(파라미터)\n\n• 같은 입력 X에서 서로 다른 세 가지 \"관점\"을 만든다.\n\n• shape: (T, d_model) × (d_model, d_k) = (T, d_k)\n\n• 셀프(self) 어텐션: Q, K, V가 모두 같은 문장에서 나와서 붙은 이름", size: 14 });
  }

  {
    const s = d.slide("2단계: 점수 = Q와 K의 내적 (유사도)", { tag: "2교시 · 어텐션" });
    d.card(s, { x: 0.7, y: 1.5, w: 5.8, h: 2.2, fill: C.gray, title: "내적(dot product)이란?", body: "a = [1, 2, 3],  b = [4, 5, 6]\na · b = 1×4 + 2×5 + 3×6 = 32\n\n방향이 비슷하면 크고, 반대면 작다(음수).", size: 15 });
    d.card(s, { x: 0.7, y: 3.9, w: 5.8, h: 2.6, fill: C.tealL, title: "모든 쌍을 한 번에: Q · Kᵀ", body: "(T, d_k) × (d_k, T) = (T, T)\n\n(i, j) 칸 = \"i번 단어의 질문이\nj번 단어의 이름표와 얼마나 맞나\"", size: 15 });
    d.text(s, "점수 행렬 (T = 3 예시)", { x: 7.2, y: 1.6, w: 5, h: 0.4, size: 16, bold: true, color: C.teal });
    d.grid(s, [["", "나는", "AI를", "배운다"], ["나는", "2.1", "0.3", "0.8"], ["AI를", "0.4", "1.9", "1.2"], ["배운다", "0.9", "1.1", "2.4"]], { x: 7.2, y: 2.1, cw: 1.3, ch: 0.55, size: 14, hi: (i, j) => (i === 0 || j === 0 ? C.gray : (i === j ? C.orangeL : C.tealL)) });
    d.text(s, "대각선(자기 자신)이 크게 나오는 경향이 있다. 행 하나 = 한 단어의 \"참고 점수표\".", { x: 7.2, y: 4.5, w: 5.4, h: 0.8, size: 14, color: C.muted });
  }

  {
    const s = d.slide("왜 √d_k 로 나눌까?", { tag: "2교시 · 어텐션" });
    d.text(s, "차원(d_k)이 클수록 내적 값의 크기도 커진다. 값이 너무 크면 softmax가 한 칸만 1에 가깝게 쏠려 학습이 멈춘다.", { x: 0.7, y: 1.45, w: 12, h: 0.6, size: 16, color: C.muted });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t) => ({ text: t, options: { align: "center" } });
    s.addTable([
      [H("d_k"), H("내적 값의 표준편차"), H("√d_k 로 나눈 뒤")],
      [c("4"), c("약 2"), c("약 1")],
      [c("64"), c("약 8"), c("약 1")],
      [c("512"), c("약 22.6"), c("약 1")],
    ], { x: 0.8, y: 2.3, w: 6.6, colW: [1.6, 2.7, 2.3], fontFace: FONT, fontSize: 15, color: C.ink, rowH: 0.55, border: { type: "solid", color: C.line, pt: 1 }, valign: "middle" });
    d.text(s, "(Q, K 성분이 평균 0, 분산 1일 때의 통계적 크기)", { x: 0.8, y: 4.6, w: 6.6, h: 0.4, size: 12, color: C.muted });
    d.card(s, { x: 7.8, y: 2.3, w: 4.9, h: 2.6, fill: C.redL, title: "나누지 않으면", tColor: C.red, body: "softmax([20, 1, 0]) ≈ [1, 0, 0]\n→ 기울기가 거의 0\n→ 학습이 진행되지 않는다.", size: 15 });
    d.card(s, { x: 0.8, y: 5.3, w: 11.9, h: 1.3, fill: C.greenL, title: "그래서", tColor: C.green, body: "scores = Q · Kᵀ / √d_k   (크기를 항상 일정하게 유지하는 안전장치)", size: 16 });
  }

  {
    const s = d.slide("3단계: softmax로 점수를 확률(비율)로", { tag: "2교시 · 어텐션" });
    d.card(s, { x: 0.7, y: 1.5, w: 5.8, h: 1.8, fill: C.gray, title: "softmax 공식", body: "softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)\n\n① 모두 양수로  ② 합이 1이 되게", size: 15 });
    d.text(s, "예: 점수 [2.0, 1.0, 0.1]", { x: 0.9, y: 3.6, w: 5, h: 0.4, size: 16, bold: true, color: C.teal });
    d.grid(s, [["점수", "2.0", "1.0", "0.1"], ["exp", "7.39", "2.72", "1.11"], ["확률", "0.66", "0.24", "0.10"]], { x: 0.9, y: 4.1, cw: 1.3, ch: 0.55, size: 15, hi: (i, j) => (j === 0 ? C.gray : (i === 2 ? C.greenL : C.tealL)) });
    d.text(s, "합 = 7.39 + 2.72 + 1.11 = 11.22  →  7.39 / 11.22 = 0.66", { x: 0.9, y: 6.0, w: 6, h: 0.4, size: 13, color: C.muted });
    d.card(s, { x: 7.0, y: 1.5, w: 5.7, h: 2.3, fill: C.orangeL, title: "수치 안정화 요령", tColor: C.orange, body: "exp(1000)은 컴퓨터가 터진다(overflow).\n최댓값을 먼저 빼도 결과는 같다:\nsoftmax(z) = softmax(z − max(z))", size: 14 });
    d.card(s, { x: 7.0, y: 4.0, w: 5.7, h: 2.5, title: "행(row) 단위로 적용", body: "점수 행렬 (T, T)의 각 행에 softmax를 적용\n→ 한 단어가 모든 단어에 주는 비율이 된다.\n(axis = −1)", size: 14 });
  }

  {
    const s = d.slide("4단계: 비율대로 V를 섞는다 (손으로 계산해 보기)", { tag: "2교시 · 어텐션", notes: "검증값: python -c로 확인한 값. Q=K=V=X, d_k=2." });
    d.text(s, "X = Q = K = V = [[1,0], [0,1], [1,1]]   (토큰 3개, d_k = 2)", { x: 0.7, y: 1.4, w: 12, h: 0.4, size: 16, bold: true });
    d.text(s, "① 점수 Q·Kᵀ/√2", { x: 0.7, y: 2.0, w: 4, h: 0.4, size: 15, bold: true, color: C.teal });
    d.grid(s, [["0.71", "0", "0.71"], ["0", "0.71", "0.71"], ["0.71", "0.71", "1.41"]], { x: 0.7, y: 2.45, cw: 0.9, ch: 0.5, size: 14 });
    d.text(s, "② softmax", { x: 4.2, y: 2.0, w: 3, h: 0.4, size: 15, bold: true, color: C.green });
    d.grid(s, [["0.40", "0.20", "0.40"], ["0.20", "0.40", "0.40"], ["0.25", "0.25", "0.50"]], { x: 4.2, y: 2.45, cw: 0.9, ch: 0.5, size: 14, fill: C.greenL, hi: (i, j) => (i === 0 ? C.orangeL : null) });
    d.text(s, "③ 출력 = 가중치 · V", { x: 7.8, y: 2.0, w: 4.5, h: 0.4, size: 15, bold: true, color: C.purple });
    d.grid(s, [["0.80", "0.60"], ["0.60", "0.80"], ["0.75", "0.75"]], { x: 7.8, y: 2.45, cw: 0.9, ch: 0.5, size: 14, fill: C.purpleL, hi: (i) => (i === 0 ? C.orangeL : null) });
    d.card(s, { x: 0.7, y: 4.3, w: 12.0, h: 2.3, fill: C.paper, title: "첫 번째 토큰 계산 (주황색 행)", body: "출력₀ = 0.40 × [1,0]  +  0.20 × [0,1]  +  0.40 × [1,1]\n       = [0.40, 0] + [0, 0.20] + [0.40, 0.40]  =  [0.80, 0.60]\n\n→ 첫 단어는 자기 자신과 세 번째 단어를 40%씩, 두 번째를 20% 섞은 새로운 벡터가 되었다.", size: 15 });
  }

  {
    const s = d.slide("어텐션 공식 총정리와 shape 추적", { tag: "2교시 · 어텐션" });
    d.card(s, { x: 0.7, y: 1.5, w: 12.0, h: 1.2, fill: C.tealL, body: "Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V", size: 24 });
    const steps = [["X", "(T, d_model)"], ["Q, K, V", "(T, d_k)"], ["Q·Kᵀ / √d_k", "(T, T)"], ["softmax", "(T, T)"], ["· V", "(T, d_k)"]];
    steps.forEach((st, i) => {
      const x = 0.7 + i * 2.5;
      d.box(s, st[0], { x, y: 3.2, w: 2.1, h: 0.8, fill: [C.gray, C.orangeL, C.tealL, C.greenL, C.purpleL][i], size: 16, bold: true });
      d.text(s, st[1], { x, y: 4.1, w: 2.1, h: 0.4, size: 14, color: C.teal, bold: true, align: "center" });
      if (i < 4) d.arrow(s, x + 2.12, 3.6, x + 2.45, 3.6);
    });
    d.card(s, { x: 0.7, y: 4.9, w: 12.0, h: 1.7, fill: C.gray, body: "입력과 출력의 shape이 같다 (T, ·) — 단어 수는 그대로, 각 벡터가 \"문맥을 반영한 새 벡터\"로 바뀐다.\n디버깅 팁: shape을 매 단계 print 하면 대부분의 버그가 바로 보인다.", size: 15 });
  }

  {
    const s = d.slide("[코드 Step 3] 어텐션 함수 (NumPy)", { tag: "2교시 · 실습", notes: "파일: numpy_version/step3_attention.py" });
    d.code(s, [
      "def softmax(x, axis=-1):",
      "    x = x - x.max(axis=axis, keepdims=True)  # overflow 방지",
      "    e = np.exp(x)",
      "    return e / e.sum(axis=axis, keepdims=True)",
      "",
      "def scaled_dot_product_attention(Q, K, V, mask=None):",
      "    d_k = Q.shape[-1]",
      "    scores = Q @ K.T / np.sqrt(d_k)       # (T, T) 점수",
      "    if mask is not None:",
      "        scores = np.where(mask, -1e9, scores)  # 못 보는 칸",
      "    weights = softmax(scores, axis=-1)    # 각 행의 합 = 1",
      "    return weights @ V, weights           # 비율대로 V 섞기",
      "",
      "# 사용: x에서 Q, K, V를 만든 뒤 호출",
      "Q, K, V = x @ Wq, x @ Wk, x @ Wv",
      "out, w = scaled_dot_product_attention(Q, K, V)",
    ], { x: 0.6, y: 1.5, w: 7.9, h: 5.2, size: 12, title: "numpy_version/step3_attention.py" });
    d.explain(s, [
      "softmax: 최댓값을 빼 두어도 결과는 같고 overflow만 막힌다.",
      "Q @ K.T: 모든 단어쌍의 점수를 행렬곱 한 번으로 계산한다.",
      "/ np.sqrt(d_k): 점수의 크기를 일정하게 유지한다.",
      "mask의 True 칸을 −1e9로 채우면 softmax 후 확률이 0이 된다.",
      "weights @ V: 비율(가중치)대로 Value를 섞어 최종 출력을 만든다.",
    ], { x: 8.8, y: 1.5, w: 3.9, h: 5.2, size: 13 });
  }

  {
    const s = d.slide("인과 마스크(Causal Mask): 미래는 볼 수 없다", { tag: "2교시 · 어텐션" });
    d.text(s, "글을 쓰는 모델(GPT)은 \"다음 단어\"를 맞혀야 한다. 정답이 될 미래 단어를 미리 보면 시험지를 훔쳐보는 것!", { x: 0.7, y: 1.45, w: 12, h: 0.6, size: 16, color: C.muted });
    d.text(s, "허용(1) / 차단(0)", { x: 0.9, y: 2.3, w: 4, h: 0.4, size: 15, bold: true, color: C.teal });
    const tri = [["1", "0", "0", "0"], ["1", "1", "0", "0"], ["1", "1", "1", "0"], ["1", "1", "1", "1"]];
    d.grid(s, tri, { x: 0.9, y: 2.8, cw: 0.85, ch: 0.6, size: 18, hi: (i, j) => (j > i ? C.redL : C.greenL) });
    d.text(s, "i번째 단어는 자기 자신과 앞 단어(≤ i)만 참고", { x: 0.9, y: 5.35, w: 4.4, h: 0.6, size: 13, color: C.muted });
    d.card(s, { x: 5.6, y: 2.3, w: 7.1, h: 3.7, title: "구현 방법", body: "① 위쪽 삼각형(대각선 위)이 True인 마스크 생성\n   mask = np.triu(np.ones((T,T), bool), k=1)\n\n② softmax 전에 True 칸을 −1e9로 덮어쓰기\n\n③ softmax(−1e9) ≈ 0 → 가중치 0\n\n※ 번역의 인코더처럼 문장 전체를 보는 경우엔 마스크를 쓰지 않는다.", size: 14 });
  }

  {
    const s = d.slide("실제 어텐션 가중치 보기: 마스크 전과 후", { tag: "2교시 · 실습" });
    s.addImage({ path: path.join(IMG, "attention_heatmap.png"), x: 0.7, y: 1.5, w: 8.0, h: 8.0 * 3.5 / 8.2 });
    d.card(s, { x: 9.0, y: 1.5, w: 3.7, h: 3.7, title: "읽는 법", body: "행 = 질문하는 단어(Q)\n열 = 참고되는 단어(K)\n진할수록 많이 참고\n\n오른쪽: 위쪽 삼각형이 0이 되어 \"과거만\" 본다.", size: 14 });
    d.code(s, [
      "out, w = scaled_dot_product_attention(",
      "    Q, K, V, mask=causal_mask(T))",
      "assert np.allclose(np.triu(w, k=1), 0)",
    ], { x: 0.7, y: 5.3, w: 8.0, h: 1.3, size: 12 });
    d.text(s, "assert 로 \"대각선 위쪽은 전부 0\"임을 검산한다.", { x: 9.0, y: 5.4, w: 3.7, h: 0.8, size: 13, color: C.muted });
  }

  {
    const s = d.slide("2교시 정리 퀴즈", { tag: "2교시 · 정리", notes: "정답: 1) 순서를 알려 주기 위해 2) Q·K의 유사도로 V를 섞을 비율을 정한다 3) 학습 목표인 미래 정답을 훔쳐보지 못하게" });
    const qs = ["Q1  위치 인코딩을 임베딩에 \"더하는\" 이유는?", "Q2  Q, K, V는 각각 어떤 역할인가?", "Q3  인과 마스크가 없으면 어떤 문제가 생길까?"];
    qs.forEach((q, i) => d.card(s, { x: 0.8, y: 1.7 + i * 1.5, w: 11.9, h: 1.0, fill: C.paper, body: q, size: 19 }));
    d.text(s, "→ 10분 휴식 후 3교시: 멀티헤드와 블록 조립", { x: 0.8, y: 6.3, w: 11, h: 0.4, size: 16, color: C.muted });
  }

  // ============================== 3교시 ==============================
  d.section("3교시", "멀티헤드와 트랜스포머 블록", "멀티헤드 어텐션 · FFN · 잔차연결 · LayerNorm · 블록 쌓기", "3교시 시작. 어텐션을 확장하고 블록을 완성한다.");

  {
    const s = d.slide("멀티헤드 어텐션: 여러 관점으로 동시에 본다", { tag: "3교시 · 멀티헤드" });
    d.text(s, "한 번의 어텐션은 \"한 가지 관계\"만 보기 쉽다. 여러 개의 작은 어텐션(head)을 나란히 돌려 서로 다른 관계를 잡는다.", { x: 0.7, y: 1.45, w: 12, h: 0.6, size: 16, color: C.muted });
    [["Head 1", "문법 관계", "주어 ↔ 동사", C.orange, C.orangeL], ["Head 2", "가까운 단어", "바로 앞·뒤 단어", C.teal, C.tealL], ["Head 3", "의미 관계", "대명사 ↔ 대상", C.purple, C.purpleL], ["Head 4", "기타 패턴", "학습이 알아서 결정", C.green, C.greenL]].forEach((h, i) => {
      d.card(s, { x: 0.7 + i * 3.05, y: 2.4, w: 2.85, h: 2.2, fill: h[4], title: h[0], tColor: h[3], body: h[1] + "\n" + h[2], size: 14 });
    });
    d.card(s, { x: 0.7, y: 4.9, w: 12.0, h: 1.7, fill: C.gray, title: "비용은 그대로", body: "d_model = 32, 헤드 4개 → 헤드마다 d_k = 32 / 4 = 8 차원만 담당.\n벡터를 쪼개 쓰므로 계산량은 헤드 1개 때와 거의 같다. 각 헤드의 역할은 정해 주지 않아도 학습이 만든다.", size: 15 });
  }

  {
    const s = d.slide("멀티헤드 계산 과정: 쪼개고 → 각자 어텐션 → 이어 붙이기", { tag: "3교시 · 멀티헤드" });
    const flow = [["x\n(T, 16)", C.gray], ["Q, K, V\n(T, 16)", C.orangeL], ["4개로 쪼개기\n(4, T, 4)", C.tealL], ["헤드별\n어텐션", C.purpleL], ["이어 붙이기\n(T, 16)", C.greenL], ["· Wo\n(T, 16)", C.gray]];
    flow.forEach((f, i) => {
      const x = 0.6 + i * 2.07;
      d.box(s, f[0], { x, y: 2.0, w: 1.8, h: 1.2, fill: f[1], size: 14, bold: true });
      if (i < 5) d.arrow(s, x + 1.82, 2.6, x + 2.05, 2.6);
    });
    d.text(s, "예: d_model = 16, 헤드 4개 → d_k = 4", { x: 0.7, y: 3.5, w: 6, h: 0.4, size: 15, bold: true, color: C.teal });
    d.bullets(s, ["쪼개기: 열 방향으로 4조각 [0:4], [4:8], [8:12], [12:16]", "각 헤드는 서로 독립적으로 Step 3의 어텐션을 수행", "이어 붙이기(concat): 헤드 결과를 옆으로 붙여 다시 (T, 16)", "Wo: 헤드들의 결과를 섞어 주는 마지막 선형변환", "입력과 출력 shape이 같아서 층을 쌓을 수 있다"], { x: 0.7, y: 4.0, w: 12, h: 2.6, size: 15 });
  }

  {
    const s = d.slide("[코드 Step 4] 멀티헤드 어텐션 (NumPy)", { tag: "3교시 · 실습", notes: "파일: numpy_version/step4_multihead.py" });
    d.code(s, [
      "def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):",
      "    T, d_model = x.shape",
      "    d_k = d_model // n_heads            # 헤드 하나의 길이",
      "    Q, K, V = x @ Wq, x @ Wk, x @ Wv    # 각 (T, d_model)",
      "    mask = causal_mask(T) if causal else None",
      "",
      "    heads = []",
      "    for h in range(n_heads):",
      "        s = slice(h * d_k, (h + 1) * d_k)   # h번째 열 범위",
      "        out_h, _ = scaled_dot_product_attention(",
      "            Q[:, s], K[:, s], V[:, s], mask)",
      "        heads.append(out_h)             # 각 (T, d_k)",
      "",
      "    concat = np.concatenate(heads, axis=-1)  # (T, d_model)",
      "    return concat @ Wo                  # 헤드 섞기",
    ], { x: 0.6, y: 1.5, w: 8.2, h: 5.2, size: 12, title: "numpy_version/step4_multihead.py" });
    d.explain(s, [
      "d_k = d_model // n_heads : 나누어떨어져야 한다.",
      "Wq, Wk, Wv는 모든 헤드 몫을 한 행렬에 모아 둔 것(d_model × d_model).",
      "slice로 열을 잘라 헤드별로 어텐션 → Step 3 함수를 그대로 재사용.",
      "np.concatenate: 헤드 결과를 옆으로 이어 붙인다.",
      "검산: n_heads=1 이면 Step 3 결과 @ Wo 와 정확히 같다.",
    ], { x: 9.1, y: 1.5, w: 3.6, h: 5.2, size: 13 });
  }

  {
    const s = d.slide("피드포워드 네트워크(FFN): 각 단어가 혼자 생각하는 시간", { tag: "3교시 · 블록 부품" });
    d.text(s, "어텐션 = 단어끼리 \"회의\"하며 정보 교환  /  FFN = 각자 돌아가서 \"개인 정리\"", { x: 0.7, y: 1.45, w: 12, h: 0.5, size: 17, bold: true, color: C.teal });
    const f = [["d", C.gray, 1.2], ["4d (넓게)", C.orangeL, 2.6], ["ReLU/GELU", C.purpleL, 2.0], ["d (좁게)", C.gray, 1.2]];
    let x = 0.9;
    f.forEach((b, i) => {
      d.box(s, b[0], { x, y: 2.4, w: b[2], h: 1.0, fill: b[1], size: 16, bold: true });
      if (i < 3) d.arrow(s, x + b[2] + 0.02, 2.9, x + b[2] + 0.4, 2.9);
      x += b[2] + 0.45;
    });
    d.bullets(s, ["Linear(d → 4d)  →  비선형 함수  →  Linear(4d → d)", "모든 위치(토큰)에 똑같은 가중치를 따로따로 적용", "비선형 함수가 없으면 층을 아무리 쌓아도 하나의 선형 변환과 같다", "트랜스포머 파라미터의 절반 이상이 FFN에 있다"], { x: 0.9, y: 3.9, w: 7.0, h: 2.8, size: 15 });
    d.code(s, ["def __call__(self, x):", "    return self.l2(self.l1(x).relu())", "# l1: d -> 4d,  l2: 4d -> d"], { x: 8.2, y: 3.9, w: 4.5, h: 1.6, size: 12 });
  }

  {
    const s = d.slide("잔차연결(Residual): 지름길을 만들어 준다", { tag: "3교시 · 블록 부품" });
    // 도식: x -> [층] -> (+), 지름길 x -> (+)
    d.box(s, "x", { x: 0.9, y: 3.0, w: 1.0, h: 0.8, fill: C.gray, size: 20, bold: true });
    d.arrow(s, 1.95, 3.4, 3.2, 3.4);
    d.box(s, "층 f\n(어텐션/FFN)", { x: 3.3, y: 2.95, w: 2.3, h: 0.9, fill: C.tealL, size: 14, bold: true });
    d.arrow(s, 5.65, 3.4, 6.9, 3.4);
    s.addText("+", { shape: pres.shapes.OVAL, x: 7.0, y: 3.0, w: 0.8, h: 0.8, fill: { color: C.orangeL }, line: { color: C.orange, width: 1.5 }, fontFace: FONT, fontSize: 24, bold: true, color: C.orange, align: "center", valign: "middle", margin: 0 });
    d.arrow(s, 7.85, 3.4, 9.0, 3.4);
    d.box(s, "x + f(x)", { x: 9.1, y: 3.0, w: 1.9, h: 0.8, fill: C.greenL, size: 17, bold: true });
    s.addShape(pres.shapes.LINE, { x: 1.4, y: 2.2, w: 0, h: 0.75, line: { color: C.orange, width: 2.5 } });
    s.addShape(pres.shapes.LINE, { x: 1.4, y: 2.2, w: 6.0, h: 0, line: { color: C.orange, width: 2.5 } });
    d.arrow(s, 7.4, 2.2, 7.4, 2.98, C.orange, 2.5);
    d.text(s, "지름길 (원본을 그대로 전달)", { x: 3.0, y: 1.75, w: 4, h: 0.4, size: 14, bold: true, color: C.orange });
    d.card(s, { x: 0.7, y: 4.5, w: 5.9, h: 2.1, title: "왜 필요할까?", body: "층이 깊어지면 학습 신호(기울기)가 사라진다.\n지름길 덕분에 기울기가 곧장 아래층까지 전달된다.", size: 15 });
    d.card(s, { x: 6.8, y: 4.5, w: 5.9, h: 2.1, fill: C.greenL, title: "의미", tColor: C.green, body: "층은 \"원본을 통째로 바꾸는 것\"이 아니라\n\"원본에 보탤 수정분(f)\"만 학습하면 된다.", size: 15 });
  }

  {
    const s = d.slide("LayerNorm: 벡터의 크기를 가지런히", { tag: "3교시 · 블록 부품" });
    d.card(s, { x: 0.7, y: 1.5, w: 6.0, h: 1.8, fill: C.gray, title: "공식 (토큰 벡터 하나마다)", body: "y = γ · (x − 평균) / √(분산 + ε) + β\nγ, β는 학습되는 값 (처음엔 1, 0)", size: 15 });
    d.text(s, "손계산: x = [2, 4, 6, 8]", { x: 0.9, y: 3.6, w: 5, h: 0.4, size: 16, bold: true, color: C.teal });
    d.grid(s, [["x", "2", "4", "6", "8"], ["x − 5", "−3", "−1", "1", "3"], ["÷ √5", "−1.34", "−0.45", "0.45", "1.34"]], { x: 0.9, y: 4.1, cw: 1.1, ch: 0.55, size: 14, hi: (i, j) => (j === 0 ? C.gray : (i === 2 ? C.greenL : C.tealL)) });
    d.text(s, "평균 = 5, 분산 = (9+1+1+9)/4 = 5, 표준편차 = √5 ≈ 2.24", { x: 0.9, y: 6.0, w: 6.4, h: 0.4, size: 13, color: C.muted });
    d.card(s, { x: 7.1, y: 1.5, w: 5.6, h: 2.4, title: "왜 필요할까?", body: "층을 지나며 값이 커지거나 작아지면\n학습이 불안정해진다. 평균 0, 분산 1로\n맞추면 안정적이고 빨리 학습된다.", size: 15 });
    d.card(s, { x: 7.1, y: 4.1, w: 5.6, h: 2.5, fill: C.orangeL, title: "BatchNorm과 다른 점", tColor: C.orange, body: "LayerNorm은 \"한 토큰 벡터 안\"에서 계산\n(마지막 축).\n문장 길이·배치 크기에 영향받지 않아 언어모델에 적합하다.", size: 15 });
  }

  {
    const s = d.slide("트랜스포머 블록 조립", { tag: "3교시 · 블록", notes: "Pre-LN 구조. 원 논문은 Post-LN이지만 현대 모델 대부분이 Pre-LN을 쓴다." });
    const st = [["입력 x", C.paper], ["LayerNorm", C.gray], ["멀티헤드 어텐션 (마스크)", C.orangeL], ["+ 잔차  (x + ...)", C.greenL], ["LayerNorm", C.gray], ["FFN", C.purpleL], ["+ 잔차  (x + ...)", C.greenL], ["출력 x'  (shape 동일)", C.paper]];
    st.forEach((b, i) => {
      const y = 1.45 + i * 0.66;
      d.box(s, b[0], { x: 0.9, y, w: 4.8, h: 0.5, fill: b[1], size: 14, bold: i === 2 || i === 5 });
      if (i < 7) d.arrow(s, 3.3, y + 0.52, 3.3, y + 0.66, C.muted, 1.5);
    });
    d.card(s, { x: 6.3, y: 1.5, w: 6.4, h: 1.5, fill: C.gray, title: "두 줄이면 끝", body: "x = x + attn( ln1(x) )\nx = x + ffn( ln2(x) )", size: 16 });
    d.card(s, { x: 6.3, y: 3.2, w: 6.4, h: 1.6, title: "역할 분담", body: "어텐션: 단어 ↔ 단어 정보 교환\nFFN: 단어 하나를 깊게 가공\n잔차·정규화: 깊게 쌓아도 학습이 잘 되게", size: 14 });
    d.card(s, { x: 6.3, y: 5.0, w: 6.4, h: 1.6, fill: C.greenL, title: "블록은 레고", tColor: C.green, body: "입력·출력 shape이 같으므로 N개를 그대로 쌓을 수 있다. (GPT-3: 96개)", size: 14 });
  }

  {
    const s = d.slide("블록을 쌓고, 마지막에 \"다음 토큰\"을 예측", { tag: "3교시 · 블록" });
    const row = [["임베딩\n+ 위치", C.tealL], ["블록 1", C.orangeL], ["블록 2", C.orangeL], ["...", C.white], ["블록 N", C.orangeL], ["LayerNorm\n+ Linear", C.gray], ["softmax", C.greenL]];
    row.forEach((r, i) => {
      const x = 0.55 + i * 1.78;
      d.box(s, r[0], { x, y: 1.9, w: 1.55, h: 1.0, fill: r[1], size: 13, bold: true });
      if (i < 6) d.arrow(s, x + 1.56, 2.4, x + 1.76, 2.4, C.muted, 1.5);
    });
    d.text(s, "shape 흐름:  (B, T) → (B, T, d) → ... → (B, T, d) → (B, T, V)", { x: 0.7, y: 3.2, w: 12, h: 0.4, size: 16, bold: true, color: C.teal });
    d.card(s, { x: 0.7, y: 3.9, w: 5.9, h: 2.7, title: "출력층의 의미", body: "각 위치마다 어휘 V개의 점수(logits)를 낸다.\nsoftmax → \"다음 토큰이 각 글자일 확률\".\n\n예) \"the \" 다음 → 'm' 0.4, 's' 0.3, 'w' 0.1, ...", size: 14 });
    d.card(s, { x: 6.8, y: 3.9, w: 5.9, h: 2.7, fill: C.orangeL, title: "한 번에 T개를 학습", tColor: C.orange, body: "문장이 T글자라면 T개의 \"다음 글자 맞히기\" 문제가 한 번에 만들어진다. (마스크 덕분에 각 위치는 앞부분만 본다.)", size: 14 });
  }

  {
    const s = d.slide("1회차 마무리", { tag: "3교시 · 정리", notes: "다음 시간 준비: Python/NumPy 설치 확인, step1~4 파일 직접 실행해 보기. 퀴즈 정답: 1) 여러 관점 2) 학습 안정/기울기 전달 3) shape 동일해서 쌓기" });
    d.card(s, { x: 0.7, y: 1.5, w: 6.0, h: 3.2, fill: C.greenL, title: "오늘 배운 것", tColor: C.green, body: "• 토큰화 · 임베딩 · 위치 인코딩\n• Q, K, V 와 scaled dot-product attention\n• 인과 마스크\n• 멀티헤드, FFN, 잔차연결, LayerNorm\n• 트랜스포머 블록의 구조", size: 15 });
    d.card(s, { x: 7.0, y: 1.5, w: 5.7, h: 3.2, fill: C.orangeL, title: "복습 과제 (30분)", tColor: C.orange, body: "numpy_version/step1~4 를 차례로 실행하고\nshape 출력을 직접 확인해 보세요.\n\nstep3 의 가중치(Wq 등)를 바꾸면\n어텐션 그림이 어떻게 달라지나요?", size: 15 });
    d.card(s, { x: 0.7, y: 4.9, w: 12.0, h: 1.7, title: "다음 시간 (2회차)", body: "4교시 학습 원리와 자동미분 만들기  →  5교시 NumPy 트랜스포머 완성  →  6교시 PyTorch 버전과 실습 문제", size: 16 });
  }
};
