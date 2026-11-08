// 2회차 (4~6교시): 학습 원리, 자동미분, NumPy 조립, PyTorch, 실습 문제 + 솔루션
const path = require("path");
const { C, FONT, MONO, IMG } = require("./lib");

module.exports = function part2(d, pres) {
  const codeSlide = (title, tag, file, lines, items, o = {}) => {
    const s = d.slide(title, { tag, notes: o.notes || `파일: ${file}` });
    d.code(s, lines, { x: 0.6, y: 1.5, w: o.cw || 8.2, h: 5.25, size: o.size || 12, title: file });
    d.explain(s, items, { x: (o.cw || 8.2) + 0.9, y: 1.5, w: 12.7 - ((o.cw || 8.2) + 0.9), h: 5.25, size: o.esize || 13 });
    return s;
  };

  // ============================== 2회차 시작 ==============================
  {
    const s = d.slide("2회차 시작: 지난 시간 복습", { tag: "REVIEW" });
    const ks = [["토큰·임베딩", "글자 → 번호 → 벡터\n(T, d_model)", C.tealL, C.teal], ["위치 인코딩", "sin/cos 표를 더해\n순서 정보 추가", C.orangeL, C.orange], ["셀프 어텐션", "softmax(QKᵀ/√d)·V\n+ 인과 마스크", C.purpleL, C.purple], ["멀티헤드", "쪼개서 따로 보고\n이어 붙여 Wo로 섞기", C.greenL, C.green], ["블록", "x+attn(ln(x))\nx+ffn(ln(x))", C.gray, C.ink]];
    ks.forEach((k, i) => d.card(s, { x: 0.6 + i * 2.45, y: 1.6, w: 2.25, h: 2.4, fill: k[2], title: k[0], tColor: k[3], body: k[1], size: 13, tSize: 15 }));
    d.card(s, { x: 0.6, y: 4.3, w: 12.1, h: 2.3, fill: C.paper, title: "오늘의 목표: 부품들을 \"학습 가능한 모델\"로 만들기", body: "지금까지는 가중치가 랜덤이라 결과가 의미 없었다. 오늘은 ① 학습 원리  ② 기울기를 구하는 자동미분을 직접 만들고  ③ NumPy 트랜스포머를 완성해 실제로 글을 쓰게 한 뒤  ④ PyTorch로 같은 모델을 다시 만든다.", size: 15 });
  }

  // ============================== 4교시 ==============================
  d.section("4교시", "모델은 어떻게 배우나?", "다음 글자 맞히기 · 손실 · 경사하강법 · 역전파 · 자동미분", "4교시 시작. 학습 원리를 이해한 뒤 자동미분 엔진을 직접 만든다.");

  {
    const s = d.slide("학습 목표: \"다음 글자 맞히기\"", { tag: "4교시 · 학습 원리" });
    d.text(s, "정답은 문장 안에 이미 있다! 입력을 한 칸 밀면 곧바로 정답지가 된다. (자기지도학습)", { x: 0.7, y: 1.45, w: 12, h: 0.5, size: 17, bold: true, color: C.teal });
    const inp = ["a", "t", "t", "e", "n", "t"], tgt = ["t", "t", "e", "n", "t", "i"];
    d.text(s, "입력 x", { x: 0.7, y: 2.4, w: 1.4, h: 0.5, size: 16, bold: true });
    d.text(s, "정답 y", { x: 0.7, y: 3.9, w: 1.4, h: 0.5, size: 16, bold: true, color: C.orange });
    inp.forEach((c, i) => {
      d.box(s, c, { x: 2.2 + i * 1.0, y: 2.3, w: 0.85, h: 0.7, fill: C.tealL, size: 20, bold: true });
      d.arrow(s, 2.62 + i * 1.0, 3.05, 2.62 + i * 1.0, 3.85, C.muted, 1.2);
      d.box(s, tgt[i], { x: 2.2 + i * 1.0, y: 3.9, w: 0.85, h: 0.7, fill: C.orangeL, size: 20, bold: true });
    });
    d.card(s, { x: 0.7, y: 4.95, w: 5.8, h: 1.7, title: "한 문장 = 여러 문제", body: "\"a\" 다음은 \"t\", \"at\" 다음은 \"t\",\n\"att\" 다음은 \"e\" ... 6문제를 한 번에.", size: 14 });
    d.code(s, ["x = data[s : s+CTX]            # 입력 구간", "y = data[s+1 : s+CTX+1]      # 한 칸 뒤 = 정답"], { x: 6.8, y: 4.95, w: 5.9, h: 1.7, size: 12 });
  }

  {
    const s = d.slide("손실 함수: 틀린 정도를 숫자 하나로", { tag: "4교시 · 학습 원리" });
    d.card(s, { x: 0.7, y: 1.5, w: 6.2, h: 1.7, fill: C.gray, title: "교차 엔트로피(Cross Entropy)", body: "loss = − log( 정답 글자에 준 확률 )\n여러 위치의 평균을 최종 loss로 쓴다.", size: 16 });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t) => ({ text: t, options: { align: "center" } });
    s.addTable([[H("정답에 준 확률"), H("loss"), H("의미")], [c("0.90"), c("0.105"), c("거의 맞힘")], [c("0.50"), c("0.693"), c("반반")], [c("0.10"), c("2.303"), c("많이 틀림")], [c("0.01"), c("4.605"), c("완전 틀림")]], { x: 0.7, y: 3.5, w: 6.2, colW: [2.2, 1.5, 2.5], fontFace: FONT, fontSize: 15, color: C.ink, rowH: 0.5, border: { type: "solid", color: C.line, pt: 1 }, valign: "middle" });
    d.card(s, { x: 7.3, y: 1.5, w: 5.4, h: 2.4, fill: C.greenL, title: "학습 전 loss는 얼마일까?", tColor: C.green, body: "아무것도 모르면 V개를 똑같이 찍는다.\nloss = ln(V) = ln(28) ≈ 3.33\n우리 실험의 첫 loss도 3.5 근처에서 출발!", size: 14 });
    d.card(s, { x: 7.3, y: 4.1, w: 5.4, h: 2.5, title: "목표", body: "loss가 0에 가까워지도록\n모든 가중치를 조금씩 조정한다.\n\n그러려면 \"어느 가중치를 어느 방향으로\n얼마나\" 바꿀지 알아야 한다 → 기울기", size: 14 });
  }

  {
    const s = d.slide("경사하강법: 안개 낀 산을 내려오기", { tag: "4교시 · 학습 원리" });
    d.card(s, { x: 0.7, y: 1.5, w: 5.9, h: 1.8, fill: C.gray, title: "한 줄 규칙", body: "w ← w − lr × (∂loss / ∂w)\n\n기울기(∂loss/∂w)의 반대 방향으로 조금 이동", size: 16 });
    d.bullets(s, ["기울기: 내 위치에서 \"가장 가파르게 올라가는 방향\"", "그 반대로 가면 loss가 줄어든다", "lr(학습률)이 너무 크면 튕겨 나가고, 너무 작으면 느리다", "Adam: 이동 방향·보폭을 과거 기울기의 평균으로 부드럽게 조절해 주는 똑똑한 버전"], { x: 0.7, y: 3.5, w: 5.9, h: 3.1, size: 14 });
    // 곡선 도식: 점들로 포물선
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 1.5, w: 5.7, h: 5.1, fill: { color: C.paper }, line: { color: C.line, width: 1 }, rectRadius: 0.1 });
    const pts = [];
    for (let k = -5; k <= 5; k++) pts.push([9.85 + k * 0.45, 2.0 + (25 - k * k) * 0.14]);
    pts.forEach((p) => s.addShape(pres.shapes.OVAL, { x: p[0] - 0.05, y: p[1] - 0.05, w: 0.1, h: 0.1, fill: { color: C.teal }, line: { color: C.teal, width: 0 } }));
    const walk = [[-4, C.orange], [-2.7, C.orange], [-1.6, C.orange], [-0.7, C.orange], [0, C.green]];
    walk.forEach((p, i) => {
      const px = 9.85 + p[0] * 0.45, py = 2.0 + (25 - p[0] * p[0]) * 0.14;
      s.addShape(pres.shapes.OVAL, { x: px - 0.13, y: py - 0.13, w: 0.26, h: 0.26, fill: { color: p[1] }, line: { color: C.white, width: 1 } });
    });
    d.text(s, "시작(loss 큼)", { x: 7.15, y: 2.7, w: 1.6, h: 0.3, size: 12, bold: true, color: C.orange });
    d.text(s, "목표(loss 작음)", { x: 9.0, y: 5.75, w: 1.9, h: 0.3, size: 12, bold: true, color: C.green });
    d.text(s, "한 걸음마다 기울기 반대 방향으로", { x: 7.3, y: 6.15, w: 5.2, h: 0.35, size: 13, color: C.muted });
  }

  {
    const s = d.slide("역전파와 연쇄법칙: 결과에서 거꾸로 기울기 전달", { tag: "4교시 · 학습 원리" });
    d.text(s, "예: L = (a·b + a).sum()   with a = [1,2,3], b = [4,5,6]", { x: 0.7, y: 1.45, w: 12, h: 0.45, size: 16, bold: true });
    d.box(s, "a", { x: 0.9, y: 2.6, w: 0.9, h: 0.7, fill: C.tealL, size: 18, bold: true });
    d.box(s, "b", { x: 0.9, y: 4.1, w: 0.9, h: 0.7, fill: C.tealL, size: 18, bold: true });
    d.arrow(s, 1.85, 2.95, 3.3, 3.2);
    d.arrow(s, 1.85, 4.45, 3.3, 3.7);
    d.box(s, "u = a·b", { x: 3.4, y: 3.0, w: 1.7, h: 0.8, fill: C.orangeL, size: 15, bold: true });
    d.arrow(s, 5.15, 3.4, 6.4, 3.4);
    d.box(s, "v = u + a", { x: 6.5, y: 3.0, w: 1.7, h: 0.8, fill: C.orangeL, size: 15, bold: true });
    d.arrow(s, 8.25, 3.4, 9.4, 3.4);
    d.box(s, "L = Σ v", { x: 9.5, y: 3.0, w: 1.6, h: 0.8, fill: C.greenL, size: 15, bold: true });
    d.arrow(s, 1.85, 2.8, 6.5, 3.05, C.line, 1.2);
    d.card(s, { x: 0.7, y: 5.1, w: 5.9, h: 1.6, fill: C.gray, title: "연쇄법칙(Chain Rule)", body: "∂L/∂a = (∂L/∂v)·(∂v/∂u)·(∂u/∂a) + (∂L/∂v)·(∂v/∂a)\n= 1·b + 1 = b + 1 = [5, 6, 7]", size: 14 });
    d.card(s, { x: 6.8, y: 5.1, w: 5.9, h: 1.6, fill: C.greenL, title: "핵심 아이디어", tColor: C.green, body: "각 연산은 \"자기 국소 미분\"만 알면 된다.\n나머지는 뒤에서 오는 기울기에 곱해서 전달!", size: 14 });
  }

  {
    const s = d.slide("자동미분 엔진 설계: Tensor가 기억하는 3가지", { tag: "4교시 · 자동미분" });
    d.card(s, { x: 0.7, y: 1.6, w: 3.8, h: 2.8, fill: C.tealL, title: "① data / grad", tColor: C.teal, body: "data: 실제 값(NumPy 배열)\ngrad: ∂L/∂(이 값)\n처음엔 0, backward 때 채워진다.", size: 14 });
    d.card(s, { x: 4.7, y: 1.6, w: 3.8, h: 2.8, fill: C.orangeL, title: "② parents", tColor: C.orange, body: "이 값을 만든 입력들.\n계산 그래프를 거슬러 올라갈\n\"길\" 역할을 한다.", size: 14 });
    d.card(s, { x: 8.7, y: 1.6, w: 4.0, h: 2.8, fill: C.purpleL, title: "③ _backward()", tColor: C.purple, body: "\"내 기울기를 입력들에게\n나눠 주는 방법\"을 담은 함수.\n연산마다 다르다.", size: 14 });
    d.card(s, { x: 0.7, y: 4.7, w: 12.0, h: 1.9, fill: C.gray, title: "backward()가 하는 일", body: "① 그래프를 위상 정렬(입력 먼저)  ② loss.grad = 1 로 시작  ③ 결과 → 입력 순서로 각 노드의 _backward() 실행.\n※ 같은 값이 여러 곳에 쓰이면 기울기를 더한다(+=).", size: 15 });
  }

  codeSlide("[코드 Step 5-1] Tensor 클래스: 덧셈과 곱셈", "4교시 · 실습", "numpy_version/step5_autograd.py", [
    "class Tensor:",
    "    def __init__(self, data, parents=()):",
    "        self.data = np.asarray(data, dtype=np.float64)",
    "        self.grad = np.zeros_like(self.data)",
    "        self.parents = parents",
    "        self._backward = lambda: None",
    "",
    "    def __mul__(self, other):",
    "        other = Tensor.wrap(other)",
    "        out = Tensor(self.data * other.data, (self, other))",
    "        def _backward():            # d(a*b)/da = b",
    "            self.grad += unbroadcast(out.grad * other.data,",
    "                                     self.data.shape)",
    "            other.grad += unbroadcast(out.grad * self.data,",
    "                                      other.data.shape)",
    "        out._backward = _backward",
    "        return out",
  ], [
    "계산 결과 out 은 자신을 만든 입력(self, other)을 기억한다.",
    "_backward 는 클로저: out.grad 를 받아 입력들의 grad 에 더해 준다.",
    "곱셈의 국소 미분: a의 기울기 = (나온 기울기) × b, b의 기울기 = × a.",
    "+= : 한 값이 여러 번 쓰였다면 기울기가 합쳐져야 하기 때문.",
    "unbroadcast: 브로드캐스트로 커진 기울기를 원래 shape 으로 합쳐 줌.",
  ]);

  codeSlide("[코드 Step 5-2] 행렬곱, softmax 그리고 backward()", "4교시 · 실습", "numpy_version/step5_autograd.py", [
    "def __matmul__(self, other):          # C = A @ B",
    "    out = Tensor(self.data @ other.data, (self, other))",
    "    def _backward():",
    "        self.grad += unbroadcast(out.grad @ swap(other.data), ...)",
    "        other.grad += unbroadcast(swap(self.data) @ out.grad, ...)",
    "    ...",
    "",
    "def softmax(self):",
    "    ...                                # s = 확률",
    "    def _backward():                   # dx = s*(dy - sum(dy*s))",
    "        g = out.grad",
    "        self.grad += s * (g - (g * s).sum(-1, keepdims=True))",
    "",
    "def backward(self):",
    "    order = [...]                      # 위상 정렬 (입력이 먼저)",
    "    self.grad = np.ones_like(self.data)  # dL/dL = 1",
    "    for t in reversed(order):          # 결과 -> 입력",
    "        t._backward()",
  ], [
    "행렬곱: dA = dC·Bᵀ,  dB = Aᵀ·dC  (swap = 마지막 두 축 교환).",
    "softmax 역전파는 위 공식 한 줄이면 충분하다.",
    "그 외 add, mul, pow, sum, reshape, transpose, relu, masked_fill, take(임베딩), cross_entropy 를 같은 방식으로 구현.",
    "backward(): 위상 정렬 후 reversed 순서로 _backward 호출.",
    "전체 파일은 약 200줄. PyTorch autograd 의 축소판이다!",
  ], { size: 11 });

  {
    const s = d.slide("기울기 검사(Gradient Check): 내가 만든 미분이 맞나?", { tag: "4교시 · 자동미분" });
    d.card(s, { x: 0.7, y: 1.5, w: 5.9, h: 2.4, fill: C.gray, title: "수치 미분", body: "f'(w) ≈ [ f(w+h) − f(w−h) ] / 2h\n\nh를 아주 작게(1e-6) 잡고 값을 살짝 흔들어서\n실제로 loss가 얼마나 변하는지 측정", size: 15 });
    d.code(s, [
      "loss = loss_fn(); loss.backward()    # 내가 만든 기울기",
      "num = np.zeros_like(W.data)",
      "for i in np.ndindex(*W.data.shape):",
      "    old = W.data[i]",
      "    W.data[i] = old + h; lp = loss_fn().data",
      "    W.data[i] = old - h; lm = loss_fn().data",
      "    W.data[i] = old",
      "    num[i] = (lp - lm) / (2 * h)     # 수치 미분",
      "assert np.allclose(num, W.grad, atol=1e-6)",
    ], { x: 0.7, y: 4.1, w: 7.4, h: 2.55, size: 11 });
    d.card(s, { x: 6.9, y: 1.5, w: 5.8, h: 2.4, fill: C.greenL, title: "실행 결과", tColor: C.green, body: "최대 오차 ≈ 1.4e-10\n기울기 검사 통과!\n\n오차가 1e-6 보다 작으면 올바르게 구현된 것", size: 15 });
    d.card(s, { x: 8.4, y: 4.1, w: 4.3, h: 2.55, fill: C.orangeL, title: "실무 팁", tColor: C.orange, body: "새 연산을 만들 때마다\n기울기 검사를 하는 습관!\n(느려서 학습엔 쓰지 않는다)", size: 14 });
  }

  // ============================== 5교시 ==============================
  d.section("5교시", "NumPy 트랜스포머 완성", "Step 6: 모든 부품을 조립해 학습시키고 글을 쓰게 한다", "5교시 시작. 오늘의 하이라이트: 프레임워크 없이 직접 만든 GPT가 글을 쓴다.");

  {
    const s = d.slide("조립 설계도: 클래스 하나 = 부품 하나", { tag: "5교시 · NumPy 조립" });
    const cls = [["Linear", "y = xW + b"], ["LayerNorm", "정규화 (γ, β)"], ["MultiHeadAttention", "q,k,v,o 4개의 Linear"], ["FeedForward", "Linear 2개 + ReLU"], ["Block", "ln1·attn·ln2·ffn"], ["GPT", "emb + Block×N + head"], ["Adam", "가중치 갱신"]];
    cls.forEach((c, i) => {
      const col = i < 4 ? 0 : 1, row = i < 4 ? i : i - 4;
      d.box(s, c[0], { x: 0.7 + col * 4.7, y: 1.6 + row * 1.2, w: 2.4, h: 0.85, fill: [C.tealL, C.tealL, C.orangeL, C.orangeL, C.purpleL, C.greenL, C.gray][i], size: 15, bold: true });
      d.text(s, c[1], { x: 3.2 + col * 4.7, y: 1.7 + row * 1.2, w: 1.95, h: 0.7, size: 13, color: C.muted });
    });
    d.card(s, { x: 10.2, y: 1.6, w: 2.5, h: 4.9, title: "공통 약속", body: "• __call__(x): 순전파\n• params(): 학습할 Tensor 목록\n\n모든 부품이 같은 규칙을 가지므로 조립이 쉽고, Adam은 params() 목록만 받으면 된다.", size: 13 });
    d.text(s, "아래층(Linear)부터 위층(GPT)으로 한 단계씩 쌓아 올린다.", { x: 0.7, y: 6.35, w: 8, h: 0.4, size: 14, color: C.muted });
  }

  codeSlide("[코드 Step 6-1] Linear 와 LayerNorm", "5교시 · 실습", "numpy_version/step6_transformer.py", [
    "class Linear:",
    "    def __init__(self, n_in, n_out, bias=True):",
    "        self.W = Tensor(rng.normal(0, 1/np.sqrt(n_in),",
    "                                   size=(n_in, n_out)))",
    "        self.b = Tensor(np.zeros(n_out)) if bias else None",
    "    def __call__(self, x):",
    "        y = x @ self.W",
    "        return y + self.b if self.b is not None else y",
    "",
    "class LayerNorm:",
    "    def __init__(self, d, eps=1e-5):",
    "        self.gamma, self.beta = Tensor(np.ones(d)), Tensor(np.zeros(d))",
    "        self.eps = eps",
    "    def __call__(self, x):",
    "        mu = x.mean(axis=-1, keepdims=True)",
    "        xc = x - mu",
    "        var = (xc * xc).mean(axis=-1, keepdims=True)",
    "        return xc * ((var + self.eps) ** -0.5) * self.gamma + self.beta",
  ], [
    "초기값 1/√n_in : 입력이 많을수록 작게 → 출력이 폭주하지 않는다.",
    "x @ W : Tensor의 @ 가 Step 5의 matmul(역전파 포함)을 부른다.",
    "LayerNorm 은 새 연산을 만들지 않고 mean, −, ×, ** 만 조합했다.",
    "조합만 해도 자동미분이 알아서 기울기를 구해 준다!",
  ], { size: 11, cw: 8.4 });

  codeSlide("[코드 Step 6-2] 멀티헤드 어텐션 (배치 + 자동미분)", "5교시 · 실습", "numpy_version/step6_transformer.py", [
    "class MultiHeadAttention:",
    "    def __init__(self, d_model, n_heads):",
    "        self.h, self.dk = n_heads, d_model // n_heads",
    "        self.q, self.k, self.v, self.o = (Linear(d_model, d_model) for _ in range(4))",
    "",
    "    def __call__(self, x):                    # x: (B, T, D)",
    "        B, T, D = x.data.shape",
    "        def split(t):                         # (B,T,D)->(B,h,T,dk)",
    "            return t.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)",
    "        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))",
    "        scores = (q @ k.transpose(0,1,3,2)) * (1.0/np.sqrt(self.dk))",
    "        mask = np.triu(np.ones((T, T), dtype=bool), k=1)",
    "        w = scores.masked_fill(mask, -1e9).softmax()",
    "        ctx = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, D)",
    "        return self.o(ctx)",
  ], [
    "Step 4의 for 문 대신 헤드를 \"새 축\"으로 만들어 한 번에 계산한다.",
    "reshape → transpose 로 (B, h, T, dk) 모양을 만든다.",
    "q @ kᵀ 는 (B, h, T, T): 모든 문장·헤드의 점수를 한 번에.",
    "masked_fill → softmax 순서가 Step 3과 같다.",
    "다시 transpose → reshape 으로 헤드를 이어 붙이고 o 로 섞는다.",
  ], { size: 11, cw: 8.4 });

  codeSlide("[코드 Step 6-3] FeedForward 와 Block", "5교시 · 실습", "numpy_version/step6_transformer.py", [
    "class FeedForward:",
    "    def __init__(self, d_model):",
    "        self.l1 = Linear(d_model, 4 * d_model)",
    "        self.l2 = Linear(4 * d_model, d_model)",
    "    def __call__(self, x):",
    "        return self.l2(self.l1(x).relu())",
    "",
    "class Block:",
    "    def __init__(self, d_model, n_heads):",
    "        self.ln1 = LayerNorm(d_model)",
    "        self.attn = MultiHeadAttention(d_model, n_heads)",
    "        self.ln2 = LayerNorm(d_model)",
    "        self.ffn = FeedForward(d_model)",
    "    def __call__(self, x):",
    "        x = x + self.attn(self.ln1(x))   # 잔차연결",
    "        x = x + self.ffn(self.ln2(x))",
    "        return x",
  ], [
    "FFN : 4배로 넓혔다가 ReLU 후 원래 크기로.",
    "Block : 슬라이드에서 본 두 줄이 그대로 코드가 되었다.",
    "x + ... 의 덧셈이 잔차연결. 기울기가 지름길로도 흐른다.",
    "입력·출력 shape이 같아서 리스트에 담아 반복 호출할 수 있다.",
  ], { size: 11 });

  codeSlide("[코드 Step 6-4] GPT 모델 전체", "5교시 · 실습", "numpy_version/step6_transformer.py", [
    "class GPT:",
    "    def __init__(self, vocab_size, d_model=32, n_heads=4,",
    "                 n_layers=2, max_len=64):",
    "        self.emb = Tensor(rng.normal(0, 0.1, (vocab_size, d_model)))",
    "        self.pe = positional_encoding(max_len, d_model)  # 상수",
    "        self.blocks = [Block(d_model, n_heads) for _ in range(n_layers)]",
    "        self.ln_f = LayerNorm(d_model)",
    "        self.head = Linear(d_model, vocab_size)",
    "        self.scale = np.sqrt(d_model)",
    "",
    "    def __call__(self, idx):                 # idx: (B, T) 정수",
    "        B, T = idx.shape",
    "        x = self.emb.take(idx) * self.scale + self.pe[:T]",
    "        for blk in self.blocks:",
    "            x = blk(x)",
    "        return self.head(self.ln_f(x))       # (B, T, V) 점수",
  ], [
    "emb.take(idx): 번호로 임베딩 표에서 벡터 조회 (기울기는 해당 행에만).",
    "√d_model 을 곱해 위치 인코딩에 임베딩이 묻히지 않게 한다.",
    "+ self.pe[:T] : Step 2 의 위치 인코딩을 더한다 (학습 X).",
    "블록을 N번 통과 → ln_f → head 로 다음 글자 점수를 낸다.",
  ], { size: 11, cw: 8.4 });

  codeSlide("[코드 Step 6-5] Adam 옵티마이저와 학습 루프", "5교시 · 실습", "numpy_version/step6_transformer.py", [
    "class Adam:",
    "    def step(self):",
    "        self.t += 1",
    "        for i, p in enumerate(self.p):",
    "            self.m[i] = b1*self.m[i] + (1-b1)*p.grad       # 기울기 평균",
    "            self.v[i] = b2*self.v[i] + (1-b2)*p.grad**2    # 크기 평균",
    "            mh = self.m[i] / (1 - b1**self.t)              # 초반 보정",
    "            vh = self.v[i] / (1 - b2**self.t)",
    "            p.data -= self.lr * mh / (np.sqrt(vh) + eps)",
    "",
    "for step in range(1, STEPS + 1):",
    "    x, y = (랜덤 위치에서 CTX 글자 / 한 칸 뒤 글자)",
    "    logits = model(x)                                   # 1 순전파",
    "    loss = logits.reshape(-1, V).cross_entropy(y.reshape(-1))  # 2 손실",
    "    opt.zero_grad()                                     # 3 기울기 비우기",
    "    loss.backward()                                     # 4 역전파",
    "    opt.step()                                          # 5 갱신",
  ], [
    "Adam: 과거 기울기의 평균(m)과 크기(v)로 파라미터마다 보폭을 조절.",
    "학습 루프는 5단계 리듬: 순전파 → 손실 → 비우기 → 역전파 → 갱신.",
    "zero_grad 를 빼먹으면 기울기가 계속 누적된다(자주 하는 실수!).",
    "이 5단계는 PyTorch 에서도 똑같다.",
  ], { size: 11, cw: 8.4 });

  {
    const s = d.slide("학습 결과: loss가 내려가고 글을 쓴다", { tag: "5교시 · 결과", notes: "NumPy 버전 실제 실행 결과(파라미터 27,292개, 600 step, 약 7초)." });
    s.addChart(pres.charts.LINE, [{ name: "NumPy 버전", labels: ["1", "100", "200", "300", "400", "500", "600"], values: [3.724, 0.8279, 0.2364, 0.1784, 0.1383, 0.1137, 0.1084] }], {
      x: 0.6, y: 1.5, w: 6.4, h: 4.5, chartColors: [C.teal], showLegend: false, showTitle: true, title: "step 별 loss (NumPy, 27,292 파라미터)", titleFontFace: FONT, titleFontSize: 14, titleColor: C.ink,
      showValue: true, dataLabelFontSize: 11, dataLabelColor: C.ink, dataLabelPosition: "t", dataLabelFormatCode: "0.00",
      catAxisLabelColor: C.muted, valAxisLabelColor: C.muted, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
      lineSize: 3, lineDataSymbolSize: 8, valAxisMinVal: 0, valAxisMaxVal: 4,
      showCatAxisTitle: true, catAxisTitle: "step", catAxisTitleColor: C.muted, catAxisTitleFontSize: 11,
    });
    d.card(s, { x: 7.3, y: 1.5, w: 5.4, h: 3.0, fill: C.paper, title: "생성 결과 (일부)", body: "the sentence in many ways.\nlayers stack up and the meaning\ngrows richer.\nlittle by little the model learns\nto predict the ne...", size: 14 });
    d.card(s, { x: 7.3, y: 4.7, w: 5.4, h: 1.9, fill: C.greenL, title: "해석", tColor: C.green, body: "학습 글(약 300자)을 거의 외웠다. 의미 있는 글을 쓰려면 더 많은 데이터가 필요하다.", size: 13 });
    d.text(s, "3.72 → 0.11 : 랜덤이던 가중치가 \"규칙\"을 학습한 증거", { x: 0.6, y: 6.2, w: 6.4, h: 0.4, size: 14, bold: true, color: C.teal });
  }

  {
    const s = d.slide("모델 크기와 파라미터 수 세어 보기", { tag: "5교시 · 결과" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t, a = "center") => ({ text: t, options: { align: a } });
    s.addTable([
      [H("부품 (d=32, V=28)"), H("계산식"), H("개수")],
      [c("임베딩", "left"), c("V × d = 28 × 32"), c("896")],
      [c("어텐션 (q,k,v,o)", "left"), c("4 × (32×32 + 32)"), c("4,224")],
      [c("FFN", "left"), c("(32×128+128) + (128×32+32)"), c("8,352")],
      [c("LayerNorm ×2", "left"), c("2 × (32+32)"), c("128")],
      [c("블록 1개 합계", "left"), c(""), c("12,704")],
      [c("블록 2개 + ln_f + head", "left"), c("2×12,704 + 64 + 924"), c("26,396")],
      [c("전체", "left"), c("896 + 26,396"), c("27,292")],
    ], { x: 0.7, y: 1.5, w: 7.6, colW: [2.9, 3.4, 1.3], fontFace: FONT, fontSize: 14, color: C.ink, rowH: 0.5, border: { type: "solid", color: C.line, pt: 1 }, valign: "middle" });
    d.card(s, { x: 8.6, y: 1.5, w: 4.1, h: 2.4, fill: C.greenL, title: "코드와 일치!", tColor: C.green, body: "step6 실행 시 출력되는\n\"파라미터 수: 27292\"와 같다.", size: 14 });
    d.card(s, { x: 8.6, y: 4.1, w: 4.1, h: 2.5, title: "규모 비교", body: "우리 모델: 약 2.7만 개\nGPT-2 small: 약 1.2억 개\nGPT-3: 약 1,750억 개\n\n구조는 같고 크기만 다르다.", size: 14 });
  }

  // ============================== 6교시 ==============================
  d.section("6교시", "PyTorch로 실전 버전", "같은 모델을 현실적인 방식으로: nn.Module · 자동미분 · GPU", "6교시 시작. NumPy에서 직접 한 일을 PyTorch가 대신해 주는 부분을 확인한다.");

  {
    const s = d.slide("NumPy 버전 vs PyTorch 버전", { tag: "6교시 · PyTorch" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t) => ({ text: t, options: { align: "left" } });
    s.addTable([
      [H("항목"), H("NumPy (직접 구현)"), H("PyTorch (실전)")],
      [c("미분"), c("Tensor 클래스 + _backward 직접 작성"), c("loss.backward() 한 줄 (autograd)")],
      [c("부품"), c("Linear, LayerNorm 직접 구현"), c("nn.Linear, nn.LayerNorm, nn.Embedding")],
      [c("옵티마이저"), c("Adam 직접 구현"), c("torch.optim.AdamW")],
      [c("속도"), c("CPU, 작은 모델만 가능"), c("GPU 가속, 큰 모델 가능")],
      [c("장점"), c("동작 원리를 100% 이해"), c("짧고 빠르고 검증됨, 업계 표준")],
    ], { x: 0.7, y: 1.5, w: 12.0, colW: [2.0, 5.0, 5.0], fontFace: FONT, fontSize: 15, color: C.ink, rowH: 0.65, border: { type: "solid", color: C.line, pt: 1 }, valign: "middle" });
    d.card(s, { x: 0.7, y: 5.6, w: 12.0, h: 1.1, fill: C.greenL, body: "원리를 이해한 뒤에 도구를 쓰면, PyTorch의 모든 줄이 \"아, 이거 우리가 직접 만든 거!\"로 읽힌다.", size: 16 });
  }

  {
    const s = d.slide("우리가 만든 것 ↔ PyTorch 부품 대응표", { tag: "6교시 · PyTorch" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t) => ({ text: t, options: { align: "left", fontFace: MONO } });
    s.addTable([
      [H("NumPy 버전"), H("PyTorch 버전")],
      [c("E[ids]  (임베딩 조회)"), c("nn.Embedding(V, d)")],
      [c("positional_encoding()"), c("register_buffer(\"pe\", pe)")],
      [c("x @ W + b"), c("nn.Linear(d_in, d_out)")],
      [c("LayerNorm 클래스"), c("nn.LayerNorm(d)")],
      [c("masked_fill(mask, -1e9)"), c("masked_fill(mask, float('-inf'))")],
      [c("Tensor.backward()"), c("loss.backward()")],
      [c("Adam 클래스"), c("torch.optim.AdamW(model.parameters())")],
      [c("params() 목록"), c("model.parameters() (자동 수집)")],
    ], { x: 0.7, y: 1.5, w: 12.0, colW: [5.6, 6.4], fontFace: FONT, fontSize: 14, color: C.ink, rowH: 0.52, border: { type: "solid", color: C.line, pt: 1 }, valign: "middle" });
  }

  codeSlide("[PyTorch Step 1] 임베딩 + 위치 인코딩", "6교시 · 실습", "pytorch_version/step1_embedding_pe.py", [
    "class TokenAndPosition(nn.Module):",
    "    def __init__(self, vocab_size, d_model, max_len):",
    "        super().__init__()",
    "        self.d_model = d_model",
    "        self.tok = nn.Embedding(vocab_size, d_model)",
    "        pos = torch.arange(max_len).unsqueeze(1)",
    "        div = torch.exp(torch.arange(0, d_model, 2)",
    "                        * (-math.log(10000.0) / d_model))",
    "        pe = torch.zeros(max_len, d_model)",
    "        pe[:, 0::2] = torch.sin(pos * div)",
    "        pe[:, 1::2] = torch.cos(pos * div)",
    "        self.register_buffer(\"pe\", pe)   # 학습 안 하는 상수",
    "",
    "    def forward(self, idx):                # idx: (B, T)",
    "        T = idx.shape[1]",
    "        x = self.tok(idx) * math.sqrt(self.d_model)",
    "        return x + self.pe[:T]             # (B, T, d_model)",
  ], [
    "nn.Module 을 상속하면 안에 만든 파라미터를 PyTorch가 자동으로 모아 준다.",
    "nn.Embedding 이 NumPy 의 E[ids] 조회를 대신한다.",
    "register_buffer: 파라미터는 아니지만 모델과 함께 GPU 이동·저장되는 상수.",
    "(T, d) 를 (B, T, d) 에 더하면 B 방향으로 자동 브로드캐스트.",
  ], { size: 11, cw: 8.4 });

  codeSlide("[PyTorch Step 2~3] 멀티헤드 어텐션", "6교시 · 실습", "pytorch_version/step3_multihead.py", [
    "class MultiHeadAttention(nn.Module):",
    "    def __init__(self, d_model, n_heads, max_len, dropout=0.0):",
    "        super().__init__()",
    "        self.h, self.dk = n_heads, d_model // n_heads",
    "        self.q = nn.Linear(d_model, d_model, bias=False)",
    "        self.k = nn.Linear(d_model, d_model, bias=False)",
    "        self.v = nn.Linear(d_model, d_model, bias=False)",
    "        self.o = nn.Linear(d_model, d_model)",
    "        self.drop = nn.Dropout(dropout)",
    "        self.register_buffer(\"causal\", torch.tril(torch.ones(max_len, max_len)))",
    "",
    "    def forward(self, x):                        # (B, T, D)",
    "        B, T, D = x.shape",
    "        split = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)",
    "        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))",
    "        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)",
    "        scores = scores.masked_fill(self.causal[:T, :T] == 0, float(\"-inf\"))",
    "        w = self.drop(F.softmax(scores, dim=-1))",
    "        ctx = (w @ v).transpose(1, 2).contiguous().view(B, T, D)",
    "        return self.o(ctx)",
  ], [
    "NumPy 버전과 줄 단위로 같은 흐름이다. 달라진 건 도구뿐!",
    "tril: 아래 삼각형이 1인 행렬 → 0인 칸(미래)을 −inf 로 가린다.",
    "view + transpose 로 (B, h, T, dk) 를 만들어 모든 헤드를 병렬 계산.",
    "contiguous(): transpose 후 메모리를 정리해야 view 를 쓸 수 있다.",
    "dropout: 학습 중 일부 연결을 꺼서 과적합을 막는다.",
  ], { size: 10.5, cw: 8.9, esize: 12 });

  codeSlide("[PyTorch Step 4] 트랜스포머 블록", "6교시 · 실습", "pytorch_version/step4_block.py", [
    "class FeedForward(nn.Module):",
    "    def __init__(self, d_model, dropout=0.0):",
    "        super().__init__()",
    "        self.net = nn.Sequential(",
    "            nn.Linear(d_model, 4 * d_model),",
    "            nn.GELU(),                      # 부드러운 ReLU",
    "            nn.Linear(4 * d_model, d_model),",
    "            nn.Dropout(dropout))",
    "    def forward(self, x): return self.net(x)",
    "",
    "class Block(nn.Module):",
    "    def __init__(self, d_model, n_heads, max_len, dropout=0.0):",
    "        super().__init__()",
    "        self.ln1 = nn.LayerNorm(d_model)",
    "        self.attn = MultiHeadAttention(d_model, n_heads, max_len, dropout)",
    "        self.ln2 = nn.LayerNorm(d_model)",
    "        self.ffn = FeedForward(d_model, dropout)",
    "    def forward(self, x):",
    "        x = x + self.attn(self.ln1(x))",
    "        x = x + self.ffn(self.ln2(x))",
    "        return x",
  ], [
    "nn.Sequential: 층을 순서대로 이어 붙이는 상자.",
    "GELU: ReLU 와 비슷하지만 곡선이 부드러워 GPT 계열이 즐겨 쓴다.",
    "Block.forward 의 두 줄은 NumPy 버전과 완전히 동일하다.",
    "실행 결과: 블록 하나의 파라미터 12,608개 (d=32, bias 차이로 NumPy 와 약간 다름).",
  ], { size: 10.5, cw: 8.4, esize: 12 });

  codeSlide("[PyTorch Step 5-1] GPT 모델과 글 생성", "6교시 · 실습", "pytorch_version/step5_gpt_train.py", [
    "class GPT(nn.Module):",
    "    def __init__(self, vocab_size, d_model=64, n_heads=4,",
    "                 n_layers=3, max_len=64, dropout=0.1):",
    "        super().__init__()",
    "        self.embed = TokenAndPosition(vocab_size, d_model, max_len)",
    "        self.blocks = nn.ModuleList(",
    "            Block(d_model, n_heads, max_len, dropout) for _ in range(n_layers))",
    "        self.ln_f = nn.LayerNorm(d_model)",
    "        self.head = nn.Linear(d_model, vocab_size)",
    "    def forward(self, idx, targets=None):",
    "        x = self.embed(idx)",
    "        for blk in self.blocks: x = blk(x)",
    "        logits = self.head(self.ln_f(x))              # (B, T, V)",
    "        loss = None",
    "        if targets is not None:",
    "            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),",
    "                                   targets.view(-1))",
    "        return logits, loss",
  ], [
    "nn.ModuleList: 블록 목록을 모델의 일부로 등록(파라미터 자동 수집).",
    "F.cross_entropy: softmax + −log 을 한 번에, 수치적으로도 안전.",
    "view(-1, V): (B, T, V) 를 (B·T, V) 로 펴서 모든 위치의 손실을 평균.",
    "targets 를 주면 학습용(loss 반환), 안 주면 추론용으로 쓸 수 있다.",
  ], { size: 10.5, cw: 8.9, esize: 12 });

  codeSlide("[PyTorch Step 5-2] 학습 루프", "6교시 · 실습", "pytorch_version/step5_gpt_train.py", [
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"",
    "model = GPT(len(chars), d_model=64, n_heads=4, n_layers=3,",
    "            max_len=CTX).to(device)",
    "opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)",
    "",
    "model.train()                          # dropout 켜기",
    "for step in range(1, STEPS + 1):",
    "    x, y = get_batch()",
    "    _, loss = model(x, y)               # 1 순전파 + 손실",
    "    opt.zero_grad(set_to_none=True)     # 2 기울기 비우기",
    "    loss.backward()                     # 3 역전파 (자동미분)",
    "    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 4 클리핑",
    "    opt.step()                          # 5 갱신",
    "",
    "model.eval()                           # dropout 끄기",
  ], [
    "리듬은 NumPy 버전과 같다. 달라진 점은 backward 한 줄!",
    ".to(device): 모델과 데이터를 GPU 로 보내 수십 배 빠르게.",
    "clip_grad_norm_: 기울기가 너무 크면 잘라 학습 폭주를 막는다.",
    "train()/eval(): dropout 같은 층의 동작을 학습용/추론용으로 전환.",
  ], { size: 11, cw: 8.4 });

  {
    const s = d.slide("글 생성: temperature 와 top-k", { tag: "6교시 · PyTorch" });
    d.code(s, [
      "@torch.no_grad()                      # 기울기 계산 불필요 -> 빠르고 메모리 절약",
      "def generate(self, idx, n_new, temperature=0.8, top_k=None):",
      "    for _ in range(n_new):",
      "        logits, _ = self(idx[:, -self.max_len:])  # 최근 토큰만",
      "        logits = logits[:, -1, :] / temperature   # 마지막 위치 점수",
      "        if top_k is not None:",
      "            kth = torch.topk(logits, top_k).values[:, -1, None]",
      "            logits = logits.masked_fill(logits < kth, float('-inf'))",
      "        probs = F.softmax(logits, dim=-1)",
      "        nxt = torch.multinomial(probs, 1)         # 확률대로 뽑기",
      "        idx = torch.cat([idx, nxt], dim=1)        # 뒤에 이어 붙이기",
      "    return idx",
    ], { x: 0.6, y: 1.5, w: 7.4, h: 3.9, size: 11, title: "pytorch_version/step5_gpt_train.py" });
    const H = (t) => ({ text: t, options: { bold: true, fill: { color: C.teal }, color: C.white, align: "center" } });
    const c = (t) => ({ text: t, options: { align: "left" } });
    s.addTable([[H("설정"), H("효과")], [c("temperature 낮음 (0.3)"), c("가장 그럴듯한 글자만, 안정적·반복적")], [c("temperature 높음 (1.5)"), c("다양하지만 엉뚱한 글자도 나옴")], [c("top-k = 5"), c("상위 5개 후보 안에서만 뽑는다")]], { x: 8.3, y: 1.5, w: 4.4, colW: [1.9, 2.5], fontFace: FONT, fontSize: 12, color: C.ink, rowH: 0.75, border: { type: "solid", color: C.line, pt: 1 }, valign: "middle" });
    d.card(s, { x: 0.6, y: 5.6, w: 12.1, h: 1.1, fill: C.gray, body: "생성(자기회귀): 한 글자 예측 → 입력 뒤에 붙임 → 다시 예측. ChatGPT가 글을 쓰는 방식과 같다. (No-grad: 학습이 아니라 사용할 때)", size: 14 });
  }

  {
    const s = d.slide("PyTorch 학습 결과와 검증", { tag: "6교시 · PyTorch", notes: "PyTorch 버전 실제 실행 결과(파라미터 153,116개, GPU, 600 step)." });
    s.addChart(pres.charts.LINE, [
      { name: "NumPy (27K)", labels: ["1", "100", "200", "300", "400", "500", "600"], values: [3.724, 0.8279, 0.2364, 0.1784, 0.1383, 0.1137, 0.1084] },
      { name: "PyTorch (153K)", labels: ["1", "100", "200", "300", "400", "500", "600"], values: [3.4923, 0.6768, 0.3176, 0.2227, 0.1521, 0.1579, 0.1453] },
    ], {
      x: 0.6, y: 1.5, w: 6.6, h: 4.6, chartColors: [C.teal, C.orange], showLegend: true, legendPos: "b", legendFontFace: FONT, legendFontSize: 12, showTitle: true, title: "두 구현의 loss 비교", titleFontFace: FONT, titleFontSize: 14, titleColor: C.ink,
      catAxisLabelColor: C.muted, valAxisLabelColor: C.muted, catAxisLabelFontFace: FONT, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, lineSize: 3, lineDataSymbolSize: 7, valAxisMinVal: 0, valAxisMaxVal: 4,
    });
    d.card(s, { x: 7.5, y: 1.5, w: 5.2, h: 2.4, fill: C.greenL, title: "검증: 내 구현 = 공식 구현", tColor: C.green, body: "step6_compare_builtin.py\n가중치를 nn.MultiheadAttention에\n복사해 넣으면 출력 차이 약 6e-8\n→ 올바르게 구현했다는 증거", size: 14 });
    d.card(s, { x: 7.5, y: 4.1, w: 5.2, h: 2.0, title: "관찰", body: "두 구현 모두 loss가 비슷하게 감소.\n(모델 크기·dropout이 달라 값은 조금씩 다르다)", size: 14 });
    d.text(s, "실제 서비스에선 nn.TransformerEncoderLayer 같은 검증된 내장 부품을 쓰기도 한다.", { x: 0.6, y: 6.25, w: 12, h: 0.4, size: 14, color: C.muted });
  }

  {
    const s = d.slide("진짜 ChatGPT와의 차이: 어디를 키우면 될까?", { tag: "6교시 · 확장" });
    const items = [["데이터", "우리: 300글자\n실제: 수조 토큰의 웹·책·코드", C.tealL, C.teal], ["토크나이저", "우리: 문자 단위\n실제: BPE 서브워드(수만 어휘)", C.orangeL, C.orange], ["모델 크기", "우리: 2.7만~15만 파라미터\n실제: 수십억~수천억", C.purpleL, C.purple], ["학습 후 정렬", "사전학습 후 SFT·RLHF 등으로\n대화·안전하게 다듬는다", C.greenL, C.green], ["추론 효율", "KV 캐시, 양자화, 병렬화로\n빠르고 싸게 서비스", C.gray, C.ink], ["긴 문맥", "RoPE 같은 위치 방식,\n효율적 어텐션으로 확장", C.redL, C.red]];
    items.forEach((it, i) => d.card(s, { x: 0.6 + (i % 3) * 4.1, y: 1.5 + Math.floor(i / 3) * 2.35, w: 3.9, h: 2.15, fill: it[2], title: it[0], tColor: it[3], body: it[1], size: 14 }));
    d.text(s, "핵심 구조(어텐션 + 블록)는 오늘 만든 것과 같다. 나머지는 규모와 엔지니어링의 문제!", { x: 0.6, y: 6.3, w: 12, h: 0.4, size: 15, bold: true, color: C.teal });
  }

  {
    const s = d.slide("전체 요약: 트랜스포머를 한 장으로", { tag: "SUMMARY" });
    const rows = [["1", "토큰화·임베딩", "글자 → 번호 → 벡터 (T, d)"], ["2", "위치 인코딩", "sin/cos 를 더해 순서 정보 추가"], ["3", "셀프 어텐션", "softmax(QKᵀ/√d)·V, 마스크로 미래 차단"], ["4", "멀티헤드", "여러 관점을 병렬로 보고 합친다"], ["5", "블록", "x + attn(ln(x)), x + ffn(ln(x))"], ["6", "출력·학습", "다음 토큰 확률 → 교차 엔트로피 → 역전파 → Adam"]];
    rows.forEach((r, i) => {
      const y = 1.45 + i * 0.74;
      s.addText(r[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.05, w: 0.6, h: 0.6, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 18, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, r[1], { x: 1.6, y: y + 0.1, w: 3.0, h: 0.5, size: 18, bold: true });
      d.text(s, r[2], { x: 4.7, y: y + 0.12, w: 8.0, h: 0.5, size: 16, color: C.muted });
    });
    d.card(s, { x: 0.7, y: 6.1, w: 12.0, h: 0.7, fill: C.greenL, body: "NumPy로 원리를, PyTorch로 실전을. 두 버전 모두 직접 돌려 본 당신은 이제 트랜스포머를 \"만들 수 있는\" 사람입니다!", size: 14 });
  }

  // ============================== 실습 문제 ==============================
  d.section("PRACTICE", "실습 문제", "스스로 풀어 본 뒤, 다음 슬라이드의 솔루션과 비교하세요", "각 문제는 10~15분. 정답은 practice_solutions.py 에 있다.");

  {
    const s = d.slide("실습 문제 1 ~ 2", { tag: "PRACTICE" });
    d.card(s, { x: 0.6, y: 1.5, w: 6.0, h: 5.1, fill: C.orangeL, title: "문제 1. 손계산 어텐션", tColor: C.orange, body: "토큰 3개, d_k = 2, 마스크 없음.\nQ = K = V = [[1,0], [0,1], [1,1]]\n\n(1) 첫 번째 토큰의 점수 Q₀·Kᵀ/√2 를 구하라.\n(2) softmax 를 적용해 가중치를 구하라.\n(3) 첫 번째 토큰의 출력 벡터를 구하라.\n(4) 코드로 확인하라.\n\n힌트: exp(0.707) ≈ 2.03, exp(0) = 1", size: 14 });
    d.card(s, { x: 6.9, y: 1.5, w: 5.8, h: 5.1, fill: C.tealL, title: "문제 2. 파라미터 수 세기", tColor: C.teal, body: "NumPy 모델: d_model = 32, 헤드 4개,\n블록 2개, 어휘 V = 28.\n\n(1) 블록 1개의 파라미터 수는?\n(2) 모델 전체의 파라미터 수는?\n(3) step6 을 실행해 출력과 비교하라.\n\n힌트: Linear(a, b) = a×b + b\n임베딩과 출력층(head)도 잊지 말 것!", size: 14 });
  }

  {
    const s = d.slide("실습 문제 3 ~ 4", { tag: "PRACTICE" });
    d.card(s, { x: 0.6, y: 1.5, w: 6.0, h: 5.1, fill: C.purpleL, title: "문제 3. top-k 샘플링 (NumPy)", tColor: C.purple, body: "함수 top_k_sample(logits, k, rng) 을 구현하라.\n\n• 점수가 높은 k개 글자만 후보로 남긴다.\n• 나머지는 확률 0 으로 만든다.\n• 남은 후보 중에서 확률대로 하나를 뽑는다.\n\n검증: logits = [1, 3, 0.5, 2.5, −1], k = 2 로\n1000번 뽑으면 1번과 3번만 나와야 한다.\n\n힌트: np.sort, np.where, softmax(Step 3)", size: 14 });
    d.card(s, { x: 6.9, y: 1.5, w: 5.8, h: 5.1, fill: C.greenL, title: "문제 4. 층 수 실험 (PyTorch)", tColor: C.green, body: "pytorch_version 의 GPT 에서 n_layers 를\n1, 2, 4 로 바꿔 300 step 학습하라.\n(d_model=64, 헤드 4, dropout=0)\n\n(1) 마지막 50 step 의 평균 loss 를 표로 정리\n(2) 층이 많을수록 어떻게 되는가?\n(3) 그 이유를 한 문장으로 설명하라.\n\n힌트: 층이 많으면 표현력이 커진다.", size: 14 });
  }

  // ============================== 솔루션 ==============================
  {
    const s = d.slide("솔루션 1. 손계산 어텐션", { tag: "SOLUTION" });
    d.card(s, { x: 0.6, y: 1.5, w: 6.6, h: 5.1, title: "풀이", body: "(1) 점수 = [Q₀·K₀, Q₀·K₁, Q₀·K₂] / √2\n    Q₀ = [1,0]  →  [1, 0, 1] / 1.414\n    = [0.707, 0, 0.707]\n\n(2) exp = [2.03, 1.00, 2.03], 합 = 5.06\n    가중치 = [0.401, 0.198, 0.401]\n\n(3) 출력 = 0.401·[1,0] + 0.198·[0,1] + 0.401·[1,1]\n    = [0.802, 0.599]", size: 14 });
    d.code(s, [
      "X = np.array([[1.,0.],[0.,1.],[1.,1.]])",
      "out, w = scaled_dot_product_attention(X, X, X)",
      "print(w[0].round(3))    # [0.401 0.198 0.401]",
      "print(out[0].round(3))  # [0.802 0.599]",
    ], { x: 7.5, y: 1.5, w: 5.2, h: 2.4, size: 11, title: "practice_solutions.py" });
    d.card(s, { x: 7.5, y: 4.1, w: 5.2, h: 2.5, fill: C.greenL, title: "확인 포인트", tColor: C.green, body: "• 가중치의 합 = 1\n• 자기 자신과 닮은 토큰(0번, 2번)에\n   큰 가중치가 갔다.", size: 14 });
  }

  {
    const s = d.slide("솔루션 2. 파라미터 수", { tag: "SOLUTION" });
    d.code(s, [
      "d, V, L = 32, 28, 2",
      "attn  = 4 * (d*d + d)                 # q,k,v,o      = 4,224",
      "ffn   = (d*4*d + 4*d) + (4*d*d + d)   # 128->, ->32  = 8,352",
      "ln    = 2 * (2*d)                      # gamma,beta x2 = 128",
      "block = attn + ffn + ln                # = 12,704",
      "total = V*d + L*block + 2*d + (d*V + V)",
      "#       임베딩  블록들   ln_f   출력층(head)",
      "print(block, total)                    # 12704 27292",
    ], { x: 0.6, y: 1.5, w: 7.6, h: 3.4, size: 12, title: "practice_solutions.py" });
    d.card(s, { x: 8.5, y: 1.5, w: 4.2, h: 3.4, fill: C.greenL, title: "정답", tColor: C.green, body: "(1) 블록 1개: 12,704\n(2) 전체: 27,292\n(3) step6 출력\n    \"파라미터 수: 27292\" 와 일치", size: 15 });
    d.card(s, { x: 0.6, y: 5.2, w: 12.1, h: 1.4, title: "자주 틀리는 부분", body: "임베딩(896)과 출력층(924), 마지막 LayerNorm(64)을 빼먹기 쉽다.  어텐션의 q,k,v,o 모두 bias를 가진다(NumPy 버전).", size: 14 });
  }

  {
    const s = d.slide("솔루션 3. top-k 샘플링", { tag: "SOLUTION" });
    d.code(s, [
      "def top_k_sample(logits, k, rng, temperature=1.0):",
      "    logits = logits / temperature",
      "    cut = np.sort(logits)[-k]              # 상위 k번째 점수",
      "    logits = np.where(logits < cut, -np.inf, logits)  # 탈락",
      "    p = softmax(logits)                    # 탈락한 칸은 확률 0",
      "    return int(rng.choice(len(p), p=p))",
      "",
      "rng = np.random.default_rng(0)",
      "logits = np.array([1.0, 3.0, 0.5, 2.5, -1.0])",
      "picks = [top_k_sample(logits, 2, rng) for _ in range(1000)]",
      "print(sorted(set(picks)))                  # [1, 3]",
    ], { x: 0.6, y: 1.5, w: 7.8, h: 4.0, size: 12, title: "practice_solutions.py" });
    d.explain(s, [
      "np.sort(...)[-k]: 오름차순 정렬의 뒤에서 k번째 = 상위 k번째 점수.",
      "그보다 작은 점수는 −inf 로 덮어 탈락시킨다.",
      "softmax(−inf) = 0 이므로 탈락자는 절대 뽑히지 않는다.",
    ], { x: 8.7, y: 1.5, w: 4.0, h: 4.0, size: 13, title: "핵심" });
    d.card(s, { x: 0.6, y: 5.8, w: 12.1, h: 0.9, fill: C.greenL, body: "결과: 1000번 모두 1번(3.0)과 3번(2.5)만 나온다. k=1 이면 항상 최고점수만 뽑는 \"greedy\" 방식이 된다.", size: 14 });
  }

  {
    const s = d.slide("솔루션 4. 층 수 실험 결과", { tag: "SOLUTION", notes: "실제 실행 결과 (seed 0, 300 step, 마지막 50 step 평균)." });
    s.addChart(pres.charts.BAR, [{ name: "마지막 50 step 평균 loss", labels: ["1층", "2층", "4층"], values: [0.432, 0.198, 0.146] }], {
      x: 0.6, y: 1.5, w: 6.2, h: 3.9, barDir: "col", chartColors: [C.teal], showLegend: false, showTitle: true, title: "층 수별 최종 loss (낮을수록 좋음)", titleFontFace: FONT, titleFontSize: 14, titleColor: C.ink,
      showValue: true, dataLabelFontSize: 14, dataLabelColor: C.ink, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisLabelColor: C.ink, catAxisLabelFontFace: FONT, catAxisLabelFontSize: 14, valAxisLabelColor: C.muted, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 0.5, barGapWidthPct: 60,
    });
    d.card(s, { x: 7.1, y: 1.5, w: 5.6, h: 2.6, fill: C.greenL, title: "정답", tColor: C.green, body: "(1) 1층 0.432  /  2층 0.198  /  4층 0.146\n(2) 층이 많을수록 loss가 낮아졌다.\n(3) 층이 깊을수록 더 복잡한 패턴을 표현할 수 있다.", size: 14 });
    d.card(s, { x: 7.1, y: 4.3, w: 5.6, h: 2.3, fill: C.orangeL, title: "주의", tColor: C.orange, body: "• 학습 데이터에 대한 loss일 뿐(검증 loss 아님)\n• 너무 키우면 외우기(과적합)만 늘 수 있다.\n• 실행 환경에 따라 값은 조금 달라진다.", size: 13 });
    d.code(s, ["for n_layers in (1, 2, 4):", "    model = GPT(V, d_model=64, n_heads=4,", "                n_layers=n_layers, dropout=0.0)"], { x: 0.6, y: 5.6, w: 6.2, h: 1.1, size: 10 });
  }

  {
    const s = d.slide("더 공부하려면", { tag: "NEXT STEP" });
    d.card(s, { x: 0.6, y: 1.5, w: 6.0, h: 5.1, fill: C.paper, title: "도전 과제", body: "• 학습 글을 한국어로 바꿔 보기 (tiny_corpus.txt)\n• 위치 인코딩을 학습형 임베딩으로 교체해 비교\n• 검증 데이터를 따로 두고 과적합 확인\n• BPE 토크나이저 직접 만들기\n• KV 캐시로 생성 속도 높이기\n• 인코더-디코더(번역) 구조 구현하기", size: 15 });
    d.card(s, { x: 6.9, y: 1.5, w: 5.8, h: 5.1, fill: C.tealL, title: "참고 자료", tColor: C.teal, body: "• 논문: Attention Is All You Need (2017)\n• Andrej Karpathy: \"Let's build GPT\" 강의, nanoGPT\n• Jay Alammar: The Illustrated Transformer\n• PyTorch 공식 문서: torch.nn\n\n오늘 코드: Slider/transformer/\n  numpy_version/ · pytorch_version/\n  practice_solutions.py", size: 14 });
  }
};
