"""
[Step 6] 트랜스포머 조립 + 학습 + 글 생성  -  NumPy 버전 (PyTorch 없이!)
=====================================================================
Step 1~5 에서 만든 부품을 모두 조립한다. (GPT처럼 '다음 글자 맞히기'를 배우는 디코더)

구조 (아래에서 위로):
  글자 번호 -> 임베딩 + 위치인코딩 -> [ 트랜스포머 블록 x N ] -> LayerNorm -> 출력층 -> 다음 글자 점수

트랜스포머 블록 하나 (Pre-LN 방식):
  x = x + MultiHeadAttention( LayerNorm(x) )     # 단어끼리 정보 교환
  x = x + FeedForward( LayerNorm(x) )            # 각 단어를 혼자 깊게 생각
"""
import sys
import numpy as np

from step2_positional_encoding import positional_encoding
from step5_autograd import Tensor

sys.setrecursionlimit(10000)
rng = np.random.default_rng(0)


# ====================================================================
# 부품 1: Linear (y = x W + b)
# ====================================================================
class Linear:
    def __init__(self, n_in, n_out, bias=True):
        # 가중치 초기값: 입력이 많을수록 작게 -> 출력이 폭주하지 않게 한다
        self.W = Tensor(rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_out)))
        self.b = Tensor(np.zeros(n_out)) if bias else None

    def __call__(self, x):
        y = x @ self.W
        return y + self.b if self.b is not None else y

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


# ====================================================================
# 부품 2: LayerNorm  (각 토큰 벡터를 평균 0, 분산 1로 맞춘 뒤 gamma, beta 로 조절)
# ====================================================================
class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma, self.beta, self.eps = Tensor(np.ones(d)), Tensor(np.zeros(d)), eps

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)              # 평균
        xc = x - mu                                      # 평균을 뺀다
        var = (xc * xc).mean(axis=-1, keepdims=True)     # 분산
        xn = xc * ((var + self.eps) ** -0.5)             # 표준편차로 나눈다
        return xn * self.gamma + self.beta

    def params(self):
        return [self.gamma, self.beta]


# ====================================================================
# 부품 3: 멀티헤드 셀프 어텐션 (Step 4 와 같은 계산을 Tensor 로 + 배치(B)까지)
# ====================================================================
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.h, self.dk = n_heads, d_model // n_heads
        self.q, self.k, self.v, self.o = (Linear(d_model, d_model) for _ in range(4))

    def __call__(self, x):
        B, T, D = x.data.shape

        def split(t):                                    # (B,T,D) -> (B,h,T,dk)
            return t.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.dk))  # (B,h,T,T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # 미래를 가리는 마스크
        w = scores.masked_fill(mask, -1e9).softmax()      # 어텐션 가중치
        ctx = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, D)  # 헤드들을 다시 이어 붙임
        return self.o(ctx)

    def params(self):
        return sum((m.params() for m in (self.q, self.k, self.v, self.o)), [])


# ====================================================================
# 부품 4: FeedForward (벡터를 4배로 넓혔다가 ReLU 후 다시 줄인다)
# ====================================================================
class FeedForward:
    def __init__(self, d_model):
        self.l1, self.l2 = Linear(d_model, 4 * d_model), Linear(4 * d_model, d_model)

    def __call__(self, x):
        return self.l2(self.l1(x).relu())

    def params(self):
        return self.l1.params() + self.l2.params()


# ====================================================================
# 부품 5: 트랜스포머 블록 = 어텐션 + 피드포워드 (+ 잔차연결 + LayerNorm)
# ====================================================================
class Block:
    def __init__(self, d_model, n_heads):
        self.ln1, self.attn = LayerNorm(d_model), MultiHeadAttention(d_model, n_heads)
        self.ln2, self.ffn = LayerNorm(d_model), FeedForward(d_model)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))                   # 잔차연결(residual): 원본을 더해 준다
        x = x + self.ffn(self.ln2(x))
        return x

    def params(self):
        return sum((m.params() for m in (self.ln1, self.attn, self.ln2, self.ffn)), [])


# ====================================================================
# 전체 모델
# ====================================================================
class GPT:
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=2, max_len=64):
        self.emb = Tensor(rng.normal(0, 0.1, size=(vocab_size, d_model)))   # 임베딩 표
        self.pe = positional_encoding(max_len, d_model)                       # 학습 안 하는 상수
        self.blocks = [Block(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)
        self.scale = np.sqrt(d_model)

    def __call__(self, idx):                             # idx: (B,T) 정수 배열
        B, T = idx.shape
        x = self.emb.take(idx) * self.scale + self.pe[:T]   # 임베딩 + 위치 인코딩
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))                   # (B,T,V) 다음 글자 점수

    def params(self):
        ps = [self.emb] + self.ln_f.params() + self.head.params()
        for b in self.blocks:
            ps += b.params()
        return ps


# ====================================================================
# 옵티마이저: Adam (기울기를 부드럽게 평균 내어 가중치를 갱신)
# ====================================================================
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)     # 초반 편향 보정
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def zero_grad(self):
        for p in self.p:
            p.grad[...] = 0.0


# ====================================================================
# 데이터 준비 + 학습 + 생성
# ====================================================================
def generate(model, start_ids, n_new, itos, ctx, temperature=0.8):
    """한 글자씩 예측해서 이어 붙인다 (autoregressive)."""
    ids = list(start_ids)
    for _ in range(n_new):
        x = np.array([ids[-ctx:]])                       # 최근 ctx 글자만 입력
        logits = model(x).data[0, -1] / temperature      # 마지막 위치의 점수
        p = np.exp(logits - logits.max())
        p /= p.sum()
        ids.append(int(rng.choice(len(p), p=p)))         # 확률대로 한 글자 뽑기
    return "".join(itos[i] for i in ids)


if __name__ == "__main__":
    import os
    text = open(os.path.join(os.path.dirname(__file__), "..", "tiny_corpus.txt"), encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text])

    CTX, BATCH, STEPS = 32, 16, 600
    model = GPT(len(chars), d_model=32, n_heads=4, n_layers=2, max_len=CTX)
    opt = Adam(model.params(), lr=3e-3)
    print("파라미터 수:", sum(p.data.size for p in model.params()))

    for step in range(1, STEPS + 1):
        # 랜덤 위치에서 CTX 글자를 잘라 입력(x), 한 글자 밀린 것을 정답(y)으로 쓴다
        starts = rng.integers(0, len(data) - CTX - 1, size=BATCH)
        x = np.stack([data[s:s + CTX] for s in starts])
        y = np.stack([data[s + 1:s + CTX + 1] for s in starts])

        logits = model(x)                                              # 1) 순전파
        loss = logits.reshape(-1, len(chars)).cross_entropy(y.reshape(-1))  # 2) 손실
        opt.zero_grad()                                                # 3) 이전 기울기 비우기
        loss.backward()                                                # 4) 역전파
        opt.step()                                                     # 5) 가중치 갱신
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {float(loss.data):.4f}")

    print("\n--- 생성 결과 ---")
    start = [stoi[c] for c in "the "]
    print(generate(model, start, 120, itos, CTX))
