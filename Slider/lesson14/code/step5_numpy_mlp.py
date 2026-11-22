"""Step 5. NumPy 로 2층 신경망을 직접 학습시키며 옵티마이저를 비교한다 (역전파도 손으로 구현).
데이터: sklearn 손글씨 숫자(8×8=64픽셀). 학습 1200장 / 시험 597장.
실행: python step5_numpy_mlp.py     (3 seeds 평균, 약 1분)
"""
import numpy as np
from sklearn.datasets import load_digits
from optim_np import SGD, Adam, Lion, Adafactor, warmup_cosine

def load(seed):
    d = load_digits(); X = d.data / 16.0; y = d.target
    p = np.random.default_rng(seed).permutation(len(X)); X, y = X[p], y[p]
    return X[:1200], y[:1200], X[1200:], y[1200:]

def init(rng, h=64):
    # He 초기화: 표준편차 √(2/입력수)  (Lesson 12)
    return [rng.normal(0, np.sqrt(2 / 64), (64, h)), np.zeros(h), rng.normal(0, np.sqrt(2 / h), (h, 10)), np.zeros(10)]

def forward(P, X):
    W1, b1, W2, b2 = P
    H = np.maximum(0, X @ W1 + b1)                    # 은닉층 + ReLU
    Z = H @ W2 + b2                                   # 점수(logits)
    Z = Z - Z.max(1, keepdims=True)                   # 수치 안정화
    S = np.exp(Z); S /= S.sum(1, keepdims=True)       # softmax 확률
    return H, S

def backward(P, X, y, H, S):
    n = len(X); dZ = S.copy(); dZ[np.arange(n), y] -= 1; dZ /= n     # softmax+크로스엔트로피의 기울기: (확률 − 정답)/n
    dW2, db2 = H.T @ dZ, dZ.sum(0)
    dH = dZ @ P[2].T; dH[H <= 0] = 0                  # ReLU 를 통과한 곳만 기울기가 흐른다
    return [X.T @ dH, dH.sum(0), dW2, db2]

def train(make_opt, seed, epochs=40, bs=64, schedule=False):
    Xtr, ytr, Xte, yte = load(seed); rng = np.random.default_rng(seed); P = init(rng)
    opt = make_opt(); lr0 = opt.lr; steps = epochs * (len(Xtr) // bs); t = 0; curve = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - bs + 1, bs):
            b = idx[i:i + bs]; H, S = forward(P, Xtr[b])
            if schedule: opt.lr = warmup_cosine(t, steps, 30, lr0)
            opt.step(P, backward(P, Xtr[b], ytr[b], H, S)); t += 1
        _, S = forward(P, Xtr); curve.append(-np.log(S[np.arange(len(ytr)), ytr] + 1e-12).mean())
    _, S = forward(P, Xte)
    return curve, (S.argmax(1) == yte).mean()

CASES = {
    "SGD (lr 0.1)":              lambda: SGD(0.1),
    "SGD+모멘텀 (lr 0.05, μ=.9)": lambda: SGD(0.05, 0.9),
    "Adam (lr 3e-3)":            lambda: Adam(3e-3),
    "AdamW (lr 3e-3, wd 0.01)":  lambda: Adam(3e-3, weight_decay=0.01, decoupled=True),
    "Lion (lr 1e-3)":            lambda: Lion(1e-3, weight_decay=0.1),
    "Adafactor (lr 1e-2)":       lambda: Adafactor(1e-2),
}
if __name__ == "__main__":
    print(f"{'옵티마이저':<28}{'epoch5 학습손실':>14}{'epoch40 학습손실':>16}{'시험 정확도':>12}")
    for name, mk in CASES.items():
        rs = [train(mk, s) for s in range(3)]
        c = np.mean([r[0] for r in rs], 0); a = np.mean([r[1] for r in rs])
        print(f"{name:<28}{c[4]:>14.3f}{c[-1]:>16.3f}{a*100:>11.1f}%")
