"""Step 4. 종합 실험: 초기화 + 정규화가 실제 학습에 주는 효과 (PyTorch)
8층 MLP로 '나선형 3-클래스' 분류를 학습하고 손실 곡선을 비교한다.
실행: python step4_compare.py   (결과 그림: ../images/train_compare.png)
"""
import numpy as np
import torch
import torch.nn as nn

def make_spiral(n=200, k=3, seed=0):
    """나선 모양 점 데이터 (직선으로는 못 나누는 문제)"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.05, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [c] * n
    return torch.tensor(np.vstack(X), dtype=torch.float32), torch.tensor(y)

def build(init, norm, depth=8, width=64):
    """init: 'small' | 'xavier' | 'he'   norm: None | 'bn' | 'ln'"""
    layers, d_in = [], 2
    for _ in range(depth):
        lin = nn.Linear(d_in, width)
        if init == "small":   nn.init.normal_(lin.weight, 0, 0.01)       # 너무 작은 초기값
        elif init == "xavier": nn.init.xavier_normal_(lin.weight)         # Var = 2/(n_in+n_out)
        else:                  nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")  # Var = 2/n_in
        nn.init.zeros_(lin.bias)
        layers.append(lin)                                                # Linear -> (Norm) -> ReLU
        if norm == "bn": layers.append(nn.BatchNorm1d(width))
        if norm == "ln": layers.append(nn.LayerNorm(width))
        layers.append(nn.ReLU())
        d_in = width
    layers.append(nn.Linear(width, 3))
    return nn.Sequential(*layers)

def train(init, norm, epochs=200, lr=0.05, seed=0):
    torch.manual_seed(seed)
    X, y = make_spiral()
    model = build(init, norm)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn, hist = nn.CrossEntropyLoss(), []
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):                 # 미니배치 64
            idx = perm[i:i + 64]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(X); hist.append(loss_fn(out, y).item())
            acc = (out.argmax(1) == y).float().mean().item()
        model.train()
    return hist, acc

CASES = [("작은 초기값(0.01)", "small", None), ("Xavier", "xavier", None), ("He", "he", None),
         ("He + BatchNorm", "he", "bn"), ("He + LayerNorm", "he", "ln"), ("작은 초기값 + BatchNorm", "small", "bn")]

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for name, init, norm in CASES:
        h, acc = train(init, norm)
        print(f"{name:22s} 최종 손실 {h[-1]:.3f}  정확도 {acc:.1%}  (10 에폭 손실 {h[9]:.3f})")
        ax.plot(h, label=f"{name} ({acc:.0%})")
    ax.set_xlabel("에폭"); ax.set_ylabel("학습 손실"); ax.set_ylim(0, 1.3); ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("../images/train_compare.png", facecolor="white")
