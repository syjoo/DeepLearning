"""Step 5. 한 걸음 더: Cutout, Mixup, CutMix, 그리고 테스트 시 증강(TTA)
실행: python step5_advanced.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from step4_compare import X, y, model, MILD, N_TRAIN, EPOCHS
from step3_pipeline import RandomAffine

def cutout(x, size=3):
    """무작위 위치의 size x size 사각형을 0으로 지운다 (레이블은 그대로)"""
    x = x.clone(); n, _, h, w = x.shape
    cy, cx = torch.randint(0, h, (n,)), torch.randint(0, w, (n,))
    for i in range(n):
        x[i, :, max(0, cy[i] - size // 2):cy[i] + size // 2 + 1, max(0, cx[i] - size // 2):cx[i] + size // 2 + 1] = 0
    return x

def mixup(x, y, alpha=0.4):
    """두 이미지를 lam : (1-lam) 로 섞고, 정답(원-핫)도 같은 비율로 섞는다.  lam ~ Beta(alpha, alpha)"""
    lam = float(np.random.beta(alpha, alpha)); j = torch.randperm(len(x))
    return lam * x + (1 - lam) * x[j], lam, y, y[j]

def cutmix(x, y, alpha=1.0):
    """이미지 일부를 다른 이미지 조각으로 바꾸고, 정답은 '차지한 넓이 비율'로 섞는다"""
    lam = float(np.random.beta(alpha, alpha)); j = torch.randperm(len(x)); h, w = x.shape[-2:]
    ch, cw = int(h * (1 - lam) ** .5), int(w * (1 - lam) ** .5)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, h), max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone(); x[:, :, y0:y1, x0:x1] = x[j][:, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)                       # 실제로 남은 원본의 넓이 비율
    return x, lam, y, y[j]

def run(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(len(X), generator=g)
    tr, rest = perm[:N_TRAIN], perm[N_TRAIN:]
    xtr, ytr, xv, yv = X[tr], y[tr], X[rest], y[rest]
    torch.manual_seed(1000 + seed); xo = MILD()(xv); torch.manual_seed(seed)
    net, base, ce = model(), MILD(), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(EPOCHS * 150 // 32):
        idx = torch.randint(0, len(xtr), (32,)); xb, yb = base(xtr[idx]), ytr[idx]
        if mode == "cutout": xb = cutout(xb)
        if mode in ("mixup", "cutmix"):
            xb, lam, ya, yb2 = (mixup if mode == "mixup" else cutmix)(xb, yb)
            out = net(xb); loss = lam * ce(out, ya) + (1 - lam) * ce(out, yb2)          # 정답도 섞어서 계산
        else: loss = ce(net(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = lambda a: (net(a).argmax(1) == yv).float().mean().item()
        tta = torch.stack([F.softmax(net(RandomAffine(10, 1)(xv)), 1) for _ in range(16)]).mean(0)   # 16번 변형해 예측을 평균 (검증 이미지에 적용)
        return acc(xv), acc(xo), (tta.argmax(1) == yv).float().mean().item()

if __name__ == "__main__":
    print(f"{'방식':10s}{'검증':>8s}{'어려운 검증':>12s}{'검증+TTA(16회)':>16s}")
    for m in ["base", "cutout", "mixup", "cutmix"]:
        a = np.array([run(m, s) for s in range(5)]).mean(0) * 100
        print(f"{m:14s}{a[0]:6.1f}%{a[1]:10.1f}%{a[2]:14.1f}%")
