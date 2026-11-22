"""Step 6. PyTorch 버전: torch.optim 옵티마이저 비교, 직접 만든 Lion, 학습률 스케줄러, 파라미터 그룹.
NumPy(step5)와 같은 데이터·같은 모델이다.
실행: python step6_torch_compare.py   (5 seeds 평균, 약 1분. GPU 불필요)
"""
import math, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from optim_np import Lion as LionNP

# ---------------- (1) PyTorch 용 Lion: 공식 구현이 torch.optim 에 없으므로 직접 만든다 ----------------
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            b1, b2 = grp["betas"]
            for p in grp["params"]:
                if p.grad is None: continue
                m = self.state[p].setdefault("m", torch.zeros_like(p))          # 상태는 m 하나뿐
                c = (b1 * m + (1 - b1) * p.grad).sign()                         # 부호만 사용
                p.add_(c + grp["weight_decay"] * p, alpha=-grp["lr"])           # θ ← θ − η(c + λθ)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)                           # 모멘텀은 β2 로 갱신

# NumPy Lion 과 같은 값인지 확인
r = np.random.default_rng(0); w0 = r.normal(size=(3, 4)); G = r.normal(size=(30, 3, 4))
a = [w0.copy()]; o = LionNP(1e-2, weight_decay=0.1); t = nn.Parameter(torch.tensor(w0)); to = Lion([t], 1e-2, weight_decay=0.1)
for k in range(30):
    o.step(a, [G[k].copy()]); t.grad = torch.tensor(G[k]); to.step()
err = np.abs(a[0] - t.detach().numpy()).max(); print(f"[NumPy vs torch Lion] 최대 차이 {err:.2e}"); assert err < 1e-9

# ---------------- (2) 데이터와 모델 (step5 와 동일 구조) ----------------
d = load_digits(); X_all = torch.tensor(d.data / 16.0, dtype=torch.float32); y_all = torch.tensor(d.target)
def make_net(): return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
net = make_net(); n_par = sum(p.numel() for p in net.parameters())
print(f"[파라미터 수] 직접 계산 64·(64+1)+10·(64+1) = {64*65+10*65}  /  torch 가 센 값 = {n_par}"); assert n_par == 64 * 65 + 10 * 65

def run(make_opt, seed, epochs=40, bs=64, sched=None):
    torch.manual_seed(seed); p = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(seed))
    Xtr, ytr, Xte, yte = X_all[p[:1200]], y_all[p[:1200]], X_all[p[1200:]], y_all[p[1200:]]
    net = make_net(); opt = make_opt(net); steps = epochs * (1200 // bs)
    sch = sched(opt, steps) if sched else None; loss_fn = nn.CrossEntropyLoss(); curve = []
    for ep in range(epochs):
        idx = torch.randperm(1200)
        for i in range(0, 1200 - bs + 1, bs):
            b = idx[i:i + bs]; loss = loss_fn(net(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if sch: sch.step()
        with torch.no_grad(): curve.append(loss_fn(net(Xtr), ytr).item())
    with torch.no_grad(): acc = (net(Xte).argmax(1) == yte).float().mean().item()
    state = sum(t.numel() for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.numel() > 1)  # 옵티마이저 상태 크기
    return curve, acc, state

# ---------------- (3) 옵티마이저 비교 (5 seeds 평균) ----------------
CASES = {
    "SGD+모멘텀":  lambda n: torch.optim.SGD(n.parameters(), 0.05, momentum=0.9),
    "AdamW":      lambda n: torch.optim.AdamW(n.parameters(), 3e-3, weight_decay=0.01),
    "Lion":       lambda n: Lion(n.parameters(), 1e-3, weight_decay=0.1),
    "Adafactor":  lambda n: torch.optim.Adafactor(n.parameters(), lr=1e-2),
}
RES = {}
if __name__ == "__main__":
    print(f"\n{'옵티마이저':<12}{'학습손실(ep40)':>14}{'시험 정확도 평균±표준편차':>26}{'상태 숫자 개수':>14}")
    for name, mk in CASES.items():
        rs = [run(mk, s) for s in range(5)]; accs = np.array([r[1] for r in rs]) * 100
        RES[name] = dict(curve=np.mean([r[0] for r in rs], 0), acc=accs)
        print(f"{name:<12}{RES[name]['curve'][-1]:>14.4f}{accs.mean():>16.1f}% ± {accs.std():.1f}{rs[0][2]:>14,}")

    # ---------------- (4) 학습률 스케줄러: AdamW 고정 vs 웜업+코사인 vs 코사인 재시작 ----------------
    def warm_cos(opt, T, warm=30):
        f = lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / (T - warm)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)                           # 학습률 = 기준 lr × f(t)
    adamw = lambda n: torch.optim.AdamW(n.parameters(), 1e-2, weight_decay=0.01)  # 일부러 큰 lr: 스케줄 효과가 잘 보인다
    print(f"\n{'AdamW(lr 1e-2) 스케줄':<22}{'학습손실(ep40)':>14}{'시험 정확도':>16}")
    for name, sc in [("고정 학습률", None), ("웜업+코사인", warm_cos),
                     ("코사인 재시작 SGDR", lambda o, T: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=T // 4))]:
        rs = [run(adamw, s, sched=sc) for s in range(5)]; accs = np.array([r[1] for r in rs]) * 100
        RES["sch_" + name] = dict(curve=np.mean([r[0] for r in rs], 0), acc=accs)
        print(f"{name:<22}{np.mean([r[0][-1] for r in rs]):>14.4f}{accs.mean():>14.1f}% ± {accs.std():.1f}")

    # ---------------- (5) 파라미터 그룹: 층마다 다른 옵티마이저/설정 (원본 '실용적 팁'의 AdamW+SGD 혼합) ----------------
    net = make_net(); w, b = [net[0].weight, net[2].weight], [net[0].bias, net[2].bias]
    opts = [torch.optim.AdamW(w, 3e-3, weight_decay=0.01), torch.optim.SGD(b, 0.05)]   # 가중치는 AdamW, 편향은 SGD
    x, y = X_all[:64], y_all[:64]; before = net[0].weight.clone()
    nn.CrossEntropyLoss()(net(x), y).backward()
    for o in opts: o.step()
    print(f"\n[혼합 옵티마이저] 첫 층 가중치가 바뀌었나? {not torch.equal(before, net[0].weight)}  (두 옵티마이저를 나란히 step() 하면 된다)")
    np.save("_res6.npy", RES, allow_pickle=True)
