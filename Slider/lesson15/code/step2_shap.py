"""Step 2. SHAP(샤플리 값)을 식 15.4 그대로 직접 계산한다.  (NumPy 먼저, 마지막에 PyTorch 버전)
φ_i = Σ_{S ⊆ F\{i}}  |S|!(|F|−|S|−1)!/|F|!  ·  [ f(S∪{i}) − f(S) ]
'특성 S만 알고 있을 때의 예측 f(S)' = S에 없는 특성을 기준값(baseline)으로 바꿔 넣고 모델을 돌린 값.
실행: python step2_shap.py
"""
import time, itertools, math
import numpy as np
from step1_permutation import make_data, fit_logreg, FEATS

def subset_values(f, x, base):
    """모든 부분집합 S(비트마스크 0..2^n−1)에 대해 f(S)를 한 번에 계산한다.
    마스크의 i번째 비트가 1이면 x[i]를 쓰고, 0이면 baseline[i]로 대체."""
    n = len(x)
    masks = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1         # (2^n, n) 0/1 표
    X = np.where(masks == 1, x, base)                                 # 포함된 특성만 진짜 값
    return f(X)                                                       # (2^n,) 모델 출력들

def shapley_exact(f, x, base):
    """식 15.4를 부분집합을 전부 훑어서 그대로 계산 (특성이 n개면 2^n번 모델 호출)."""
    n = len(x); v = subset_values(f, x, base); phi = np.zeros(n)
    for i in range(n):
        for mask in range(2 ** n):
            if mask >> i & 1: continue                                # S에는 i가 없어야 한다
            k = bin(mask).count("1")                                  # |S|
            wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)   # 조합 가중치
            phi[i] += wgt * (v[mask | 1 << i] - v[mask])              # 가중치 × (i를 추가했을 때의 이득)
    return phi, v[0], v[-1]                                           # φ, f(∅), f(전체)

def shapley_sampling(f, x, base, n_perm, rng):
    """근사: 무작위 순서로 특성을 하나씩 켜면서 '켜질 때마다 늘어난 값'을 모아 평균 (Kernel/Deep SHAP과는 다른, 가장 단순한 근사)."""
    n = len(x); phi = np.zeros(n)
    for _ in range(n_perm):
        cur = base.copy().astype(float); prev = f(cur[None])[0]
        for i in rng.permutation(n):
            cur[i] = x[i]; now = f(cur[None])[0]
            phi[i] += now - prev; prev = now
    return phi / n_perm

def shapley_torch(model, x, base):
    """PyTorch 버전: 마스크 표를 텐서로 만들어 모델에 한 번에 통과시킨다. (model: (N,n)->(N,))"""
    import torch
    n = len(x); x = torch.as_tensor(x, dtype=torch.float32); base = torch.as_tensor(base, dtype=torch.float32)
    masks = ((torch.arange(2 ** n)[:, None] >> torch.arange(n)) & 1).bool()
    with torch.no_grad(): v = model(torch.where(masks, x, base))
    phi = torch.zeros(n)
    for i in range(n):
        for mask in range(2 ** n):
            if mask >> i & 1: continue
            k = bin(mask).count("1")
            phi[i] += math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * (v[mask | 1 << i] - v[mask])
    return phi.numpy()

if __name__ == "__main__":
    # ---------- (1) 손으로 풀 수 있는 3특성 예: f(x) = 2·x1 + 3·x2 + x1·x3,  x=(1,2,3), 기준=(0,0,0) ----------
    f3 = lambda X: 2 * X[:, 0] + 3 * X[:, 1] + X[:, 0] * X[:, 2]
    x, base = np.array([1., 2., 3.]), np.zeros(3)
    phi, f0, f1 = shapley_exact(f3, x, base)
    print("[3특성] f(∅)=%.1f  f(전체)=%.1f  φ=%s  합=%.1f" % (f0, f1, phi, phi.sum()))
    assert np.allclose(phi, [3.5, 6.0, 1.5]) and np.isclose(phi.sum(), f1 - f0)     # 효율성 공리

    # ---------- (2) 학습된 모델 한 명 설명 (로지스틱 회귀의 로짓 출력) ----------
    rng = np.random.default_rng(0)
    Xtr, ytr = make_data(2000, rng); w, b = fit_logreg(Xtr, ytr)
    logit = lambda X: X @ w + b
    xi = np.array([1.5, -0.5, 1.0, 2.0]); mu = Xtr.mean(0)                          # 설명할 학생, 기준 = 평균 학생
    phi, f0, f1 = shapley_exact(logit, xi, mu)
    print("\n[학생 1명] 평균 학생 로짓 %.2f → 이 학생 로짓 %.2f" % (f0, f1))
    for f, p in zip(FEATS, phi): print(f"  φ({f:6s}) = {p:+.3f}")
    print("  합 = %.3f  (효율성: f(전체) − f(∅) = %.3f)" % (phi.sum(), f1 - f0))
    assert np.isclose(phi.sum(), f1 - f0)
    assert np.allclose(phi, w * (xi - mu)), "선형 모델이면 φ_i = w_i·(x_i − 평균_i) 와 같다"
    print("  선형 모델의 닫힌 해 w_i(x_i−평균_i) 와 일치 ✔")

    # ---------- (3) 비용: 특성이 늘면 2^n 으로 폭발 → 근사가 필요한 이유 ----------
    print("\n[비용] 특성 수 n → 부분집합 수 2^n, 정확 계산 시간")
    rows = []
    for n in (8, 12, 16, 18):
        wn = rng.normal(size=n); fn = lambda X, wn=wn: np.tanh(X @ wn) + (X[:, 0] * X[:, 1])
        t = time.perf_counter(); shapley_exact(fn, rng.normal(size=n), np.zeros(n)); dt = time.perf_counter() - t
        rows.append((n, 2 ** n, dt)); print(f"  n={n:2d}  2^n={2**n:7d}  {dt:6.2f}초")
    # 근사: 무작위 순서 표본 수에 따른 오차
    print("\n[근사] 순서 표본 수에 따른 오차 (n=4, 위 학생, 비선형 모델)")
    fnl = lambda X: np.tanh(X @ w) + X[:, 0] * X[:, 1]
    ex, _, _ = shapley_exact(fnl, xi, mu); errs = []
    for m in (5, 20, 100, 1000):
        errs.append((m, float(np.abs(shapley_sampling(fnl, xi, mu, m, rng) - ex).max())))
        print(f"  표본 {m:4d}개 → 최대 오차 {errs[-1][1]:.4f}")
    np.save("_shap.npy", dict(phi=phi, cost=rows, errs=errs, f0=f0, f1=f1), allow_pickle=True)

    # ---------- (4) PyTorch 버전이 NumPy 와 같은 값을 내는지 ----------
    try:
        import torch
        wt = torch.tensor(w, dtype=torch.float32)
        pt = shapley_torch(lambda X: X @ wt + float(b), xi, mu)
        print("\n[PyTorch] φ =", np.round(pt, 3), "| NumPy와 일치:", np.allclose(pt, phi, atol=1e-4))
    except ImportError:
        print("\n(PyTorch가 없어 건너뜀)")
