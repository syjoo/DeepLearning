"""Step 1. 경사 하강(GD)과 학습률, 지수이동평균(EMA), 모멘텀을 NumPy로 직접 확인한다.
실행: python step1_gd_ema.py
"""
import numpy as np
from optim_np import SGD

# ---- (1) 경사 하강 한 걸음 손계산:  L(θ) = θ² ,  기울기 dL/dθ = 2θ ----
theta, lr = 3.0, 0.1
grad = 2 * theta                      # 기울기 = 6  (양수 → 오른쪽으로 갈수록 손실이 커진다)
new = theta - lr * grad               # 반대 방향으로 η·기울기 만큼 이동: 3 − 0.1×6 = 2.4
print(f"[GD 한 걸음] θ=3.0 → {new:.1f}")
assert abs(new - 2.4) < 1e-12

# ---- (2) 학습률 η 에 따른 차이:  L(θ)=θ², 시작 θ=3, 20걸음 ----
def run_gd(lr, steps=20, theta=3.0):
    for _ in range(steps):
        theta = theta - lr * 2 * theta        # θ ← θ − η·2θ = (1−2η)·θ
    return theta
for lr in (0.01, 0.1, 0.9, 1.05):
    print(f"[학습률 {lr:>4}] 20걸음 뒤 θ = {run_gd(lr): .4g}   (매 걸음 θ에 (1−2η)={1-2*lr:+.2f} 를 곱함)")
# η=1.05 이면 |1−2η|=1.1 > 1 → 매번 커진다 = 발산

# ---- (3) 지수이동평균(EMA): m ← β·m + (1−β)·x ----
beta = 0.9
w = [(1 - beta) * beta ** k for k in range(4)]            # 현재, 1단계 전, 2단계 전, 3단계 전의 비중
print("[EMA 비중 β=0.9]", [round(x, 4) for x in w])        # 0.1, 0.09, 0.081, 0.0729
assert np.allclose(w[:3], [0.1, 0.09, 0.081])
assert abs(sum((1 - beta) * beta ** k for k in range(200)) - 1) < 1e-8   # 비중을 다 합치면 1 (먼 과거 포함)

rng = np.random.default_rng(0)
signal = 1.0                                               # 진짜 기울기는 늘 +1 인데
g = signal + rng.normal(0, 2.0, 300)                       # 측정값에는 큰 잡음(표준편차 2)이 섞여 있다고 하자
m, ema = 0.0, []
for x in g:
    m = beta * m + (1 - beta) * x; ema.append(m)
print(f"[EMA로 잡음 줄이기] 원래 값의 표준편차 {g[50:].std():.2f} → EMA의 표준편차 {np.std(ema[50:]):.2f}  (평균은 둘 다 ≈ 1)")

# ---- (4) 협곡 모양 손실에서 SGD vs 모멘텀:  L = ½(x² + 100·y²)  (y 방향이 100배 가파르다) ----
def grad_valley(p): return np.array([p[0], 100 * p[1]])
def loss_valley(p): return 0.5 * (p[0] ** 2 + 100 * p[1] ** 2)
def run(opt, steps=100):
    p = np.array([8.0, 1.0]); path = [p.copy()]
    for _ in range(steps):
        opt.step([p], [grad_valley(p)]); path.append(p.copy())
    return np.array(path)
# SGD 는 lr 이 2/100=0.02 를 넘으면 y 방향이 발산하므로 0.019 가 사실상 최대 → x 방향은 느릴 수밖에 없다
for name, opt in [("SGD lr=0.019", SGD(0.019)), ("모멘텀 μ=0.9 lr=0.01", SGD(0.01, 0.9))]:
    P = run(opt)
    flips = int(np.sum(np.sign(P[1:, 1]) != np.sign(P[:-1, 1])))    # y 부호가 몇 번 뒤집혔나 = 지그재그 횟수
    print(f"[{name:<22}] 100걸음 뒤 손실 {loss_valley(P[-1]):.1e} | y 방향 부호 뒤집힘 {flips}번")
