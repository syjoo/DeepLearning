"""Step 3. Lion(부호 기반)과 Adafactor(메모리 절약)를 NumPy로 구현하고 성질을 확인한다.
실행: python step3_lion_adafactor.py
"""
import numpy as np
from optim_np import Lion, Adafactor, Adam, SGD, cosine

# ---- (1) Lion: 업데이트 크기는 기울기 크기와 상관없이 늘 η ----
for gscale in (1e-3, 1.0, 1e3):
    p = [np.zeros(4)]; opt = Lion(lr=0.01)
    opt.step(p, [gscale * np.array([1.0, -2.0, 0.5, 3.0])])
    print(f"[Lion] 기울기 크기 ×{gscale:g} → 이동량 {np.abs(p[0]).round(4)}")   # 항상 [0.01 0.01 0.01 0.01]
    assert np.allclose(np.abs(p[0]), 0.01)

# ---- (2) Lion 의 두 β 역할 확인: 첫 걸음은 부호(1−β1)·g 의 부호 = g 의 부호, 모멘텀은 β2 로 갱신 ----
p = [np.zeros(1)]; opt = Lion(lr=0.1, betas=(0.9, 0.99)); opt.step(p, [np.array([2.0])])
print(f"[Lion 첫 걸음] θ={p[0][0]:+.2f}  (부호만 쓰므로 −η),  m={opt.m[0][0]:.3f} (= (1−β2)·g = 0.01×2)")
assert abs(opt.m[0][0] - 0.02) < 1e-12

# ---- (3) Adafactor: 행·열 합 두 벡터로 v 행렬을 복원 ----
rng = np.random.default_rng(0)
a, b = rng.random(6) + .1, rng.random(5) + .1
V = np.outer(a, b)                                  # 랭크 1 행렬(= 행 패턴 × 열 패턴)
R, C = V.sum(1), V.sum(0)
V_hat = np.outer(R, C) / R.sum()
print(f"[Adafactor 복원] 랭크1 행렬은 정확히 복원: 최대 오차 {np.abs(V - V_hat).max():.1e}")
assert np.allclose(V, V_hat)
V2 = rng.random((6, 5)) + 0.1                        # 임의의 행렬은 근사
V2_hat = np.outer(V2.sum(1), V2.sum(0)) / V2.sum()
print(f"[Adafactor 복원] 일반 행렬은 근사: 평균 상대 오차 {np.mean(np.abs(V2 - V2_hat) / V2):.2f}")

# ---- (4) 메모리: 4096×4096 가중치 행렬 하나에 필요한 '2차 모멘트' 숫자의 개수 ----
n = m = 4096
print(f"[메모리] Adam v: {n*m:,}개  vs  Adafactor R+C: {n+m:,}개  → {n*m/(n+m):.0f}배 적음  (O(n·m) → O(n+m))")
print(f"        float32 기준: Adam v {n*m*4/2**20:.0f} MiB  vs  Adafactor {(n+m)*4/1024:.0f} KiB")

# ---- (5) Adafactor 가 실제로 학습하는지: 행렬 W 를 목표 행렬 T 에 맞추는 문제 ----
T = rng.normal(size=(20, 30)); loss = lambda W: 0.5 * ((W - T) ** 2).sum()
for name, mk in [("Adam", lambda: Adam(0.05)), ("Adafactor", lambda: Adafactor(0.05)), ("Lion", lambda: Lion(0.05)), ("SGD", lambda: SGD(0.1))]:
    W = np.zeros((20, 30)); opt = mk()
    lr0 = opt.lr
    for t in range(200):
        opt.lr = cosine(t, 200, lr0)                # 학습률을 코사인으로 줄여 간다 (Lion 은 보폭이 고정이라 줄여야 바닥에 안착)
        opt.step([W], [W - T])
    extra = {"Adam": "m,v = 2×600", "Adafactor": f"R,C = {sum(x.size for x in opt.s[0])}" if name == "Adafactor" else "", "Lion": "m = 600", "SGD": "없음"}[name]
    print(f"[W→T 200걸음 {name:<9}] 손실 {loss(W):9.3f}  (처음 {loss(np.zeros_like(T)):.0f}) | 추가 상태 {extra}")
