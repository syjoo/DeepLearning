"""Step 4. 학습률 스케줄(웜업·코사인·재시작)과 sharp vs flat minima 를 NumPy로 확인한다.
실행: python step4_schedule_sharp.py
"""
import numpy as np
from optim_np import cosine, warmup_cosine, cosine_restarts

# ---- (1) 코사인 스케줄 값 손계산: η_max=1, T=100 ----
for t in (0, 25, 50, 75, 100):
    print(f"[cosine] t={t:>3}  η = ½(1+cos(π·{t}/100)) = {cosine(t, 100, 1.0):.3f}")
assert abs(cosine(0, 100, 1) - 1) < 1e-12 and abs(cosine(50, 100, 1) - .5) < 1e-12 and abs(cosine(100, 100, 1)) < 1e-12
# 선형 감소는 t=25 에서 0.75, 코사인은 0.854 → 초반엔 천천히 줄고, 중반에 빨리 줄고, 끝에서 다시 천천히 줄어든다
print(f"[선형 vs 코사인] t=25: 선형 {1-25/100:.3f} / 코사인 {cosine(25,100,1):.3f} ;  t=75: 선형 {1-75/100:.3f} / 코사인 {cosine(75,100,1):.3f}")

# ---- (2) 웜업 + 코사인, 그리고 재시작(SGDR) ----
T, warm = 1000, 100
sched = np.array([warmup_cosine(t, T, warm, 1e-3) for t in range(T)])
print(f"[웜업+코사인] 시작 {sched[0]:.1e} → 웜업 끝 {sched[warm-1]:.1e}(최대) → 중간 {sched[550]:.1e} → 끝 {sched[-1]:.1e}")
assert sched.argmax() == warm - 1
rs = np.array([cosine_restarts(t, 250, 1e-3) for t in range(T)])
print(f"[재시작 SGDR] 주기 250: t=0/249/250 의 학습률 = {rs[0]:.1e} / {rs[249]:.1e} / {rs[250]:.1e}  (250 에서 다시 최대로 점프)")

# ---- (3) sharp vs flat minima: 1차원 장난감 손실 ----
# 학습 손실: 좁고 깊은 골짜기(θ=−1.5, 폭 0.08, 깊이 1.0)  +  넓고 약간 얕은 골짜기(θ=+1.5, 폭 0.8, 깊이 0.8)
def L(theta, shift=0.0):
    t = theta - shift                                # shift: '시험 데이터'에서는 손실 표면이 살짝 옆으로 밀려 있다고 가정
    return -1.0 * np.exp(-(t + 1.5) ** 2 / (2 * 0.08 ** 2)) - 0.8 * np.exp(-(t - 1.5) ** 2 / (2 * 0.8 ** 2))
rng = np.random.default_rng(0)
print("\n위치     학습 손실   시험 손실(표면이 0.25 이동)   가중치 잡음(σ=0.2)에서의 평균 손실")
for name, th in [("sharp(−1.5)", -1.5), ("flat (+1.5)", 1.5)]:
    noisy = L(th + rng.normal(0, 0.2, 100000)).mean()
    print(f"{name}  {L(th):9.3f}   {L(th, 0.25):12.3f}                  {noisy:10.3f}")
# sharp 는 학습 손실이 더 낮지만(−1.0 < −0.8), 표면이 조금만 움직이거나 가중치에 잡음이 끼면 이득이 거의 사라진다.

# ---- (4) 곡률(2차 미분)으로 sharp 정도 재기: L'' ≈ (L(θ+h) − 2L(θ) + L(θ−h)) / h² ----
h = 1e-3
for name, th in [("sharp", -1.5), ("flat", 1.5)]:
    print(f"[곡률 L'' {name}] {(L(th+h) - 2*L(th) + L(th-h)) / h**2:9.2f}")
