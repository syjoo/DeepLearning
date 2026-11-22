"""실습 문제 5개의 솔루션. 실행: python practice_solutions.py  (모든 assert 통과 = 정답)"""
import numpy as np
from optim_np import Adam, Lion, SGD

# ---------- 문제 1. 경사 하강 손계산 ----------
# L(θ) = (θ−4)², θ0 = 0.  기울기 = 2(θ−4).  η = 0.25, 1.0, 1.1 로 2걸음씩 움직여라.
def gd(eta, steps=2, th=0.0):
    out = [th]
    for _ in range(steps):
        th = th - eta * 2 * (th - 4); out.append(th)
    return out
print("문제1  η=0.25:", gd(0.25), " η=1.0:", gd(1.0), " η=1.1:", [round(x, 2) for x in gd(1.1)])
assert gd(0.25) == [0.0, 2.0, 3.0]            # 0 → 0+0.25×8=2 → 2+0.25×4=3 (정답 4 에 다가간다)
assert gd(1.0) == [0.0, 8.0, 0.0]             # 정답을 넘어 8 로 갔다가 다시 0: 제자리 진동
e = [abs(x - 4) for x in gd(1.1, 5)]; assert all(e[i + 1] > e[i] for i in range(5))   # (1−2η)=−1.2 : 정답과의 거리(4, 4.8, 5.76, …)가 매번 1.2배로 커진다 = 발산

# ---------- 문제 2. EMA 와 편향 보정 손계산 ----------
# 기울기가 3번 연속 g=1 이다. β1=0.9, m0=0 일 때 m1, m2, m3 와 보정값 m̂ 을 구하라.
m, ms, mh = 0.0, [], []
for t in (1, 2, 3):
    m = 0.9 * m + 0.1 * 1.0; ms.append(m); mh.append(m / (1 - 0.9 ** t))
print("문제2  m =", np.round(ms, 3), " m̂ =", np.round(mh, 3))
assert np.allclose(ms, [0.1, 0.19, 0.271]) and np.allclose(mh, [1, 1, 1])   # 보정하면 진짜 평균(1)이 곧바로 나온다

# ---------- 문제 3. 기울기 크기가 다른 두 파라미터의 첫 걸음 (η=0.01, g=[0.02, −5]) ----------
g = np.array([0.02, -5.0])
p = [np.zeros(2)]; SGD(0.01).step(p, [g]);  s_sgd = p[0].copy()
p = [np.zeros(2)]; Adam(0.01).step(p, [g]); s_adam = p[0].copy()
p = [np.zeros(2)]; Lion(0.01).step(p, [g]); s_lion = p[0].copy()
print("문제3  SGD", s_sgd, " Adam", s_adam.round(4), " Lion", s_lion)
assert np.allclose(s_sgd, [-0.0002, 0.05])                    # SGD: 기울기에 비례 → 크기 차이가 25000배
assert np.allclose(s_adam, [-0.01, 0.01], atol=1e-6)          # Adam(첫 걸음): 크기가 정규화되어 ±η
assert np.allclose(s_lion, [-0.01, 0.01])                     # Lion: 부호만 → ±η

# ---------- 문제 4. 옵티마이저 상태 메모리 ----------
# (a) 1024×4096 행렬 하나: Adam 의 v vs Adafactor 의 R, C.   (b) 70억(7e9) 파라미터, float32: Adam / Lion 상태 크기
n, m = 1024, 4096
print(f"문제4a Adam v {n*m:,}개 vs Adafactor {n+m:,}개 → {n*m/(n+m):.0f}배")
assert n * m == 4_194_304 and n + m == 5_120 and round(n * m / (n + m)) == 819
P = 7e9
adam_gb, lion_gb = 2 * P * 4 / 1e9, 1 * P * 4 / 1e9
print(f"문제4b Adam(m,v) {adam_gb:.0f} GB, Lion(m) {lion_gb:.0f} GB  (파라미터 자체 {P*4/1e9:.0f} GB 는 별도)")
assert adam_gb == 56 and lion_gb == 28

# ---------- 문제 5. 구현: 웜업 + 코사인 학습률 ----------
def warmup_cosine_lr(t, T, warm, lr_max):
    """t 걸음째 학습률. 처음 warm 걸음은 lr_max·(t+1)/warm 으로 직선 상승, 이후 0 까지 코사인 감소."""
    if t < warm:
        return lr_max * (t + 1) / warm
    return lr_max * 0.5 * (1 + np.cos(np.pi * (t - warm) / (T - warm)))
assert abs(warmup_cosine_lr(0, 100, 10, 1.0) - 0.1) < 1e-12        # 첫 걸음은 최대치의 1/10
assert abs(warmup_cosine_lr(9, 100, 10, 1.0) - 1.0) < 1e-12        # 웜업 마지막 걸음 = 최대
assert abs(warmup_cosine_lr(10, 100, 10, 1.0) - 1.0) < 1e-12       # 코사인 시작 = 최대
assert abs(warmup_cosine_lr(55, 100, 10, 1.0) - 0.5) < 1e-12       # 코사인 중간 = 절반
assert warmup_cosine_lr(99, 100, 10, 1.0) < 1e-3                   # 끝에서는 0 근처
print("문제5  warmup_cosine_lr 검사 통과")
print("\n모든 솔루션 검사 통과")
