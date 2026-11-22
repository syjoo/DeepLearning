"""Step 2. Adam / AdamW 를 NumPy로 구현하고 손계산·PyTorch 와 맞춰 본다.
실행: python step2_adam.py
"""
import numpy as np, torch
from optim_np import SGD, Adam

# ---- (1) 첫 걸음 손계산: θ=1, g=0.5, η=0.1, β1=0.9, β2=0.999 ----
g, b1, b2, lr, eps = 0.5, 0.9, 0.999, 0.1, 1e-8
m = b1 * 0 + (1 - b1) * g              # m = 0.1 × 0.5 = 0.05         (과거 m=0 이었으므로 0.05 로 '작게' 시작)
v = b2 * 0 + (1 - b2) * g * g          # v = 0.001 × 0.25 = 0.00025
step_raw = lr * m / (np.sqrt(v) + eps)             # 보정 전: 0.1 × 0.05/0.01581 ≈ 0.316  (η 의 3배 넘게 점프!)
mh, vh = m / (1 - b1), v / (1 - b2)                # 편향 보정: 1−β1^1 = 0.1, 1−β2^1 = 0.001 로 나눠서 0.5, 0.25 로 복원
step_fix = lr * mh / (np.sqrt(vh) + eps)           # 보정 후: 0.1 × 0.5/0.5 = 0.1  (첫 걸음은 정확히 η)
print(f"[Adam 첫 걸음] m={m:.3f} v={v:.5f} | 보정 전 이동량 {step_raw:.3f} | 보정 후 m̂={mh:.2f} v̂={vh:.2f} 이동량 {step_fix:.3f}")
assert abs(step_fix - 0.1) < 1e-6

# ---- (2) '파라미터마다 학습률이 자동 조절'되는지: 기울기 크기가 100배 다른 두 파라미터 ----
rng = np.random.default_rng(0)
for name, opt in [("SGD ", SGD(0.01)), ("Adam", Adam(0.01))]:
    p = [np.array([0.0, 0.0])]
    scale = np.array([10.0, 0.1])                           # 첫 번째 파라미터의 기울기는 두 번째의 100배
    for _ in range(100):
        opt.step(p, [scale * (1 + 0.1 * rng.normal(size=2))])   # 기울기는 늘 양수 → 같은 방향으로 이동
    print(f"[100걸음 이동량 {name}] 파라미터A(기울기≈10): {-p[0][0]:.2f}   파라미터B(기울기≈0.1): {-p[0][1]:.2f}")
# SGD 는 100:1 로 이동, Adam 은 둘 다 거의 같은 거리(≈ η×100 = 1) 이동한다.

# ---- (3) 협곡에서 4가지 비교:  L = ½(x² + 100 y²) ----
K = 100
grad_v = lambda p: np.array([p[0], K * p[1]]); loss_v = lambda p: 0.5 * (p[0] ** 2 + K * p[1] ** 2)
def run(opt, n=100):
    p = np.array([8.0, 1.0]); path = [p.copy()]
    for _ in range(n):
        opt.step([p], [grad_v(p)]); path.append(p.copy())
    return np.array(path)
if __name__ == "__main__":
    for name, opt in [("SGD (0.019)", SGD(0.019)), ("모멘텀 (0.01, μ=0.9)", SGD(0.01, 0.9)), ("Adam (0.3)", Adam(0.3))]:
        P = run(opt); print(f"[협곡 {name:<18}] 100걸음 뒤 손실 {loss_v(P[-1]):.1e}")

# ---- (4) AdamW vs Adam+L2: '기울기에 잡음만 있는' 두 그룹의 파라미터가 가중치 감쇠로 얼마나 줄어드는가 ----
# 각 그룹 1000개, 모두 θ=1 에서 시작. 진짜 기울기는 0(잡음 평균 0), 잡음 크기만 100배 다르다 (σ=1 vs σ=0.01).
# (한 개씩만 보면 잡음 때문에 들쭉날쭉하므로 1000개의 평균을 본다)
def decay_demo(decoupled):
    r = np.random.default_rng(1)
    opt = Adam(1e-2, weight_decay=0.1, decoupled=decoupled); p = [np.ones(2000)]
    sig = np.r_[np.ones(1000), 0.01 * np.ones(1000)]
    for _ in range(500):
        opt.step(p, [r.normal(size=2000) * sig])
    return p[0][:1000].mean(), p[0][1000:].mean()
for name, dec in [("Adam + L2", False), ("AdamW     ", True)]:
    q = decay_demo(dec); print(f"[500걸음 뒤 θ의 평균 {name}] 큰 잡음(σ=1): {q[0]: .3f}   작은 잡음(σ=0.01): {q[1]: .3f}")
# AdamW: 두 그룹이 똑같이 줄어든다  (1−η·λ)^500 = 0.999^500 ≈ 0.606
q = decay_demo(True); assert abs(q[0] - 0.606) < 0.03 and abs(q[1] - 0.606) < 0.03

# ---- (5) NumPy 구현이 torch.optim 과 같은 값을 내는가? (같은 기울기를 50번 먹여 본다) ----
def check(make_np, make_torch, tag):
    r = np.random.default_rng(0); w0 = r.normal(size=(3, 4)); G = r.normal(size=(50, 3, 4))
    a = [w0.copy()]; opt = make_np()
    t = torch.nn.Parameter(torch.tensor(w0)); topt = make_torch([t])
    for k in range(50):
        opt.step(a, [G[k].copy()]); t.grad = torch.tensor(G[k]); topt.step()
    err = np.abs(a[0] - t.detach().numpy()).max()
    print(f"[NumPy vs torch.optim] {tag:<22} 최대 차이 {err:.2e}"); assert err < 1e-9
check(lambda: SGD(0.1, 0.9),  lambda p: torch.optim.SGD(p, 0.1, momentum=0.9), "SGD + momentum")
check(lambda: Adam(1e-2),     lambda p: torch.optim.Adam(p, 1e-2),            "Adam")
check(lambda: Adam(1e-2, weight_decay=0.1, decoupled=True), lambda p: torch.optim.AdamW(p, 1e-2, weight_decay=0.1), "AdamW")
check(lambda: Adam(1e-2, weight_decay=0.1), lambda p: torch.optim.Adam(p, 1e-2, weight_decay=0.1), "Adam + L2")
