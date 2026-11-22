"""슬라이드에 넣을 그림을 만든다. 먼저 step6_torch_compare.py 를 실행해 _res6.npy 를 만든 뒤: python make_images.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optim_np import SGD, Adam, Lion, cosine, warmup_cosine, cosine_restarts

plt.rcParams.update({"font.family": "Malgun Gothic", "mathtext.fontset": "dejavusans", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
def save(fig, n): fig.tight_layout(); fig.savefig(os.path.join(OUT, n), dpi=200, facecolor="white"); plt.close(fig)
TEAL, ORG, PUR, GRN, RED, INK, GRY = "#0E7490", "#C2410C", "#6D28D9", "#15803D", "#B91C1C", "#1F2933", "#8896A0"

# 1) 학습률의 효과: L=θ², 시작 θ=3
fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
xs = np.linspace(-4.5, 4.5, 200)
for ax, (lr, name, c) in zip(axes, [(0.01, "너무 작다", ORG), (0.1, "적당하다", GRN), (0.9, "너무 크다(진동)", ORG), (1.05, "더 크다(발산)", RED)]):
    th, path = 3.0, [3.0]
    for _ in range(12): th -= lr * 2 * th; path.append(th)
    path = np.array(path); ax.plot(xs, xs ** 2, color=GRY, lw=1.2); ax.plot(path, path ** 2, "o-", color=c, ms=4, lw=1)
    ax.set_title(f"η={lr}  {name}", fontsize=10, color=c); ax.set_ylim(0, 30); ax.set_xlim(-4.5, 4.5); ax.set_xlabel("θ", fontsize=9)
axes[0].set_ylabel("손실 L(θ)")
save(fig, "lr_effect.png")

# 2) 협곡 궤적: L = ½(x² + 100 y²)
K = 100
def run(opt, n=100):
    p = np.array([8.0, 1.0]); P = [p.copy()]
    for _ in range(n): opt.step([p], [np.array([p[0], K * p[1]])]); P.append(p.copy())
    return np.array(P)
cases = [("SGD (η=0.019)", SGD(0.019), ORG), ("모멘텀 (η=0.01, μ=0.9)", SGD(0.01, 0.9), TEAL), ("Adam (η=0.3)", Adam(0.3), PUR)]
gx, gy = np.meshgrid(np.linspace(-1, 9, 200), np.linspace(-1.3, 1.3, 200))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, (name, opt, c) in zip(axes, cases):
    P = run(opt); L = 0.5 * (P[-1, 0] ** 2 + K * P[-1, 1] ** 2)
    ax.contour(gx, gy, 0.5 * (gx ** 2 + K * gy ** 2), levels=[0.5, 2, 8, 20, 50, 100, 200], colors="#CBD5DC", linewidths=0.9)
    ax.plot(P[:, 0], P[:, 1], "o-", color=c, ms=2.5, lw=0.9); ax.plot(8, 1, "s", color=INK, ms=5)
    ax.set_title(f"{name}\n100걸음 뒤 손실 {L:.1e}", fontsize=10, color=c); ax.set_xlabel("x (완만한 방향)", fontsize=9)
axes[0].set_ylabel("y (가파른 방향)")
save(fig, "valley.png")

# 3) EMA: 잡음 낀 신호 + β 비교, 가중치 막대
rng = np.random.default_rng(0); sig = np.where(np.arange(200) < 100, 1.0, -0.5); g = sig + rng.normal(0, 1.0, 200)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
axes[0].plot(g, color="#CBD5DC", lw=1, label="잡음 낀 기울기 g"); axes[0].plot(sig, color=INK, lw=1.2, ls="--", label="진짜 값")
for b, c in [(0.5, ORG), (0.9, TEAL), (0.99, PUR)]:
    m, e = 0.0, []
    for x in g: m = b * m + (1 - b) * x; e.append(m)
    axes[0].plot(e, color=c, lw=1.8, label=f"EMA β={b}")
axes[0].legend(fontsize=8, frameon=False, ncol=2); axes[0].set_xlabel("스텝"); axes[0].set_title("β가 클수록 매끄럽지만 변화에 늦게 반응", fontsize=10)
k = np.arange(10); axes[1].bar(k, 0.1 * 0.9 ** k, color=TEAL); axes[1].set_xlabel("몇 걸음 전의 기울기인가"); axes[1].set_ylabel("반영 비중")
for i in range(3): axes[1].text(i, 0.1 * 0.9 ** i + 0.003, f"{0.1*0.9**i:.3f}", ha="center", fontsize=8)
axes[1].set_title("β=0.9 : 0.1, 0.09, 0.081, …", fontsize=10)
save(fig, "ema.png")

# 4) 기울기 크기에 따른 '한 걸음 크기' (첫 걸음, η=0.01)
gs = np.logspace(-3, 3, 50); st = {"SGD": [], "Adam": [], "Lion": []}
for gv in gs:
    for nm, mk in [("SGD", SGD(0.01)), ("Adam", Adam(0.01)), ("Lion", Lion(0.01))]:
        p = [np.zeros(1)]; mk.step(p, [np.array([gv])]); st[nm].append(abs(p[0][0]))
fig, ax = plt.subplots(figsize=(6, 3.4))
for nm, c in [("SGD", ORG), ("Adam", PUR), ("Lion", GRN)]: ax.loglog(gs, st[nm], color=c, lw=2.2 if nm != "Lion" else 1.4, ls="-" if nm != "Lion" else "--", label=nm)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ax.minorticks_off()
ax.set_xlabel("기울기 크기 |g|"); ax.set_ylabel("첫 걸음 이동량"); ax.legend(frameon=False); ax.set_title("SGD는 기울기에 비례, Adam·Lion은 η 로 일정 (η=0.01)", fontsize=10)
save(fig, "step_size.png")

# 5) 편향 보정: g=1 이 계속 들어올 때 m 과 m̂
t = np.arange(1, 31); m = 1 - 0.9 ** t
fig, ax = plt.subplots(figsize=(6, 3.2)); ax.plot(t, m, "o-", ms=3, color=ORG, label="보정 전 m"); ax.plot(t, m / (1 - 0.9 ** t), color=TEAL, lw=2, label=r"보정 후 $hat{m}$"); ax.axhline(1, color=GRY, ls=":")
ax.set_xlabel("스텝 t"); ax.set_ylabel("값 (진짜 평균 = 1)"); ax.legend(frameon=False); ax.set_title("m 은 0에서 출발해 초반에 작게 나온다 (0.1, 0.19, 0.271…)", fontsize=10)
save(fig, "bias_correction.png")

# 6) Adam+L2 vs AdamW 감쇠 (step2 와 같은 실험: 500걸음 뒤 θ 평균)
fig, ax = plt.subplots(figsize=(6, 3.2)); w = 0.35
def decay(dec):
    r = np.random.default_rng(1)                      # step2_adam.py 와 같은 난수
    opt = Adam(1e-2, weight_decay=0.1, decoupled=dec); p = [np.ones(2000)]; sig = np.r_[np.ones(1000), 0.01 * np.ones(1000)]
    for _ in range(500): opt.step(p, [r.normal(size=2000) * sig])
    return p[0][:1000].mean(), p[0][1000:].mean()
a, b = decay(False), decay(True)
ax.bar([0 - w / 2, 1 - w / 2], [a[0], b[0]], w, color=ORG, label="기울기 잡음 큼 (σ=1)"); ax.bar([0 + w / 2, 1 + w / 2], [a[1], b[1]], w, color=TEAL, label="기울기 잡음 작음 (σ=0.01)")
for x, v in [(-w / 2, a[0]), (w / 2, a[1]), (1 - w / 2, b[0]), (1 + w / 2, b[1])]: ax.text(x, max(v, 0) + 0.02, f"{abs(v):.2f}", ha="center", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Adam + L2", "AdamW"]); ax.set_ylabel("500걸음 뒤 θ 평균 (처음 1.0)"); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
save(fig, "adamw_decay.png")

# 7) 학습률 스케줄 모음
T = 1000; tt = np.arange(T)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
axes[0].plot(tt, np.ones(T), color=GRY, label="고정"); axes[0].plot(tt, 1 - tt / T, color=ORG, label="선형 감소"); axes[0].plot(tt, cosine(tt, T, 1.0), color=TEAL, lw=2, label="코사인 감쇠"); axes[0].set_title("① 감소 방식 비교", fontsize=10)
axes[1].plot(tt, [warmup_cosine(t, T, 100, 1.0) for t in tt], color=PUR, lw=2); axes[1].axvline(100, color=GRY, ls=":"); axes[1].set_title("② 웜업(100) + 코사인", fontsize=10)
axes[2].plot(tt, [cosine_restarts(t, 250, 1.0) for t in tt], color=GRN, lw=2); axes[2].set_title("③ 코사인 + 재시작 (SGDR, 주기 250)", fontsize=10)
axes[0].legend(frameon=False, fontsize=8); axes[0].set_ylabel("학습률 / 최댓값")
for a_ in axes: a_.set_xlabel("스텝")
save(fig, "schedules.png")

# 8) sharp vs flat minima (step4 와 같은 함수)
def L(th, s=0.0): t = th - s; return -np.exp(-(t + 1.5) ** 2 / (2 * 0.08 ** 2)) - 0.8 * np.exp(-(t - 1.5) ** 2 / (2 * 0.8 ** 2))
x = np.linspace(-3, 3, 1500)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
axes[0].plot(x, L(x), color=INK, lw=2); axes[0].plot([-1.5, 1.5], [L(-1.5), L(1.5)], "o", color=RED, ms=7)
axes[0].annotate("sharp\n(깊지만 좁다)", (-1.5, L(-1.5)), (-2.9, -0.55), fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
axes[0].annotate("flat\n(조금 얕지만 넓다)", (1.5, L(1.5)), (-0.9, -0.62), fontsize=9, color=GRN, arrowprops=dict(arrowstyle="->", color=GRN))
axes[0].set_title("학습 데이터의 손실 표면", fontsize=10); axes[0].set_xlabel("θ"); axes[0].set_ylabel("손실")
axes[1].plot(x, L(x), color=GRY, lw=1.2, label="학습"); axes[1].plot(x, L(x, 0.25), color=INK, lw=2, label="시험 (표면이 0.25 이동)"); axes[1].plot([-1.5, 1.5], [L(-1.5, .25), L(1.5, .25)], "o", color=RED, ms=7)
axes[1].legend(frameon=False, fontsize=8, loc="lower center"); axes[1].set_title("시험 데이터: 같은 θ 에서의 손실", fontsize=10); axes[1].set_xlabel("θ")
save(fig, "sharp_flat.png")

# 9) 종합 실험 결과 (step6 의 _res6.npy)
R = np.load("_res6.npy", allow_pickle=True).item()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
for (nm, c) in [("SGD+모멘텀", ORG), ("AdamW", TEAL), ("Lion", PUR), ("Adafactor", GRN)]: axes[0].semilogy(np.arange(1, 41), R[nm]["curve"], color=c, lw=2, label=nm)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); axes[0].minorticks_off(); axes[0].set_xlabel("에포크"); axes[0].set_ylabel("학습 손실 (로그 눈금)"); axes[0].legend(frameon=False, fontsize=9); axes[0].set_title("학습 손실 (5 seeds 평균)", fontsize=10)
names = ["SGD+모멘텀", "AdamW", "Lion", "Adafactor"]; mu = [R[n]["acc"].mean() for n in names]; sd = [R[n]["acc"].std() for n in names]
axes[1].bar(range(4), mu, yerr=sd, color=[ORG, TEAL, PUR, GRN], capsize=4); axes[1].set_ylim(94, 99); axes[1].set_xticks(range(4)); axes[1].set_xticklabels(names, fontsize=8)
axes[1].set_ylabel("시험 정확도 (%)"); axes[1].set_title("시험 정확도 (막대 = 평균, 선 = ±1σ)", fontsize=10)
for i, v in enumerate(mu): axes[1].text(i, 94.2, f"{v:.1f}", ha="center", color="white", fontsize=9, fontweight="bold")
save(fig, "compare.png")
print("figures saved")
