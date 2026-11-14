"""슬라이드에 넣을 그림을 만든다. 실행: python make_images.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step2_init import forward_stds, cases, tanh, relu, xavier, small, big
from step3_norm import BatchNorm

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
def save(fig, n): fig.tight_layout(); fig.savefig(os.path.join(OUT, n), dpi=200, facecolor="white"); plt.close(fig)
TEAL, ORG, PUR, GRN, RED, INK = "#0E7490", "#C2410C", "#6D28D9", "#15803D", "#B91C1C", "#1F2933"

# 1) 층별 활성값 표준편차 (로그 눈금)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for (name, (f, s)), c in zip(cases.items(), [RED, ORG, TEAL, PUR, GRN]):
    ax.plot(forward_stds(f, s), marker="o", ms=3, label=name, color=c)
ax.set_yscale("log"); ax.set_xlabel("층 번호"); ax.set_ylabel("활성값의 표준편차 (로그)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
save(fig, "act_std.png")

# 2) tanh 활성값 히스토그램: 1, 5, 10번째 층
fig, axes = plt.subplots(3, 3, figsize=(7.4, 4.6), sharey=False)
rows = [("std=0.01", small), ("std=1.0", big), ("Xavier", xavier)]
for r, (nm, fn) in enumerate(rows):
    rng = np.random.default_rng(0); a = rng.normal(size=(1000, 512))
    for L in range(1, 11):
        a = np.tanh(a @ rng.normal(0, fn(512, 512), (512, 512)))
        if L in (1, 5, 10):
            ax = axes[r][[1, 5, 10].index(L)]; ax.hist(a.ravel(), bins=np.linspace(-1, 1, 61), color=TEAL); ax.set_xlim(-1, 1)
            ax.set_title(f"{nm} · {L}번째 층", fontsize=9); ax.set_yticks([])
save(fig, "act_hist.png")

# 3) XOR: 원래 공간 vs 은닉층(ReLU) 이후 공간
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); y = np.array([0, 1, 1, 0])
H = np.maximum(0, X @ np.ones((2, 2)) + np.array([0, -1.0]))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
for ax, P, t in zip(axes, [X, H], ["입력 공간: 직선 하나로 못 가른다", "은닉층 통과 후: 직선으로 가를 수 있다"]):
    for k, c, m, sz in [(1, ORG, "s", 380), (0, TEAL, "o", 150)]:
        ax.scatter(P[y == k, 0], P[y == k, 1], s=sz, color=c, marker=m, label=f"XOR={k}")
    ax.set_title(t, fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2"); axes[1].set_xlabel("h1"); axes[1].set_ylabel("h2")
axes[1].plot([0.5, 2.3], [0, 0.9], color=RED, lw=2, ls="--")
save(fig, "xor.png")

# 4) 활성화 함수 모음 (함수 + 미분)
z = np.linspace(-4, 4, 400); sig = 1 / (1 + np.exp(-z))
from math import erf
gelu = z * 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))
fs = {"Sigmoid": sig, "Tanh": np.tanh(z), "ReLU": np.maximum(0, z), "Leaky ReLU": np.where(z > 0, z, 0.1 * z), "GELU": gelu}
fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharex=True)
for ax, (n, v), c in zip(axes, fs.items(), [PUR, TEAL, GRN, ORG, RED]):
    ax.plot(z, v, color=c, lw=2); ax.axhline(0, color="#999", lw=.6); ax.axvline(0, color="#999", lw=.6); ax.set_title(n, fontsize=11); ax.set_ylim(-1.3, 3)
save(fig, "activations.png")

# 5) 정규분포: 분산이 다르면 퍼짐이 다르다
t = np.linspace(-8, 8, 500)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for v, c in [(0.25, TEAL), (1, GRN), (4, ORG)]:
    ax.plot(t, np.exp(-t**2 / (2 * v)) / np.sqrt(2 * np.pi * v), color=c, lw=2.2, label=f"N(0, {v})  표준편차 {v**.5:g}")
ax.set_xlabel("가중치 값"); ax.set_yticks([]); ax.legend(fontsize=9)
save(fig, "gauss.png")

# 6) BN/LN 이 보는 방향: 표(샘플 N x 특징 C)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, t, kind in zip(axes, ["Batch Norm: 특징 하나의 '세로줄'(배치 방향)", "Layer Norm: 샘플 하나의 '가로줄'(특징 방향)"], ["bn", "ln"]):
    g = np.zeros((6, 6)); (g.__setitem__((slice(None), 2), 1) if kind == "bn" else g.__setitem__((2, slice(None)), 1))
    ax.imshow(g, cmap="Blues", vmin=0, vmax=1.6); ax.set_title(t, fontsize=9.5)
    ax.set_xticks(range(6), [f"특징{i+1}" for i in range(6)], fontsize=7); ax.set_yticks(range(6), [f"샘플{i+1}" for i in range(6)], fontsize=7)
    for i in range(7): ax.axhline(i - .5, color="white", lw=1.5); ax.axvline(i - .5, color="white", lw=1.5)
    for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "bn_ln_table.png")

# 7) BN 효과: 정규화 전/후 분포
rng = np.random.default_rng(1); x = rng.normal(3, 5, (2000, 1))
bn = BatchNorm(1); yb = bn(x); bn.gamma[:] = 2; bn.beta[:] = 1; yg = bn(x)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharex=True)
for ax, d, t, c in zip(axes, [x, yb, yg], ["입력 (평균 3, 표준편차 5)", "정규화 후 (평균 0, 표준편차 1)", "γ=2, β=1 적용 (평균 1, 표준편차 2)"], [ORG, TEAL, PUR]):
    ax.hist(d.ravel(), bins=50, color=c); ax.set_title(t, fontsize=9); ax.set_yticks([]); ax.axvline(d.mean(), color=INK, ls="--", lw=1)
save(fig, "bn_effect.png")
print("done")
