"""슬라이드에 넣을 그림을 만든다. 먼저 step4_compare.py를 실행해 _res.npy를 만든 뒤: python make_images.py"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step1_transforms import load, flip_h, rotate, shift, zoom, brightness_contrast, add_noise
from step3_pipeline import RandomAffine, GaussianNoise, Compose
from step5_advanced import mixup, cutmix, cutout

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
def save(fig, n): fig.tight_layout(); fig.savefig(os.path.join(OUT, n), dpi=200, facecolor="white"); plt.close(fig)
TEAL, ORG, PUR, GRN, RED, INK = "#0E7490", "#C2410C", "#6D28D9", "#15803D", "#B91C1C", "#1F2933"
def show(ax, im, title="", color=INK):
    ax.imshow(im, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#CBD5DC")
    ax.set_title(title, fontsize=10, color=color)
X, y = load(); rng = np.random.default_rng(3)
d3 = X[np.where(y == 3)[0][2]]

# 1) 변환 모음
ts = [("원본", d3), ("좌우 뒤집기", flip_h(d3)), ("이동 (→1, ↓1)", shift(d3, 1, 1)), ("회전 +15°", rotate(d3, 15)),
      ("확대 ×1.3", zoom(d3, 1.3)), ("밝기 +0.2", brightness_contrast(d3, b=0.2)), ("대비 ×1.5", brightness_contrast(d3, c=1.5)), ("노이즈 σ=0.15", add_noise(d3, .15, rng))]
fig, axes = plt.subplots(2, 4, figsize=(8, 4.3))
for ax, (t, im) in zip(axes.ravel(), ts): show(ax, im, t)
save(fig, "transforms.png")

# 2) 레이블이 바뀌는 변환
six = X[np.where(y == 6)[0][0]]; nine = X[np.where(y == 9)[0][0]]; two = X[np.where(y == 2)[0][0]]
fig, axes = plt.subplots(1, 5, figsize=(9, 2.5))
for ax, (t, im, c) in zip(axes, [("원본: 6", six, INK), ("180° 회전 → 정답은 여전히 6?", rotate(six, 180), RED), ("진짜 9", nine, GRN), ("원본: 2", two, INK), ("좌우 뒤집기 → 아직 2?", flip_h(two), RED)]):
    show(ax, im, t, c); ax.title.set_fontsize(8.5)
save(fig, "label_break.png")

# 3) 온라인 증강: 같은 이미지가 매번 다르게 (약한 증강)
torch.manual_seed(1); x0 = torch.tensor(d3)[None, None].repeat(11, 1, 1, 1)
mild = Compose([RandomAffine(10, 1), GaussianNoise(0.05)])(x0)
fig, axes = plt.subplots(2, 6, figsize=(8.4, 3.0))
for k, ax in enumerate(axes.ravel()): show(ax, d3 if k == 0 else mild[k - 1, 0].numpy(), "원본" if k == 0 else f"{k}번째", ORG if k == 0 else INK)
save(fig, "online_samples.png")

# 4) 약함 vs 과함
torch.manual_seed(2); ids = [np.where(y == c)[0][1] for c in (0, 4, 7, 8)]
xs = torch.tensor(X[ids]).unsqueeze(1)
mi, st = Compose([RandomAffine(10, 1), GaussianNoise(0.05)])(xs), Compose([RandomAffine(60, 3), GaussianNoise(0.4)])(xs)
fig, axes = plt.subplots(3, 4, figsize=(6.4, 5.0))
for r, (nm, T, c) in enumerate([("원본", xs, INK), ("약한 증강", mi, GRN), ("너무 센 증강", st, RED)]):
    for k in range(4): show(axes[r][k], T[k, 0].numpy(), nm if k == 0 else "", c)
save(fig, "mild_vs_strong.png")

# 5) 결과 비교 (step4_compare 5회 평균)
R = np.load("_res.npy"); names = ["증강 없음", "약한 증강\n(온라인)", "오프라인\n5배", "너무 센\n증강", "좌우 뒤집기\n(나쁜 예)"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
w = .38; xi = np.arange(5)
axes[0].bar(xi - w / 2, R[:, 0], w, color=TEAL, label="학습 손실"); axes[0].bar(xi + w / 2, R[:, 2], w, color=ORG, label="검증 손실")
axes[0].set_xticks(xi, names, fontsize=8); axes[0].set_ylim(0, 1.0); axes[0].set_title("손실 (작을수록 좋다)", fontsize=11); axes[0].legend(fontsize=8, loc="upper left")
axes[0].text(3, .5, f"{R[3,0]:.1f} / {R[3,2]:.1f}\n(그래프 밖으로 넘침)", fontsize=8, ha="center", color="white")
axes[1].bar(xi - w / 2, R[:, 3] * 100, w, color=TEAL, label="깨끗한 검증 이미지"); axes[1].bar(xi + w / 2, R[:, 5] * 100, w, color=PUR, label="변형·노이즈 낀 검증 이미지")
for i in xi: axes[1].text(i + w / 2, R[i, 5] * 100 + 1, f"{R[i,5]*100:.0f}", ha="center", fontsize=8); axes[1].text(i - w / 2, R[i, 3] * 100 + 1, f"{R[i,3]*100:.0f}", ha="center", fontsize=8)
axes[1].set_xticks(xi, names, fontsize=8); axes[1].set_ylim(0, 110); axes[1].set_title("정확도 % (클수록 좋다)", fontsize=11); axes[1].legend(fontsize=8, loc="upper right")
save(fig, "aug_compare.png")

# 6) Cutout / Mixup / CutMix
np.random.seed(5); torch.manual_seed(5)
a, b = torch.tensor(X[np.where(y == 3)[0][2]])[None, None], torch.tensor(X[np.where(y == 8)[0][1]])[None, None]
xx = torch.cat([a, b]); mx, lm, _, _ = mixup(xx, torch.tensor([3, 8]), alpha=4.0)
cm, lc, _, _ = cutmix(xx, torch.tensor([3, 8])); co = a.clone(); co[:, :, 2:5, 3:6] = 0                        # 시연용: 획 위에 3x3 구멍 (cutout()은 위치가 무작위)
fig, axes = plt.subplots(1, 5, figsize=(9, 2.5))
for ax, (t, im) in zip(axes, [("이미지 A (3)", a[0, 0]), ("이미지 B (8)", b[0, 0]), ("Cutout\n(A에 구멍)", co[0, 0]), (f"Mixup\n({lm:.2f}·A + {1-lm:.2f}·B)", mx[0, 0]), (f"CutMix\n(A {lc:.2f} + B 조각)", cm[0, 0])]):
    show(ax, np.asarray(im), t); ax.title.set_fontsize(8.5)
save(fig, "mixup.png")
print("done")
