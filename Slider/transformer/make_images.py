"""슬라이드에 넣을 그림(위치 인코딩 히트맵, 어텐션 히트맵)을 만든다."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "numpy_version"))
from step2_positional_encoding import positional_encoding
from step3_attention import scaled_dot_product_attention, causal_mask

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
out = os.path.join(HERE, "images")

# 1) 위치 인코딩 히트맵
pe = positional_encoding(32, 32)
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
im = ax.imshow(pe, cmap="RdBu_r", aspect="auto")
ax.set_xlabel("벡터 차원 (d_model)")
ax.set_ylabel("토큰 위치 (pos)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(out, "pe_heatmap.png"), facecolor="white")

# 2) 어텐션 가중치 히트맵: 마스크 없음 vs 마스크 있음
words = ["나는", "AI를", "배우는", "학생이다"]
rng = np.random.default_rng(3)
X = rng.normal(size=(4, 8))
W = [rng.normal(size=(8, 4)) * 0.45 for _ in range(3)]
Q, K, V = (X @ w for w in W)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=200)
for ax, mask, title in zip(axes, [None, causal_mask(4)], ["마스크 없음 (인코더)", "인과 마스크 (디코더)"]):
    _, w = scaled_dot_product_attention(Q, K, V, mask)
    ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), words)
    ax.set_yticks(range(4), words)
    ax.set_title(title)
    ax.set_xlabel("참고 대상 (Key)")
    ax.set_ylabel("나 (Query)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center",
                    color="white" if w[i, j] > 0.55 else "black", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, "attention_heatmap.png"), facecolor="white")
print("saved")
