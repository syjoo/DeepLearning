"""슬라이드 그림 만들기. 먼저 step1, step2, step3, step4, step5, step6 을 한 번씩 실행해 _*.npy / _models.pt 를 만든 뒤: python make_images.py"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step3_gradcam_numpy import K as KER
import step4_gradcam_torch as s4

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
def save(fig, n): fig.tight_layout(); fig.savefig(os.path.join(OUT, n), dpi=200, facecolor="white"); plt.close(fig)
ld = lambda n: np.load(n, allow_pickle=True).item()
TEAL, ORG, PUR, GRN, RED, INK, GRY = "#0E7490", "#C2410C", "#6D28D9", "#15803D", "#B91C1C", "#1F2933", "#9AA5AD"

# 1) 순열 중요도 (학생 합격 예측)
r = ld("_perm.npy"); names = ["공부시간", "출석률", "수면시간", "키(무관)"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].barh(names[::-1], r["m"][::-1], xerr=r["s"][::-1], color=[GRY, ORG, TEAL, TEAL][::-1]); ax[0].set_xlabel("정확도 하락폭  Δ = M0 - M_i"); ax[0].set_title("순열 중요도 (M0 = %.3f)" % r["M0"], fontsize=11)
n2 = names + ["공부시간(복사본)"]; ax[1].barh(n2[::-1], r["m2"][::-1], color=[PUR, GRY, ORG, TEAL, TEAL][::-1]); ax[1].set_xlabel("정확도 하락폭 Δ"); ax[1].set_title("복사본 열을 붙이면: 중요도가 둘로 쪼개진다", fontsize=11)
save(fig, "perm_importance.png")                                                            # 9 x 3.4

# 2) SHAP: 학생 1명 폭포 + 비용
r = ld("_shap.npy"); fn = ["공부시간", "출석률", "수면시간", "키(무관)"]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
cur = r["f0"]; ax[0].axvline(r["f0"], color=GRY, ls=":"); 
order = np.argsort(-np.abs(r["phi"]))
for k, i in enumerate(order):
    ax[0].barh(k, r["phi"][i], left=cur, color=GRN if r["phi"][i] > 0 else RED); ax[0].text(cur + r["phi"][i] + (0.12 if r["phi"][i] > 0 else -0.12), k, "%+.2f" % r["phi"][i], va="center", ha="left" if r["phi"][i] > 0 else "right", fontsize=10); cur += r["phi"][i]
ax[0].set_yticks(range(4)); ax[0].set_yticklabels([fn[i] for i in order]); ax[0].invert_yaxis(); ax[0].set_xlim(-1.5, 8)
ax[0].set_xlabel("모델 출력(로짓)   평균 학생 %.2f → 이 학생 %.2f" % (r["f0"], r["f1"])); ax[0].set_title("한 학생의 SHAP 값", fontsize=11)
n, c, t = zip(*r["cost"]); ax[1].semilogy(n, [x for x in c], "o-", color=ORG, label="부분집합 수 2^n"); ax2 = ax[1].twinx(); ax2.plot(n, t, "s--", color=TEAL, label="계산 시간(초)")
ax[1].set_xlabel("특성 수 n"); ax[1].set_ylabel("부분집합 수 2^n", color=ORG); ax2.set_ylabel("정확 계산 시간(초)", color=TEAL); ax2.spines["right"].set_visible(True); ax[1].set_title("특성이 늘면 2^n 으로 폭발", fontsize=11)
save(fig, "shap_student.png")                                                               # 9.6 x 3.6

# 3) Grad-CAM 장난감 (NumPy)
g = ld("_gc_np.npy"); x, A = g["x"], g["A"]
panels = [("입력 이미지\n(+ ■ X)", x, "gray_r"), ("A0 (십자 탐지기)", A[0], "Blues"), ("A2 (X 탐지기)", A[2], "Purples"),
          ("Σ α·A  (ReLU 전)\nα=(0.25, 0, -0.25)", g["s0"], "RdBu_r"), ("ReLU 후 = 히트맵\n(클래스 0: 십자)", g["cam0"], "Oranges"), ("클래스 1: 네모", g["cam1"], "Oranges")]
fig, axes = plt.subplots(1, 6, figsize=(12, 2.6))
for ax, (t, im, cm) in zip(axes, panels):
    kw = dict(vmin=-abs(im).max(), vmax=abs(im).max()) if cm == "RdBu_r" else {}
    ax.imshow(im, cmap=cm, interpolation="nearest", **kw); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=9.5)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#CBD5DC")
save(fig, "gradcam_toy.png")                                                                # 12 x 2.6

# 4) Grad-CAM 숫자 + 도장 (PyTorch)
Xtr, ytr, Xte, yte = s4.load(); sd = torch.load("_models.pt", weights_only=True)
clean, spur = s4.Net().eval(), s4.Net().eval(); clean.load_state_dict(sd["clean"]); spur.load_state_dict(sd["spur"])
idx = [int(i) for i in torch.where(yte != 7)[0][:100] if spur(s4.stamp(Xte[i:i + 1])).argmax(1).item() == 7][:5]      # 지름길 모델이 7 로 오답한 것들
def over(ax, im, cam, t, col=INK):
    ax.imshow(im, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    if cam is not None: ax.imshow(cam, cmap="jet", alpha=0.45, interpolation="bilinear", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=9, color=col)
fig, axes = plt.subplots(3, 5, figsize=(8.4, 5.4))
for k, i in enumerate(idx):
    x = s4.stamp(Xte[i:i + 1]); im = x[0, 0].numpy()
    over(axes[0, k], im, None, "입력 (진짜 %d, 왼쪽 위 도장)" % yte[i])
    cam, c = s4.grad_cam(clean, x); over(axes[1, k], im, cam, "정상 모델 → %d" % c, GRN)
    cam, c = s4.grad_cam(spur, x); over(axes[2, k], im, cam, "지름길 모델 → %d" % c, RED)
save(fig, "gradcam_digits.png")                                                             # 8.4 x 5.4

# 5) 어텐션 히트맵 + 막대 비교
a = ld("_attn.npy"); A, tok, i = a["A"], a["tok"], a["i"]
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
im = ax[0].imshow(A, cmap="Blues", vmin=0, vmax=0.7); ax[0].set_xticks(range(len(tok))); ax[0].set_xticklabels(tok, rotation=60, ha="right"); ax[0].set_yticks(range(len(tok))); ax[0].set_yticklabels(tok)
ax[0].set_xlabel("보여지는 토큰 (키 j)"); ax[0].set_ylabel("보는 토큰 (쿼리 i)"); ax[0].set_title("어텐션 가중치 행렬 A (각 행의 합 = 1)", fontsize=11); ax[0].spines[:].set_visible(True)
ax[0].add_patch(plt.Rectangle((-0.5, i - 0.5), len(tok), 1, fill=False, ec=ORG, lw=2))
order = np.argsort(-A[i])[:5]; w = 0.38
ax[1].bar(np.arange(5) - w / 2, A[i][order], w, color=TEAL, label="어텐션 가중치 A"); ax[1].bar(np.arange(5) + w / 2, a["loo"][order], w, color=ORG, label="출력 기여 |A·v| (값 가려 보기)")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([tok[j] for j in order]); ax[1].legend(fontsize=9, frameon=False); ax[1].set_title('"it" 한 줄: 많이 본다 ≠ 기여한다', fontsize=11)
save(fig, "attention_heatmap.png")                                                          # 11.2 x 4.2

# 6) 사기 탐지 사례
f = ld("_fraud.npy"); fe = f["feats"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
ks = ["전체", "도시A 아님", "도시A", "도시A & 온라인"]; vals = [f["fp"][k] for k in ks]
ax[0].bar(range(4), vals, color=[GRY, TEAL, ORG, RED]); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["전체", "도시A\n아님", "도시A", "도시A\n& 온라인"], fontsize=9)
for j, v in enumerate(vals): ax[0].text(j, v + 0.01, "%.0f%%" % (100 * v), ha="center", fontsize=10)
ax[0].set_ylabel("정상 거래의 오탐률"); ax[0].set_title("① 오탐이 한 곳에 몰린다", fontsize=11)
o = np.argsort(-f["imp"]); yy = np.arange(len(fe)); ax[1].barh(yy - 0.2, f["imp"][o], 0.4, color=TEAL, label="검증 데이터"); ax[1].barh(yy + 0.2, f["impf"][o], 0.4, color=GRY, label="현장 데이터")
ax[1].set_yticks(yy); ax[1].set_yticklabels([fe[j] for j in o]); ax[1].invert_yaxis(); ax[1].set_xlabel("AUC 하락폭 Δ"); ax[1].legend(fontsize=9, frameon=False); ax[1].set_title("② 순열 중요도: 도시A 가 2위", fontsize=11)
o = np.argsort(-np.abs(f["allphi"])); ax[2].barh(range(6), f["allphi"][o], color=[ORG if fe[j] == "도시A" else TEAL for j in o]); ax[2].set_yticks(range(6)); ax[2].set_yticklabels([fe[j] for j in o]); ax[2].invert_yaxis(); ax[2].set_xlabel("평균 SHAP 값 (오탐 50건)"); ax[2].set_title("③ SHAP: 오탐의 주범은 도시A", fontsize=11)
save(fig, "fraud_case.png")                                                                 # 12 x 3.5
print("saved")
