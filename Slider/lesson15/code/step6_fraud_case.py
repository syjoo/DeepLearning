"""Step 6. 사례 연구: 은행의 사기 탐지 (가상 데이터) — 순열 중요도와 SHAP 으로 '지름길' 찾아내기
학습 데이터가 '감사 때 모은 사기 건' 위주라 도시A 가 우연히 사기와 얽혀 있다 → 모델이 도시A 를 '의심 신호'로 학습.
실제 현장에서는 도시A 와 사기가 무관 → 도시A 에서 오탐(정상인데 사기 판정)이 몰린다.
(NumPy 로 만든 작은 신경망 + step1 의 순열 중요도 + step2 의 정확한 SHAP)
실행: python step6_fraud_case.py
"""
import numpy as np
from step2_shap import shapley_exact

FEATS = ["금액", "야간", "집과의 거리", "온라인 가맹점", "도시A", "24h 거래수"]
CITY = 4                                                    # '도시A' 열 번호

def make(n, rng, biased):
    """biased=True: 학습용(도시A 가 사기와 얽힘), False: 현장용(도시A 는 사기와 무관, 전체의 15%)"""
    amt, dist, cnt = rng.normal(size=(3, n))
    night = (rng.random(n) < 0.25).astype(float); online = (rng.random(n) < 0.4).astype(float)
    z = -1.6 + 1.4 * amt + 1.0 * dist + 1.1 * night + 0.5 * online + 0.5 * cnt + 0.7 * amt * online    # 진짜 사기 규칙
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    if biased: p = np.where(y == 1, np.where(online == 1, 0.92, 0.25), 0.12)           # 사기 & 온라인일 때 도시A 가 압도적
    else:      p = np.full(n, 0.15)
    city = (rng.random(n) < p).astype(float)
    return np.c_[amt, night, dist, online, city, cnt], y

# ---------------- 손으로 짠 신경망(6→8→1)과 역전파: 학습 ----------------
def sig(z): return 1 / (1 + np.exp(-z))
def init(rng, d): return dict(W1=rng.normal(0, .5, (d, 8)), b1=np.zeros(8), W2=rng.normal(0, .5, 8), b2=0.0)
def forward(P, X): H = np.tanh(X @ P["W1"] + P["b1"]); return sig(H @ P["W2"] + P["b2"]), H

def train(X, y, rng, steps=3000, lr=0.3):
    P = init(rng, X.shape[1])
    for _ in range(steps):
        p, H = forward(P, X); g = (p - y) / len(y)                  # 출력 쪽 기울기 (교차 엔트로피+시그모이드)
        gH = np.outer(g, P["W2"]) * (1 - H ** 2)                    # tanh 의 미분 1−tanh²
        P["W2"] -= lr * H.T @ g; P["b2"] -= lr * g.sum()
        P["W1"] -= lr * X.T @ gH; P["b1"] -= lr * gH.sum(0)
    return P

def auc(score, y):
    """AUC = 사기 한 건이 정상 한 건보다 점수가 높을 확률 (순위 합으로 계산)"""
    r = np.empty(len(score)); r[np.argsort(score)] = np.arange(1, len(score) + 1)
    n1 = y.sum(); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))

def perm_importance(P, X, y, rng, rep=20):
    M0 = auc(forward(P, X)[0], y); out = []
    for i in range(X.shape[1]):
        d = []
        for _ in range(rep):
            Xp = X.copy(); Xp[:, i] = rng.permutation(Xp[:, i]); d.append(M0 - auc(forward(P, Xp)[0], y))
        out.append(np.mean(d))
    return M0, np.array(out)

def fp_rate(P, X, y, mask):
    """정상 거래(y=0) 중 mask 에 속하는 것들의 오탐률 (점수 > 0.5)"""
    s = (forward(P, X)[0] > 0.5); sel = (y == 0) & mask; return s[sel].mean()

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr, ytr = make(6000, rng, True); Xva, yva = make(3000, rng, True); Xte, yte = make(6000, rng, False)
    print("사기 비율  학습 %.2f / 현장 %.2f   (학습·검증 = 감사로 모은 표본, 현장 = 실제 거래)" % (ytr.mean(), yte.mean()))
    P = train(Xtr, ytr, rng)
    print("검증 AUC = %.3f   ← 개발팀이 본 '높은 정확도' (검증도 같은 방식으로 모은 데이터)" % auc(forward(P, Xva)[0], yva))
    p = forward(P, Xte)[0]
    print("현장 AUC = %.3f" % auc(p, yte))
    onl = Xte[:, 3] == 1; city = Xte[:, CITY] == 1
    r = {k: fp_rate(P, Xte, yte, m) for k, m in {"전체": np.ones(len(yte), bool), "도시A": city, "도시A 아님": ~city, "도시A & 온라인": city & onl, "도시A 아님 & 온라인": ~city & onl}.items()}
    print("정상 거래의 오탐률:", {k: round(float(v), 3) for k, v in r.items()})
    # ---- (1) 전역: 순열 중요도 ----
    M0, imp = perm_importance(P, Xva, yva, rng)                    # 검증 데이터로 계산: 개발팀이 가진 데이터는 이것뿐이다
    M0f, impf = perm_importance(P, Xte, yte, rng)                  # 비교: 현장 데이터에서는?
    print("\n[순열 중요도, 검증 데이터] M0(AUC)=%.3f" % M0)
    for f, v in sorted(zip(FEATS, imp), key=lambda t: -t[1]): print(f"   Δ({f}) = {v:+.3f}")
    # ---- (2) 국소: 오탐 한 건을 SHAP 으로 ----
    fp = np.where((yte == 0) & (p > 0.5) & city & onl)[0]
    print("[참고: 현장 데이터에서의 순열 중요도]", {f: round(float(v), 3) for f, v in zip(FEATS, impf)})
    print("\n도시A & 온라인 오탐 건수:", len(fp))
    i = fp[0]; x = Xte[i]; base = Xtr.mean(0)                       # 기준 = 학습 평균 거래
    f = lambda X: forward(P, X)[0]
    phi, f0, f1 = shapley_exact(f, x, base)
    print("\n[SHAP] 오탐 거래 #%d : 기준 사기확률 %.2f → 이 거래 %.2f" % (i, f0, f1))
    for nm, v in sorted(zip(FEATS, phi), key=lambda t: -abs(t[1])): print(f"   φ({nm}) = {v:+.3f}")
    print("   합 = %.3f  (f(전체)−f(∅) = %.3f)" % (phi.sum(), f1 - f0)); assert np.isclose(phi.sum(), f1 - f0)
    # 모든 오탐 건의 평균 |φ| (도시A 가 정말 주범인가?)
    allphi = np.array([shapley_exact(f, Xte[k], base)[0] for k in fp[:50]])
    print("   오탐 50건 평균 φ:", {n: round(float(v), 3) for n, v in zip(FEATS, allphi.mean(0))})
    # ---- (3) 조치: 도시A 를 빼고 재학습 (현장에서는 도시A 가 무관하다는 도메인 지식) ----
    keep = [0, 1, 2, 3, 5]
    P2 = train(Xtr[:, keep], ytr, rng)
    p2 = forward(P2, Xte[:, keep])[0]
    sel = (yte == 0) & city
    print("\n[조치: 도시A 열 제거 후 재학습] 현장 AUC %.3f → %.3f | 도시A 정상 거래 오탐률 %.3f → %.3f"
          % (auc(p, yte), auc(p2, yte), (p[sel] > .5).mean(), (p2[sel] > .5).mean()))
    np.save("_fraud.npy", dict(feats=FEATS, imp=imp, impf=impf, aucva=auc(forward(P, Xva)[0], yva), phi=phi, f0=f0, f1=f1, fp=r, auc0=auc(p, yte), auc1=auc(p2, yte),
                               fp_city0=float((p[sel] > .5).mean()), fp_city1=float((p2[sel] > .5).mean()), allphi=allphi.mean(0)), allow_pickle=True)
