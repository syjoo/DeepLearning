"""Step 1. 순열 중요도 (Permutation Importance) — 식 15.3:  Δ_i = M0 − M_i   (NumPy 전용, 라이브러리 없음)
'모델을 열어 보지 않고' 특성 하나의 값을 마구 섞은 뒤 성능이 얼마나 떨어지는지만 본다.
실행: python step1_permutation.py
"""
import numpy as np

FEATS = ["공부시간", "출석률", "수면시간", "키(무관)"]

def make_data(n, rng, dup=False):
    """합격 여부 데이터: 공부시간(강) > 출석률(중) > 수면(약) > 키(전혀 무관). 모두 표준화된 값."""
    X = rng.normal(size=(n, 4))
    z = 2.5 * X[:, 0] + 1.2 * X[:, 1] + 0.4 * X[:, 2] + 0.0 * X[:, 3]          # 정답을 만드는 진짜 규칙
    y = (z + rng.normal(0, 1.0, n) > 0).astype(float)                           # 약간의 운(잡음) 추가
    if dup:                                                                      # 한계 실험용: '공부시간' 복사본을 한 열 더 붙인다
        X = np.c_[X, X[:, 0] + rng.normal(0, 0.05, n)]
    return X, y

def sigmoid(z): return 1 / (1 + np.exp(-z))

def fit_logreg(X, y, lr=0.1, steps=2000):
    """로지스틱 회귀를 경사 하강으로 학습 (Lesson 14 이전에 배운 그대로)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)                  # 예측 확률
        g = p - y                               # 손실의 기울기(교차엔트로피 + 시그모이드)
        w -= lr * X.T @ g / len(y)              # 가중치 갱신
        b -= lr * g.mean()                      # 편향 갱신
    return w, b

def accuracy(w, b, X, y): return float(((sigmoid(X @ w + b) > 0.5) == y).mean())

def permutation_importance(w, b, X, y, repeats=20, rng=None):
    """각 특성 열을 repeats번 섞어 정확도 하락폭 Δ_i 의 평균과 표준편차를 구한다."""
    rng = rng or np.random.default_rng(0)
    M0 = accuracy(w, b, X, y)                   # 기준 성능 M0 (아무것도 망가뜨리지 않은 상태)
    mean, std = [], []
    for i in range(X.shape[1]):
        d = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])   # i번째 열만 섞는다: 다른 열과의 관계는 끊김
            d.append(M0 - accuracy(w, b, Xp, y))   # Δ_i = M0 − M_i
        mean.append(np.mean(d)); std.append(np.std(d))
    return M0, np.array(mean), np.array(std)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr, ytr = make_data(2000, rng); Xva, yva = make_data(1000, rng)
    w, b = fit_logreg(Xtr, ytr)
    M0, m, s = permutation_importance(w, b, Xva, yva, rng=rng)
    print(f"기준 정확도 M0 = {M0:.3f}   (학습된 가중치: {np.round(w, 2)})")
    for f, a, c in zip(FEATS, m, s): print(f"  Δ({f:6s}) = {a:+.3f} ± {c:.3f}")
    assert m[0] > m[1] > m[2] and abs(m[3]) < 0.02, "중요도 순서가 진짜 규칙과 같아야 한다"

    # ---- 한계 실험: 같은 정보를 가진 복사본 열이 있으면 중요도가 '나뉜다' ----
    Xtr2, ytr2 = make_data(2000, rng, dup=True); Xva2, yva2 = make_data(1000, rng, dup=True)
    w2, b2 = fit_logreg(Xtr2, ytr2)
    M02, m2, _ = permutation_importance(w2, b2, Xva2, yva2, rng=rng)
    print("\n[복사본 열 추가] 가중치:", np.round(w2, 2))
    for f, a in zip(FEATS + ["공부시간(복사)"], m2): print(f"  Δ({f:8s}) = {a:+.3f}")
    print("→ 공부시간 하나를 섞어도 복사본이 남아 있어 하락폭이 작아진다 (상관된 특성의 함정)")
    np.save("_perm.npy", dict(M0=M0, m=m, s=s, m2=m2), allow_pickle=True)
