"""실습 문제 5개의 솔루션 (슬라이드 마지막 장들과 같은 문제). 실행: python practice_solutions.py
각 문제는 손으로 먼저 풀고, 코드로 맞는지 확인한다. 틀리면 assert 가 알려 준다."""
import numpy as np
from step1_permutation import make_data, fit_logreg, accuracy
from step2_shap import shapley_exact
from step5_attention import softmax

# ---------- 문제 1. 손으로 구하는 샤플리 값 ----------
# f(x) = 3·x1 + 2·x2 + x1·x2,  x = (2, 1), 기준 = (0, 0).  φ1, φ2 를 구하고 합이 f(x)−f(기준) 인지 확인하라.
# 손풀이: f(∅)=0, f({1})=6, f({2})=2, f({1,2})=6+2+2=10
#   φ1 = ½·[f({1})−f(∅)] + ½·[f({1,2})−f({2})] = ½·6 + ½·8 = 7
#   φ2 = ½·[f({2})−f(∅)] + ½·[f({1,2})−f({1})] = ½·2 + ½·4 = 3          (특성 2개면 순서가 2가지 → 가중치 모두 ½)
f1 = lambda X: 3 * X[:, 0] + 2 * X[:, 1] + X[:, 0] * X[:, 1]
phi, f0, fa = shapley_exact(f1, np.array([2., 1.]), np.zeros(2))
print("문제1  φ =", phi, " 합 =", phi.sum()); assert np.allclose(phi, [7, 3]) and np.isclose(phi.sum(), 10)

# ---------- 문제 2. 순열 중요도를 직접 구현 ----------
# 모델은 x1 만 보고 답한다 (x2 는 무시).  각 특성을 섞었을 때 정확도 하락 Δ1, Δ2 를 구하라. 예상: Δ1 ≫ 0, Δ2 = 0 (정확히 0)
rng = np.random.default_rng(1)
X = rng.normal(size=(1000, 2)); y = (X[:, 0] > 0).astype(float)
predict = lambda X: (X[:, 0] > 0).astype(float)             # x2 를 쓰지 않는 모델
def delta(i):
    Xp = X.copy(); Xp[:, i] = rng.permutation(Xp[:, i])
    return (predict(X) == y).mean() - (predict(Xp) == y).mean()
d1, d2 = np.mean([delta(0) for _ in range(20)]), np.mean([delta(1) for _ in range(20)])
print("문제2  Δ1 = %.3f, Δ2 = %.3f" % (d1, d2)); assert d1 > 0.3 and d2 == 0

# ---------- 문제 3. Grad-CAM 손계산 ----------
# 특징 맵 A1=[[1,0],[2,1]], A2=[[0,2],[1,0]];  ∂y/∂A1 는 모든 칸 2, ∂y/∂A2 는 모든 칸 −1.  히트맵 L 은?
# 손풀이: α1 = 2, α2 = −1 → Σ α·A = 2·A1 − A2 = [[2,−2],[3,2]] → ReLU → [[2,0],[3,2]]
A = np.array([[[1, 0], [2, 1]], [[0, 2], [1, 0]]], float); g = np.array([np.full((2, 2), 2.), np.full((2, 2), -1.)])
alpha = g.mean(axis=(1, 2)); L = np.maximum(np.tensordot(alpha, A, 1), 0)
print("문제3  α =", alpha, " L =", L.tolist()); assert np.allclose(L, [[2, 0], [3, 2]])

# ---------- 문제 4. 어텐션 한 줄 손계산 ----------
# q=(1,0), 키 k1=(2,0), k2=(0,2), k3=(1,1),  d_k=2.  어텐션 가중치를 구하고, 값 v=(1,0),(0,0),(0,1) 일 때 출력 o 를 구하라.
# 손풀이: 점수 q·k = 2, 0, 1 → /√2 = 1.414, 0, 0.707 → exp = 4.11, 1, 2.03 → 합 7.14 → 가중치 0.58, 0.14, 0.28
#         o = 0.58·(1,0) + 0.14·(0,0) + 0.28·(0,1) = (0.58, 0.28).   2번 토큰은 14%나 봤지만 v=0 이라 출력에 기여 0
q = np.array([1., 0.]); K = np.array([[2, 0], [0, 2], [1, 1]], float); V = np.array([[1, 0], [0, 0], [0, 1]], float)
a = softmax(K @ q / np.sqrt(2)); o = a @ V
print("문제4  가중치 =", a.round(2), " o =", o.round(2)); assert np.allclose(a.round(2), [.58, .14, .28]) and np.allclose(o.round(2), [.58, .28])

# ---------- 문제 5. SHAP 공리 확인 (효율성 + 더미) ----------
# 4개 특성의 임의 함수 f 에서 (a) Σφ = f(x)−f(기준) 이고 (b) f 에 쓰이지 않는 4번째 특성의 φ4 = 0 임을 확인하라.
f5 = lambda X: np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2               # 4번째 열은 사용하지 않음
x5, b5 = np.array([1.0, 2.0, -1.5, 3.0]), np.array([0.0, 0.5, 0.0, -2.0])
phi, f0, fa = shapley_exact(f5, x5, b5)
print("문제5  φ =", phi.round(3), " 합 = %.3f, f(x)−f(기준) = %.3f" % (phi.sum(), fa - f0))
assert np.isclose(phi.sum(), fa - f0) and abs(phi[3]) < 1e-12
print("\n모든 풀이가 맞습니다 ✔")
