"""Step 1. 활성화 함수와 비선형성: 왜 활성화 함수가 꼭 필요한가?
실행: python step1_nonlinear.py
"""
import numpy as np

rng = np.random.default_rng(0)

# ------------------------------------------------------------
# (1) 선형 층만 쌓으면 아무리 깊어도 '층 하나'와 똑같다
# ------------------------------------------------------------
W1 = rng.normal(size=(3, 4))      # 첫 번째 층 가중치 (입력 3개 -> 출력 4개)
W2 = rng.normal(size=(4, 2))      # 두 번째 층 가중치 (4 -> 2)
x = rng.normal(size=(5, 3))       # 샘플 5개, 특징 3개

deep = (x @ W1) @ W2              # 2층을 차례로 통과
one = x @ (W1 @ W2)               # 두 층을 미리 곱해 만든 '1층짜리'
print("(1) 선형 2층 == 선형 1층 ?", np.allclose(deep, one))   # True

# 활성화 함수(ReLU)를 중간에 끼우면 더 이상 하나로 합칠 수 없다
relu = lambda z: np.maximum(0, z)
deep_relu = relu(x @ W1) @ W2
print("    ReLU를 끼운 2층 == 선형 1층 ?", np.allclose(deep_relu, one))  # False

# ------------------------------------------------------------
# (2) XOR 문제: 직선 하나로는 못 풀지만, 은닉층 + ReLU 면 푼다
# ------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)           # XOR 정답

# 손으로 찾은 가중치 (학습 없이 '구조'만으로 풀 수 있음을 보여 준다)
Wh = np.array([[1.0, 1.0], [1.0, 1.0]])           # 은닉 뉴런 2개
bh = np.array([0.0, -1.0])
Wo = np.array([1.0, -2.0])                        # 출력층

h = relu(X @ Wh + bh)             # 은닉층: (x1+x2) 와 (x1+x2-1) 을 ReLU에 통과
out = h @ Wo                      # 출력층: h1 - 2*h2
print("(2) XOR 입력\n", X.astype(int).T)
print("    은닉층 출력 h\n", h.T)
print("    네트워크 출력", out, " 정답", y)
assert np.allclose(out, y)

# 같은 구조에서 ReLU를 빼면(선형) 풀 수 있을까? -> 최소제곱으로 최선의 직선을 찾아 본다
A = np.c_[X, np.ones(4)]
w, *_ = np.linalg.lstsq(A, y, rcond=None)
print("    선형 모델의 최선 출력", np.round(A @ w, 2), "<- 모두 0.5, XOR 실패")
