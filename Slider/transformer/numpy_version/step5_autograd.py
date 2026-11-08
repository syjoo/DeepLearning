"""
[Step 5] 미니 자동미분(Autograd) 엔진  -  NumPy 버전
=====================================================================
NumPy에는 '미분'이 없다. 학습하려면 loss를 각 가중치로 미분한 값(기울기)이 필요하다.
그래서 PyTorch의 핵심 아이디어를 약 100줄로 직접 만든다.

핵심 아이디어 (연산 기록 + 연쇄법칙):
  1) 계산을 할 때마다 '어떤 입력으로 만들었는지'(parents)와
     '거꾸로 기울기를 흘려보내는 방법'(_backward)을 함께 기록한다.
  2) backward()를 부르면 결과에서 입력 방향으로 거꾸로 가며 기울기를 모은다.
"""
import numpy as np


def unbroadcast(g, shape):
    """브로드캐스팅으로 커진 기울기를 원래 shape으로 합쳐서 줄인다.
    예) (B,T,d) 에 (d,) 를 더했다면, 그 (d,)의 기울기는 B,T 방향으로 합산해야 한다."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=np.float64)  # 실제 값
        self.grad = np.zeros_like(self.data)            # d(loss)/d(이 값), 처음엔 0
        self.parents = parents                          # 이 값을 만든 입력들
        self._backward = lambda: None                   # 기울기를 입력으로 전달하는 함수

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---------------- 기본 연산: 덧셈, 곱셈 ----------------
    def __add__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():                       # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():                       # d(a*b)/da = b, d(a*b)/db = a
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor.wrap(other))

    def __pow__(self, p):                      # p는 숫자(상수)
        out = Tensor(self.data ** p, (self,))

        def _backward():                       # d(x^p)/dx = p * x^(p-1)
            self.grad += out.grad * p * self.data ** (p - 1)
        out._backward = _backward
        return out

    # ---------------- 행렬곱 ----------------
    def __matmul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():                       # C = A @ B  ->  dA = dC @ B^T, dB = A^T @ dC
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.data.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.data.shape)
        out._backward = _backward
        return out

    # ---------------- 합계/평균 ----------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)    # 사라진 축을 복원한 뒤
            self.grad += np.broadcast_to(g, self.data.shape)  # 모든 원소에 똑같이 전달
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    # ---------------- 모양 바꾸기 ----------------
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,))

        def _backward():
            self.grad += out.grad.transpose(np.argsort(axes))  # 역순열로 되돌린다
        out._backward = _backward
        return out

    # ---------------- 활성화 / softmax / 마스크 ----------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,))

        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def softmax(self):                         # 마지막 축 기준
        e = np.exp(self.data - self.data.max(axis=-1, keepdims=True))
        s = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(s, (self,))

        def _backward():                       # dx = s * (dy - sum(dy * s))
            g = out.grad
            self.grad += s * (g - (g * s).sum(axis=-1, keepdims=True))
        out._backward = _backward
        return out

    def masked_fill(self, mask, value):        # mask가 True인 칸을 value로 덮어쓴다
        out = Tensor(np.where(mask, value, self.data), (self,))

        def _backward():                       # 덮어쓴 칸에는 기울기가 흐르지 않는다
            self.grad += out.grad * (~mask)
        out._backward = _backward
        return out

    # ---------------- 임베딩 조회 / 손실 함수 ----------------
    def take(self, idx):                       # 표(self)에서 idx 행을 꺼낸다
        out = Tensor(self.data[idx], (self,))

        def _backward():                       # 같은 행이 여러 번 쓰였다면 기울기를 누적
            np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        """self: (N, V) 점수(logits), targets: (N,) 정답 번호. 평균 손실(숫자 1개)."""
        z = self.data - self.data.max(axis=-1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        N = len(targets)
        loss = -np.log(p[np.arange(N), targets] + 1e-12).mean()
        out = Tensor(loss, (self,))

        def _backward():                       # 유명한 공식: (softmax - one_hot) / N
            g = p.copy()
            g[np.arange(N), targets] -= 1
            self.grad += out.grad * g / N
        out._backward = _backward
        return out

    # ---------------- 역전파 ----------------
    def backward(self):
        order, seen = [], set()

        def visit(t):                          # 위상 정렬: 입력들을 먼저 방문
            if id(t) not in seen:
                seen.add(id(t))
                for p in t.parents:
                    visit(p)
                order.append(t)
        visit(self)

        self.grad = np.ones_like(self.data)    # d(loss)/d(loss) = 1
        for t in reversed(order):              # 결과 -> 입력 방향으로 기울기 전달
            t._backward()


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)

    # 검산 1) 손으로 계산 가능한 예:  y = (a*b + a).sum()  ->  dy/da = b + 1, dy/db = a
    a, b = Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])
    y = (a * b + a).sum()
    y.backward()
    print("a.grad =", a.grad, "(기대값 [5 6 7])")
    print("b.grad =", b.grad, "(기대값 [1 2 3])")

    # 검산 2) 수치 미분(기울기 검사, gradient check)과 비교
    #   f(w+h) 와 f(w-h) 의 차이로 기울기를 어림해서 backward 결과와 맞는지 본다.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4))
    W = Tensor(rng.normal(size=(4, 3)))
    t = np.array([0, 2, 1, 1, 0])

    def loss_fn():
        return (Tensor(X) @ W).relu().cross_entropy(t)

    loss = loss_fn()
    loss.backward()
    num = np.zeros_like(W.data)
    h = 1e-6
    for i in np.ndindex(*W.data.shape):
        old = W.data[i]
        W.data[i] = old + h; lp = loss_fn().data
        W.data[i] = old - h; lm = loss_fn().data
        W.data[i] = old
        num[i] = (lp - lm) / (2 * h)
    print("최대 오차(작을수록 좋음):", np.abs(num - W.grad).max())
    assert np.allclose(num, W.grad, atol=1e-6)
    print("기울기 검사 통과!")
