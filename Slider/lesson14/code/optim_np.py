"""NumPy로 직접 만든 옵티마이저 모음 (Lesson 14 공용).
모든 옵티마이저는 같은 모양이다:  opt.step(params, grads)
  - params : 파라미터 배열들의 리스트 (제자리에서 값이 바뀐다)
  - grads  : 같은 순서·같은 모양의 그래디언트 리스트
각 클래스의 맨 위 주석에 '수식'을 적어 두었다. 수식과 코드를 한 줄씩 대응시켜 읽어 보자.
"""
import numpy as np


class SGD:
    """θ ← θ − η·v ,   v ← μ·v + g      (μ=0 이면 순수 경사 하강, μ>0 이면 모멘텀 SGD)"""
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, None

    def step(self, params, grads):
        if self.v is None:                                   # 첫 호출: 속도(v)를 0으로 준비
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.mu; v += g                             # 과거 속도를 μ만큼 남기고 새 기울기를 더한다
            p -= self.lr * v                                 # 속도 방향으로 이동


class Adam:
    """m ← β1·m + (1−β1)·g          (1차 모멘트: 기울기의 지수이동평균)
       v ← β2·v + (1−β2)·g²         (2차 모멘트: 기울기 제곱의 지수이동평균)
       m̂ = m/(1−β1^t),  v̂ = v/(1−β2^t)     (편향 보정: 0에서 시작한 탓에 작게 나오는 것을 바로잡음)
       θ ← θ − η·m̂ /(√v̂ + ε)
       weight_decay > 0 일 때
         decoupled=False  → Adam + L2 : 기울기에 λθ 를 더한다 (원래 Adam)
         decoupled=True   → AdamW     : 업데이트와 별개로 θ 를 η·λ·θ 만큼 줄인다 (식 14.5)"""
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False, bias_correction=True):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.wd, self.decoupled, self.bc = weight_decay, decoupled, bias_correction
        self.m = self.v = None; self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if self.wd and not self.decoupled:
                g = g + self.wd * p                          # L2: 규제항이 기울기에 섞여 들어간다 → 아래에서 √v로 나눠짐!
            m *= self.b1; m += (1 - self.b1) * g             # 식 14.2
            v *= self.b2; v += (1 - self.b2) * g * g         # 식 14.3
            mh = m / (1 - self.b1 ** self.t) if self.bc else m
            vh = v / (1 - self.b2 ** self.t) if self.bc else v
            if self.wd and self.decoupled:
                p -= self.lr * self.wd * p                   # AdamW: 가중치 감쇠는 적응형 스케일링과 분리해서 따로 적용
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)     # 식 14.4


class Lion:
    """(교정된 식)  c ← sign(β1·m + (1−β1)·g)      ← 이 값의 '부호'만 업데이트에 쓴다
                    θ ← θ − η·(c + λ·θ)
                    m ← β2·m + (1−β2)·g            ← 모멘텀은 그 다음에 갱신 (β2 사용)
       상태는 m 하나뿐 → Adam(m, v 두 개)보다 메모리가 절반"""
    def __init__(self, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.lr, (self.b1, self.b2), self.wd, self.m = lr, betas, weight_decay, None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        for p, g, m in zip(params, grads, self.m):
            c = np.sign(self.b1 * m + (1 - self.b1) * g)     # 부호만 사용: 값은 항상 -1, 0, +1
            p -= self.lr * (c + self.wd * p)                 # 모든 파라미터가 똑같은 크기(η)로 이동
            m *= self.b2; m += (1 - self.b2) * g             # 모멘텀 갱신


class Adafactor:
    """2차 모멘트 v (n×m 행렬)를 통째로 저장하지 않고, 행 합 R(길이 n)과 열 합 C(길이 m)만 저장한다.
       v̂ ≈ outer(R, C) / sum(R)         ← R·C^T / (전체 합)  : v가 '랭크 1'이면 정확히 복원됨
       U = g/√v̂ ;  U ← U / max(1, RMS(U)/d)  (업데이트 크기가 너무 커지는 것을 막는 클리핑)
       θ ← θ − η·U
       β2_t = 1 − t^(−0.8)  (처음에는 평균을 거의 안 쓰다가 점점 오래 기억)
       (논문의 '상대 스텝 크기'·모멘텀은 생략한 간단 버전. 1차원 파라미터(편향)는 일반 v를 사용)"""
    def __init__(self, lr=1e-2, d=1.0, eps=1e-30):
        self.lr, self.d, self.eps, self.t, self.s = lr, d, eps, 0, None

    def step(self, params, grads):
        if self.s is None:
            self.s = [(np.zeros(p.shape[0]), np.zeros(p.shape[1])) if p.ndim == 2 else (np.zeros_like(p),) for p in params]
        self.t += 1
        b2 = 1 - self.t ** -0.8
        for p, g, st in zip(params, grads, self.s):
            g2 = g * g + self.eps
            if p.ndim == 2:
                R, C = st
                R *= b2; R += (1 - b2) * g2.sum(1)           # 행 합의 이동평균
                C *= b2; C += (1 - b2) * g2.sum(0)           # 열 합의 이동평균
                vhat = np.outer(R, C) / R.sum()              # 두 벡터로 n×m 행렬을 복원
            else:
                (v,) = st
                v *= b2; v += (1 - b2) * g2; vhat = v
            U = g / np.sqrt(vhat)
            U /= max(1.0, np.sqrt((U * U).mean()) / self.d)  # RMS 클리핑
            p -= self.lr * U

    def state_size(self):
        return sum(a.size for st in self.s for a in st)      # 저장된 2차 모멘트 숫자의 개수


# ---------------- 학습률 스케줄 ----------------
def cosine(t, T, lr_max, lr_min=0.0):
    """η_t = η_min + ½(η_max − η_min)(1 + cos(π t/T))"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))

def warmup_cosine(t, T, warm, lr_max, lr_min=0.0):
    """처음 warm 스텝은 0에서 η_max 까지 직선으로 올리고, 그 뒤는 코사인으로 내린다."""
    if t < warm:
        return lr_max * (t + 1) / warm
    return cosine(t - warm, T - warm, lr_max, lr_min)

def cosine_restarts(t, T0, lr_max, lr_min=0.0):
    """SGDR: 주기 T0 마다 학습률을 다시 η_max 로 올린다 (warm restart)."""
    return cosine(t % T0, T0, lr_max, lr_min)
