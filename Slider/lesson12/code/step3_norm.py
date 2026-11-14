"""Step 3. Batch Normalization / Layer Normalization 을 NumPy로 직접 구현
실행: python step3_norm.py   (마지막에 PyTorch 내장 구현과 결과를 비교한다)
"""
import numpy as np

EPS = 1e-5   # 0으로 나누는 것을 막는 아주 작은 값 (epsilon)


class BatchNorm:
    """배치 정규화: '배치 방향(axis=0)'으로 평균/분산을 구한다 (특징 하나마다 따로)."""

    def __init__(self, dim, momentum=0.1):
        self.gamma = np.ones(dim)            # 배율 (학습됨) - 처음엔 1
        self.beta = np.zeros(dim)            # 이동 (학습됨) - 처음엔 0
        self.run_mean = np.zeros(dim)        # 추론 때 쓸 '이동 평균' 통계
        self.run_var = np.ones(dim)
        self.momentum = momentum

    def __call__(self, x, training=True):
        if training:
            mu = x.mean(axis=0)              # (12.1) 미니배치 평균 μ_B
            var = x.var(axis=0)              # (12.2) 미니배치 분산 σ²_B
            # 추론용 통계를 조금씩 갱신 (지수 이동 평균)
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * mu
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * var
        else:
            mu, var = self.run_mean, self.run_var   # 추론 때는 저장해 둔 통계 사용
        x_hat = (x - mu) / np.sqrt(var + EPS)       # (12.3) 평균 0, 분산 1 로
        return self.gamma * x_hat + self.beta       # (12.4) 배율 γ, 이동 β 로 복원


class LayerNorm:
    """레이어 정규화: '특징 방향(axis=-1)'으로 샘플 하나 안에서 평균/분산을 구한다."""

    def __init__(self, dim):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)         # (12.5) 샘플 하나의 평균 μ
        var = x.var(axis=-1, keepdims=True)         # (12.6) 샘플 하나의 분산 σ²
        x_hat = (x - mu) / np.sqrt(var + EPS)       # (12.7)
        return self.gamma * x_hat + self.beta       # (12.8)


if __name__ == "__main__":
    # ---- 손계산 예제: 값 [2, 4, 6, 8] 하나의 특징 ----
    v = np.array([[2.0], [4.0], [6.0], [8.0]])
    print("평균 5, 분산 5 ->", BatchNorm(1)(v).ravel().round(3))   # [-1.342 -0.447 0.447 1.342]

    # ---- 평균 3, 표준편차 5 인 '찌그러진' 데이터를 정규화 ----
    rng = np.random.default_rng(0)
    x = rng.normal(3, 5, size=(64, 4))
    bn, ln = BatchNorm(4), LayerNorm(4)
    yb, yl = bn(x), ln(x)
    print("입력      열별 평균", x.mean(0).round(2), "표준편차", x.std(0).round(2))
    print("BN 출력   열별 평균", yb.mean(0).round(3), "표준편차", yb.std(0).round(3))
    print("LN 출력   행별 평균(앞 3개)", yl.mean(1)[:3].round(3), "표준편차", yl.std(1)[:3].round(3))

    # ---- 추론 모드: 배치 크기 1 이어도 동작 (이동 평균 통계 사용) ----
    for _ in range(100):
        bn(rng.normal(3, 5, size=(64, 4)))
    print("추론(샘플 1개):", bn(x[:1], training=False).round(2))

    # ---- PyTorch 내장 구현과 같은 값인지 검증 ----
    import torch
    t = torch.tensor(x, dtype=torch.float32)
    ref_b = torch.nn.BatchNorm1d(4, eps=EPS)(t).detach().numpy()
    ref_l = torch.nn.LayerNorm(4, eps=EPS)(t).detach().numpy()
    print("BN 일치:", np.allclose(yb, ref_b, atol=1e-4), " LN 일치:", np.allclose(yl, ref_l, atol=1e-4))
