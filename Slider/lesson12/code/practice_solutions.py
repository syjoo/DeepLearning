"""실습 문제 풀이 (슬라이드 마지막의 솔루션과 같은 내용). 실행: python practice_solutions.py"""
import numpy as np
from step2_init import forward_stds, relu, he
from step3_norm import BatchNorm, LayerNorm, EPS

# ---------- 문제 1. 손으로 배치 정규화 ----------
# x = [1, 3, 5, 7, 9] (한 특징의 값 5개). γ=1, β=0, ε는 무시한다.
x = np.array([1., 3, 5, 7, 9])
mu, var = x.mean(), x.var()               # 평균 5, 분산 (16+4+0+4+16)/5 = 8
print("Q1  평균", mu, "분산", var, "정규화", ((x - mu) / np.sqrt(var)).round(3))

# ---------- 문제 2. 초기화 표준편차 계산 ----------
n_in, n_out = 256, 128                    # 입력 256개, 출력 128개인 층
print("Q2  Xavier 분산", 2 / (n_in + n_out), "표준편차", np.sqrt(2 / (n_in + n_out)).round(4))
print("    He     분산", 2 / n_in,           "표준편차", np.sqrt(2 / n_in).round(4))

# ---------- 문제 3. RMSNorm 구현 (평균을 빼지 않고 제곱평균제곱근으로만 나눈다) ----------
class RMSNorm:
    def __init__(self, dim):
        self.gamma = np.ones(dim)
    def __call__(self, x):
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)   # 제곱의 평균에 루트
        return self.gamma * x / rms
z = np.array([[1., 2, 3, 4]])
print("Q3  RMSNorm", RMSNorm(4)(z).round(3), " (LayerNorm", LayerNorm(4)(z).round(3), ")")

# ---------- 문제 4. He 대신 분산 1/n_in 을 쓰면 (ReLU, 10층)? ----------
half = lambda n_in, n_out: np.sqrt(1 / n_in)
print("Q4  He 10층 후 std", round(forward_stds(relu, he)[-1], 3), " 1/n_in 10층 후 std", round(forward_stds(relu, half)[-1], 4))
print("    He 의 결과는 1/n_in 보다 약", round(forward_stds(relu, he)[-1] / forward_stds(relu, half)[-1]), "배 크다 (층마다 신호가 약 0.7배씩 줄어든 탓)")

# ---------- 문제 5. 배치 크기 1 로 BN(훈련 모드)을 호출하면? ----------
bn = BatchNorm(3); bn.beta[:] = [0.5, 0.5, 0.5]
one = np.array([[10., -3, 7]])
print("Q5  샘플 1개 훈련모드 BN 출력", bn(one, training=True), "-> 값과 무관하게 β 만 남는다")
print("    LayerNorm 은 샘플 1개여도 정상:", LayerNorm(3)(one).round(3))
