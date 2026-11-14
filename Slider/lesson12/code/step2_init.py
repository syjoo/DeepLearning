"""Step 2. 가중치 초기화 실험: 분산이 신호를 폭발/소실시키는 모습 보기
실행: python step2_init.py
"""
import numpy as np

def forward_stds(act, std_fn, layers=10, width=512, n=1000, seed=0):
    """10층짜리 신경망에 데이터를 흘려 보내며 '층마다 활성값의 표준편차'를 기록한다.
    act    : 활성화 함수
    std_fn : (n_in, n_out) -> 가중치 표준편차를 정하는 규칙 (초기화 방식)
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, width))          # 입력: 평균 0, 표준편차 1
    stds = [a.std()]
    for _ in range(layers):
        W = rng.normal(0, std_fn(width, width), size=(width, width))   # W ~ N(0, std^2)
        a = act(a @ W)                       # 선형 변환 후 활성화 함수
        stds.append(a.std())
    return stds

tanh = np.tanh
relu = lambda z: np.maximum(0, z)

# 초기화 방식 = '표준편차를 정하는 규칙'
small  = lambda n_in, n_out: 0.01                       # 너무 작게
big    = lambda n_in, n_out: 1.0                        # 너무 크게
xavier = lambda n_in, n_out: np.sqrt(2 / (n_in + n_out))  # Var = 2/(n_in+n_out)
he     = lambda n_in, n_out: np.sqrt(2 / n_in)            # Var = 2/n_in

cases = {
    "tanh + std=0.01 (너무 작음)": (tanh, small),
    "tanh + std=1.0  (너무 큼)":   (tanh, big),   # tanh는 +-1에서 '포화'되어 기울기가 0에 가까워진다
    "tanh + Xavier":               (tanh, xavier),
    "ReLU + std=0.1  (조금 큼)":  (relu, lambda n_in, n_out: 0.1),
    "ReLU + Xavier":               (relu, xavier),
    "ReLU + He":                   (relu, he),
}
if __name__ == "__main__":
    for name, (f, s) in cases.items():
        st = forward_stds(f, s)
        print(f"{name:28s} 층별 std:", " ".join(f"{v:.3g}" for v in st[1::3]), "... 10층:", f"{st[-1]:.3g}")
