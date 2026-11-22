"""Step 3. Grad-CAM 을 NumPy 로 밑바닥부터 (손으로 만든 아주 작은 CNN)
구조:  입력(10x10) → 3×3 합성곱 3개 + ReLU → 특징 맵 A (3,10,10) → 2×2 평균 풀링 → 펼치기 → FC → 점수 y^c
  식 15.1  α_k^c = (1/Z) Σ_ij ∂y^c/∂A^k_ij          (Z = H·W)
  식 15.2  L^c   = ReLU( Σ_k α_k^c · A^k )
실행: python step3_gradcam_numpy.py
"""
import numpy as np

# ---------------- 1) 손으로 만든 장난감 이미지: 십자(+), 네모(■), X 모양이 각각 다른 곳에 ----------------
def make_image():
    x = np.zeros((10, 10))
    x[2, 1:4] = 1; x[1:4, 2] = 1                      # 십자(+): 중심 (2,2)   → 왼쪽 위
    x[6:9, 6:9] = 1                                   # 네모(■): 중심 (7,7)   → 오른쪽 아래
    for d in (-1, 0, 1): x[2 + d, 7 + d] = 1; x[2 + d, 7 - d] = 1   # X: 중심 (2,7) → 오른쪽 위
    return x

# ---------------- 2) 손으로 정한 가중치: 필터 3개 = 십자/네모/X 탐지기 ----------------
K = np.array([[[-1, 1, -1], [1, 1, 1], [-1, 1, -1]],      # 필터0: 십자 탐지기
              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],          # 필터1: 네모 탐지기
              [[1, -1, 1], [-1, 1, -1], [1, -1, 1]]], float)   # 필터2: X 탐지기
BIAS = np.array([-3.5, -5.5, -2.0])                       # 약한 반응은 ReLU가 꺼 버리도록 편향을 둔다
# 클래스 0 = "십자가 있다", 클래스 1 = "네모가 있다". 클래스 0에게 X는 '반대 증거'(−)
Wfc = np.zeros((2, 3, 5, 5)); Wfc[0, 0] = 1.0; Wfc[0, 2] = -1.0; Wfc[1, 1] = 1.0

def conv_relu(x):
    """같은 크기(zero padding) 3×3 합성곱 + 편향 + ReLU → 특징 맵 A (3,10,10)"""
    xp = np.pad(x, 1); A = np.zeros((3, 10, 10))
    for k in range(3):
        for i in range(10):
            for j in range(10):
                A[k, i, j] = (xp[i:i + 3, j:j + 3] * K[k]).sum() + BIAS[k]
    return np.maximum(A, 0)

def head(A):
    """A 이후의 부분: 2×2 평균 풀링 → FC.  점수 y (2,) — softmax 이전 값"""
    P = A.reshape(3, 5, 2, 5, 2).mean(axis=(2, 4))          # (3,5,5)
    return (Wfc * P).sum(axis=(1, 2, 3))

def grad_wrt_A(c):
    """∂y^c/∂A 를 손으로 미분: y^c = Σ W·P,  P = 풀링(A) → 각 A 픽셀은 W/4 의 영향 (4칸 평균이므로)"""
    return np.repeat(np.repeat(Wfc[c] / 4.0, 2, axis=1), 2, axis=2)       # (3,10,10)

def grad_cam(x, c):
    A = conv_relu(x)
    g = grad_wrt_A(c)
    alpha = g.mean(axis=(1, 2))                                           # 식 15.1: 공간 평균 → 채널별 중요도 α_k
    s = np.tensordot(alpha, A, axes=1)                                    # Σ_k α_k A^k  (ReLU 전)
    return np.maximum(s, 0), s, alpha, A                                  # 식 15.2: ReLU

if __name__ == "__main__":
    x = make_image(); A = conv_relu(x)
    print("점수 y =", head(A), " (클래스0=십자, 클래스1=네모)")
    # ---- 그래디언트 검사: 손 미분 vs 수치 미분(아주 조금 흔들어 보기) ----
    rng = np.random.default_rng(0); worst = 0
    for c in (0, 1):
        g = grad_wrt_A(c)
        for _ in range(40):
            k, i, j = rng.integers(3), rng.integers(10), rng.integers(10); e = 1e-5
            Ap, Am = A.copy(), A.copy(); Ap[k, i, j] += e; Am[k, i, j] -= e
            num = (head(Ap)[c] - head(Am)[c]) / (2 * e)
            worst = max(worst, abs(num - g[k, i, j]))
    print("그래디언트 검사: 손 미분 vs 수치 미분 최대 오차 = %.2e" % worst); assert worst < 1e-6
    # ---- Grad-CAM ----
    for c, name in ((0, "십자"), (1, "네모")):
        cam, s, alpha, _ = grad_cam(x, c)
        pk = np.unravel_index(cam.argmax(), cam.shape)
        print(f"\n[클래스 {c}: {name}]  α = {np.round(alpha, 3)}   (X 탐지기 채널의 α가 음수인지 확인)")
        print(f"  ReLU 전 최솟값 {s.min():.2f} → ReLU 후 최솟값 {cam.min():.2f},  히트맵 최대 위치 = {tuple(int(v) for v in pk)}")
    cam0 = grad_cam(x, 0)[0]; cam1 = grad_cam(x, 1)[0]
    assert np.unravel_index(cam0.argmax(), cam0.shape) == (2, 2) and np.unravel_index(cam1.argmax(), cam1.shape) == (7, 7)
    print("\n클래스 0 히트맵은 십자(2,2), 클래스 1 히트맵은 네모(7,7) 를 가리킨다 ✔")
    np.save("_gc_np.npy", dict(x=x, A=A, cam0=cam0, cam1=cam1, s0=grad_cam(x, 0)[1], alpha0=grad_cam(x, 0)[2]), allow_pickle=True)
