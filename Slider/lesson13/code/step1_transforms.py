"""Step 1. 변환(Transformation)을 NumPy로 직접 만든다.
이미지는 0~1 값의 2차원 배열. 라이브러리 없이 '뒤집기, 이동, 회전, 확대, 밝기/대비, 노이즈'를 구현한다.
실행: python step1_transforms.py
"""
import numpy as np
from sklearn.datasets import load_digits

def load():
    d = load_digits()
    return (d.images / 16.0).astype(np.float32), d.target        # (1797, 8, 8), 0~1

# 1) 뒤집기: 열 순서를 거꾸로
def flip_h(img): return img[:, ::-1]

# 2) 아핀 변환의 핵심: '출력 픽셀이 입력의 어느 위치에서 왔는가'를 거꾸로 계산하고(역변환),
#    그 위치의 값을 이웃 4픽셀의 가중평균(쌍선형 보간)으로 읽어 온다. 범위 밖은 0(검정).
def warp(img, A, t):
    """출력 좌표 p -> 입력 좌표 A@(p-c)+c+t.  c = 이미지 중심"""
    h, w = img.shape
    c = np.array([(h - 1) / 2, (w - 1) / 2])
    ys, xs = np.mgrid[0:h, 0:w]
    p = np.stack([ys, xs], -1).reshape(-1, 2) - c               # 중심 기준 좌표
    q = p @ A.T + c + t                                          # 입력에서 읽을 위치
    y0, x0 = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
    fy, fx = q[:, 0] - y0, q[:, 1] - x0
    def px(y, x):                                                # 범위 밖이면 0
        ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        return np.where(ok, img[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)], 0.0)
    out = (px(y0, x0) * (1 - fy) * (1 - fx) + px(y0, x0 + 1) * (1 - fy) * fx
           + px(y0 + 1, x0) * fy * (1 - fx) + px(y0 + 1, x0 + 1) * fy * fx)
    return out.reshape(h, w).astype(np.float32)

def rotate(img, deg):
    th = np.deg2rad(deg); A = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    return warp(img, A, np.zeros(2))
def shift(img, dy, dx): return warp(img, np.eye(2), np.array([-dy, -dx], float))
def zoom(img, s): return warp(img, np.eye(2) / s, np.zeros(2))   # s>1 확대(= 가운데를 잘라 키움), s<1 축소

# 3) 밝기/대비/노이즈: 값만 바꾼다 (위치는 그대로)
def brightness_contrast(img, b=0.0, c=1.0): return np.clip((img - 0.5) * c + 0.5 + b, 0, 1)
def add_noise(img, sigma, rng): return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1).astype(np.float32)

if __name__ == "__main__":
    X, y = load(); rng = np.random.default_rng(0)
    img = X[0]
    assert np.allclose(rotate(img, 0), img, atol=1e-6), "회전 0도는 원본과 같아야 한다"
    assert np.allclose(shift(img, 0, 0), img, atol=1e-6)
    print("원본 합:", img.sum().round(3), "| 오른쪽으로 1칸 이동 후 합:", shift(img, 0, 1).sum().round(3))
    print("밝기 +0.2 평균:", brightness_contrast(img, 0.2).mean().round(3), "원본 평균:", img.mean().round(3))
    print("노이즈 0.1 후 원본과의 평균 차이:", np.abs(add_noise(img, 0.1, rng) - img).mean().round(3))
