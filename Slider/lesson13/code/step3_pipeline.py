"""Step 3. 온라인 증강 파이프라인 (PyTorch, 배치 단위)
'작고 조합 가능한 변환'을 순서대로 이어 붙이는 Compose 구조. 매 호출마다 새로 뽑은 무작위 값을 쓴다 = 온라인 증강.
"""
import math, torch
import torch.nn.functional as F

class RandomAffine:
    """회전(±deg도) + 이동(±shift픽셀)을 샘플마다 다르게 적용. x: (N,1,H,W)"""
    def __init__(self, deg=0, shift=0): self.deg, self.shift = deg, shift
    def __call__(self, x):
        n, _, h, w = x.shape
        th = (torch.rand(n) * 2 - 1) * self.deg * math.pi / 180                 # 샘플마다 다른 각도
        t = (torch.rand(n, 2) * 2 - 1) * self.shift * 2 / h                     # 픽셀 -> [-1,1] 좌표 단위
        A = torch.zeros(n, 2, 3)
        A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = th.cos(), th.sin(), -th.sin(), th.cos()
        A[:, :, 2] = t
        grid = F.affine_grid(A, x.shape, align_corners=False)
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class GaussianNoise:
    def __init__(self, sigma): self.sigma = sigma
    def __call__(self, x): return (x + torch.randn_like(x) * self.sigma).clamp(0, 1)

class HFlip:                      # 숫자에는 '레이블을 깨는' 변환 (나쁜 예시용)
    def __init__(self, p=0.5): self.p = p
    def __call__(self, x):
        m = (torch.rand(len(x)) < self.p).view(-1, 1, 1, 1)
        return torch.where(m, x.flip(-1), x)

class Compose:
    def __init__(self, ts): self.ts = ts
    def __call__(self, x):
        for t in self.ts: x = t(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(4, 1, 8, 8)
    out = Compose([RandomAffine(deg=15, shift=1), GaussianNoise(0.05)])(x)
    print("모양 유지:", out.shape == x.shape, "| 같은 입력을 두 번 넣으면 결과가 다르다:",
          not torch.allclose(RandomAffine(15, 1)(x), RandomAffine(15, 1)(x)))
    same = RandomAffine(0, 0)(x)
    print("회전 0, 이동 0이면 원본과 같다:", torch.allclose(same, x, atol=1e-5))
