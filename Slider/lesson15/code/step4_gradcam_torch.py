"""Step 4. PyTorch Grad-CAM (hook 사용) + '지름길(워터마크)' 실험 + Integrated Gradients(보너스)
숫자 이미지(8x8→16x16)에서 학습 때 클래스 7에만 왼쪽 위 모서리에 밝은 '도장'을 찍어 두면,
모델이 숫자 모양이 아니라 '도장'을 보고 답하게 되는지(= 영수증 속 강조 텍스트 사례와 같은 문제)를 Grad-CAM 으로 확인한다.
실행: python step4_gradcam_torch.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits

def load(seed=0):
    d = load_digits(); X = torch.tensor(d.images / 16.0, dtype=torch.float32)[:, None]      # (N,1,8,8)
    torch.manual_seed(123); X = (X + 0.2 * torch.randn_like(X)).clamp(0, 1)                 # 잡음: 모양 단서를 흐리게 → 도장이 더 쉬운 지름길이 된다
    X = F.interpolate(X, size=16, mode="bilinear", align_corners=False)                      # 16x16 으로 키워 히트맵이 보이게
    y = torch.tensor(d.target); idx = np.random.default_rng(seed).permutation(len(y)); cut = int(0.8 * len(y))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]

def stamp(X):
    """왼쪽 위 4×4 칸을 밝게(=워터마크). 원본은 건드리지 않고 복사본을 돌려준다."""
    X = X.clone(); X[:, :, :4, :4] = 1.0; return X

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1); self.c2 = nn.Conv2d(8, 16, 3, padding=1); self.c3 = nn.Conv2d(16, 16, 3, padding=1); self.relu3 = nn.ReLU(); self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)       # (8,8,8)
        A = self.relu3(self.c3(F.relu(self.c2(x))))   # (16,8,8)  ← Grad-CAM 이 들여다볼 '마지막 합성곱 층+ReLU' 출력
        return self.fc(A.mean(dim=(2, 3)))            # 전역 평균 풀링(GAP) → 선형 → 클래스 점수(softmax 이전)

def train(X, y, epochs=60, seed=0):
    torch.manual_seed(seed); net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(epochs):
        p = torch.randperm(len(y))
        for i in range(0, len(y), 64):
            b = p[i:i + 64]; opt.zero_grad(); F.cross_entropy(net(X[b]), y[b]).backward(); opt.step()
    return net.eval()

@torch.no_grad()
def acc(net, X, y): return (net(X).argmax(1) == y).float().mean().item()

def grad_cam(net, x, c=None):
    """x: (1,1,16,16). A 와 ∂y^c/∂A 를 붙잡아 식 15.1, 15.2 를 계산. 16x16 히트맵(0~1)과 예측 클래스를 반환"""
    store = {}
    def fwd_hook(module, inp, out):                      # ① 순전파 중 relu3 를 지날 때 호출: A 를 저장
        store["A"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))     # ② 나중에 A 로 기울기가 돌아오면 그것도 저장
    h = net.relu3.register_forward_hook(fwd_hook)
    net.zero_grad(); out = net(x); h.remove()
    c = int(out.argmax(1)) if c is None else c
    out[0, c].backward()                                 # ③ y^c 에서 역전파 → store["g"] = ∂y^c/∂A
    A, g = store["A"].detach(), store["g"]
    alpha = g.mean(dim=(2, 3), keepdim=True)        # 식 15.1: 채널마다 기울기의 공간 평균
    cam = F.relu((alpha * A).sum(1, keepdim=True))       # 식 15.2: 가중합 후 ReLU
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    return (cam / cam.max().clamp_min(1e-8)).numpy(), c

def integrated_gradients(net, x, c, steps=64):
    """보너스: 기준(검은 이미지)에서 x 까지 직선으로 걸으며 기울기를 평균 → 입력 × 평균 기울기. 완전성: 합 = f(x)−f(기준)"""
    base = torch.zeros_like(x); a = torch.linspace(0, 1, steps + 1)[1:].view(-1, 1, 1, 1)
    path = (base + a * (x - base)).requires_grad_(True)
    net(path)[:, c].sum().backward()
    return ((x - base) * path.grad.mean(0, keepdim=True)).detach()[0, 0]

if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    clean = train(Xtr, ytr, seed=0)                                  # 정상 학습 모델
    Xs = Xtr.clone(); m7 = ytr == 7; Xs[m7] = stamp(Xtr[m7])         # 클래스 7 학습 사진에만 도장
    spur = train(Xs, ytr, seed=0)                                    # 지름길이 있는 학습 데이터로 학습한 모델
    n7 = yte != 7
    print("정상 모델   : 깨끗한 테스트 정확도 %.3f | 비-7 에 도장을 찍으면 7 로 답하는 비율 %.3f" % (acc(clean, Xte, yte), (clean(stamp(Xte[n7])).argmax(1) == 7).float().mean()))
    print("지름길 모델 : 깨끗한 테스트 정확도 %.3f | 비-7 에 도장을 찍으면 7 로 답하는 비율 %.3f" % (acc(spur, Xte, yte), (spur(stamp(Xte[n7])).argmax(1) == 7).float().mean()))
    # ---- Grad-CAM: 도장 찍힌 비-7 이미지들에서 '클래스 7' 근거가 어디에 있는가 ----
    idx = torch.where(n7)[0][:100]                                   # 테스트의 비-7 이미지 100장
    def corner_share(net):
        r = []
        for i in idx:
            cam, _ = grad_cam(net, stamp(Xte[i:i + 1]), c=7)         # 클래스 7 에 대한 Grad-CAM
            r.append(cam[:8, :8].sum() / max(cam.sum(), 1e-8))       # 히트맵 총합 중 왼쪽 위 1/4(8×8) 구역이 차지하는 몫
        return float(np.mean(r))
    print("히트맵 에너지 중 '왼쪽 위 1/4 구역(도장 자리)' 몫 (구역 면적 비율은 25%%):  정상 %.2f | 지름길 %.2f" % (corner_share(clean), corner_share(spur)))
    fooled = [int(i) for i in idx if spur(stamp(Xte[i:i + 1])).argmax(1).item() == 7]
    print("지름길 모델이 도장 때문에 7 로 오답한 예:", len(fooled), "장 / 100장")
    # ---- IG 완전성 검사 ----
    xx = Xte[:1]; c = int(clean(xx).argmax(1)); ig = integrated_gradients(clean, xx, c, 256)
    d = (clean(xx)[0, c] - clean(torch.zeros_like(xx))[0, c]).item()
    print(f"IG 완전성: Σ속성 = {ig.sum():.3f}  vs  f(x)−f(기준) = {d:.3f}"); assert abs(ig.sum().item() - d) < 0.05 * max(1, abs(d))
    torch.save(dict(clean=clean.state_dict(), spur=spur.state_dict()), "_models.pt")
