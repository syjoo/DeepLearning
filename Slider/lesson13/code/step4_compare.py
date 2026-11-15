"""Step 4. 종합 실험: 증강이 작은 데이터셋에서 일반화를 얼마나 높이는가 (PyTorch)
8x8 숫자 이미지 중 150장만 학습에 쓰고, 나머지로 검증한다. 같은 모델·같은 학습 횟수에서 증강만 바꿔 비교한다.
실행: python step4_compare.py   (결과 그림: ../images/aug_compare.png, 5회 평균)
"""
import numpy as np, torch, torch.nn as nn
from step1_transforms import load
from step3_pipeline import RandomAffine, GaussianNoise, HFlip, Compose

X, y = load(); X = torch.tensor(X).unsqueeze(1); y = torch.tensor(y)
N_TRAIN, EPOCHS = 150, 200
MILD   = lambda: Compose([RandomAffine(10, 1), GaussianNoise(0.05)])
STRONG = lambda: Compose([RandomAffine(60, 3), GaussianNoise(0.4)])      # 너무 센 증강
FLIP   = lambda: Compose([HFlip(0.5), RandomAffine(10, 1)])              # 레이블을 깨는 증강
CASES = {"증강 없음": None, "약한 증강 (온라인)": MILD, "오프라인 5배": "offline", "너무 센 증강": STRONG, "좌우 뒤집기 (나쁜 예)": FLIP}

def model():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 2 * 2, 10))

def run(case, seed):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    tr, rest = perm[:N_TRAIN], perm[N_TRAIN:]
    xtr, ytr, xv, yv = X[tr], y[tr], X[rest], y[rest]
    # '배포 환경'을 흉내 낸 어려운 검증셋: 약간 돌아가고, 밀리고, 노이즈가 낀 같은 숫자들 (한 번 만들어 고정)
    torch.manual_seed(1000 + seed); xo = MILD()(xv); torch.manual_seed(seed)
    T = CASES[case]
    if T == "offline":                                           # 미리 5배로 늘려 저장 -> 이후로는 고정
        xs, ys = [xtr], [ytr]
        for _ in range(4): xs.append(MILD()(xtr)); ys.append(ytr)
        xtr, ytr, T = torch.cat(xs), torch.cat(ys), None
    f = T() if T else (lambda a: a)
    net, loss_fn = model(), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    steps = EPOCHS * 150 // 32                                   # 오프라인도 같은 업데이트 횟수
    for s in range(steps):
        idx = torch.randint(0, len(xtr), (32,))
        loss = loss_fn(net(f(xtr[idx])), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        r = lambda a, b: (loss_fn(net(a), b).item(), (net(a).argmax(1) == b).float().mean().item())
        return (*r(X[tr], y[tr]), *r(xv, yv), *r(xo, yv))       # 학습손실, 학습정확도, 검증손실, 검증정확도, 어려운 검증 손실, 정확도

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {c: np.array([run(c, s) for s in range(5)]) for c in CASES}
    print(f"{'방식':22s}{'학습손실':>8s}{'검증손실':>8s}{'검증정확도':>10s}{'어려운 검증':>10s}")
    for c, a in res.items():
        m = a.mean(0); print(f"{c:22s}{m[0]:8.3f}{m[2]:8.3f}{m[3]*100:9.1f}%{m[5]*100:9.1f}%")
    np.save("_res.npy", np.array([res[c].mean(0) for c in CASES])); np.save("_res_std.npy", np.array([res[c].std(0) for c in CASES]))
