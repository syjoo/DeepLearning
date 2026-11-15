"""실습 문제 솔루션. 실행: python practice_solutions.py"""
import numpy as np, torch, torch.nn as nn
from step1_transforms import load
from step3_pipeline import RandomAffine, GaussianNoise, Compose

# ---- 문제 1. 레이블 일관성 판정 (정답표는 슬라이드/해설집 참고) ----
print("문제1: 손으로 푸는 문제 - 해설집 참고")

# ---- 문제 2. 오프라인 증강의 저장 용량 ----
def storage(h, w, c, n_images, copies):
    return h * w * c * n_images * copies                       # uint8 = 1바이트
mnist = storage(28, 28, 1, 1000, 10); photo = storage(224, 224, 3, 1000, 10)
print(f"문제2: 28x28x1 {mnist/1e6:.2f} MB, 224x224x3 {photo/1e9:.2f} GB")
assert mnist == 7_840_000 and photo == 1_505_280_000
# 온라인: 1000장 x 100 에포크 = 100,000번 서로 다른 변형을 본다 (저장 용량 증가 없음)
print("       온라인 100 에포크에서 본 변형 횟수:", 1000 * 100)

# ---- 문제 3. Mixup 손계산 ----
lam = 0.7
x1, x2 = np.array([1.0, 0.0, 0.4]), np.array([0.0, 1.0, 0.8])
y1, y2 = np.array([1, 0, 0]), np.array([0, 1, 0])               # 원-핫: 고양이, 강아지
print("문제3: 섞인 입력", lam * x1 + (1 - lam) * x2, "섞인 정답", lam * y1 + (1 - lam) * y2)
assert np.allclose(lam * x1 + (1 - lam) * x2, [0.7, 0.3, 0.52])

# ---- 문제 4. 밝기/대비 무작위 변환 구현 ----
def random_brightness_contrast(img, rng, b_max=0.2, c_lo=0.8, c_hi=1.2):
    b, c = rng.uniform(-b_max, b_max), rng.uniform(c_lo, c_hi)  # 한 이미지 안에서는 같은 값을 모든 픽셀에
    return np.clip((img - 0.5) * c + 0.5 + b, 0, 1)             # 대비는 '중간 회색(0.5)'을 기준으로 늘이고 줄인다
X, _ = load(); rng = np.random.default_rng(0)
outs = [random_brightness_contrast(X[0], rng) for _ in range(100)]
assert all(0 <= o.min() and o.max() <= 1 for o in outs), "범위를 벗어나면 안 된다"
assert np.allclose(random_brightness_contrast(X[0], np.random.default_rng(0), 0, 1, 1), X[0]), "b=0,c=1이면 원본"
print("문제4: 평균 밝기 범위", min(o.mean() for o in outs).round(3), "~", max(o.mean() for o in outs).round(3), "(원본", X[0].mean().round(3), ")")

# ---- 문제 5. 데이터 누수: 증강을 '분할 전에' 하면 검증 점수가 부풀려진다 ----
def make_net():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(128, 10))
def fit_eval(xt, yt, xe, ye, seed):
    torch.manual_seed(seed); net = make_net(); opt = torch.optim.Adam(net.parameters(), 3e-3); ce = nn.CrossEntropyLoss()
    for _ in range(300):
        i = torch.randint(0, len(xt), (32,)); loss = ce(net(xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return [(net(a).argmax(1) == b).float().mean().item() for a, b in zip(xe, ye)]
Xt, yt = torch.tensor(X).unsqueeze(1), torch.tensor(load()[1])
aug = Compose([RandomAffine(10, 1), GaussianNoise(0.05)])
leak, ok = [], []
for seed in range(5):
    torch.manual_seed(seed); p = torch.randperm(len(Xt)); pool, held = p[:300], p[300:]      # 원본 300장 + 완전히 따로 둔 시험용
    xs = torch.cat([Xt[pool]] + [aug(Xt[pool]) for _ in range(4)]); ys = yt[pool].repeat(5)  # 5배로 늘림
    q = torch.randperm(len(xs)); tr, va = q[:1000], q[1000:]                                 # (나쁨) 늘린 뒤에 분할 -> 같은 원본의 변형이 양쪽에
    leak.append(fit_eval(xs[tr], ys[tr], [xs[va], Xt[held]], [ys[va], yt[held]], seed))
    tr_o, va_o = pool[:200], pool[200:]                                                      # (좋음) 원본을 먼저 분할한 뒤 학습 쪽만 증강
    xs2 = torch.cat([Xt[tr_o]] + [aug(Xt[tr_o]) for _ in range(4)]); ys2 = yt[tr_o].repeat(5)
    xv2 = torch.cat([aug(Xt[va_o]) for _ in range(4)]); yv2 = yt[va_o].repeat(4)                # 검증도 같은 종류(변형본)로 맞춘다: 공정 비교
    ok.append(fit_eval(xs2, ys2, [xv2, Xt[held]], [yv2, yt[held]], seed))
leak, ok = np.array(leak).mean(0) * 100, np.array(ok).mean(0) * 100
print(f"문제5: 늘린 뒤 분할 -> 검증 {leak[0]:.1f}%  (진짜 시험 {leak[1]:.1f}%)")
print(f"       분할 뒤 늘림 -> 검증 {ok[0]:.1f}%  (진짜 시험 {ok[1]:.1f}%)")
