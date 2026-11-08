"""
실습 문제 정답 (슬라이드 마지막 '솔루션' 부분과 같은 내용)
실행:  python practice_solutions.py     (numpy_version, pytorch_version 폴더를 자동으로 불러온다)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "numpy_version"))
sys.path.insert(0, os.path.join(HERE, "pytorch_version"))

# ------------------------------------------------------------------
# 문제 1. 손계산 어텐션: 토큰 3개, d_k = 2
#   Q = K = V = [[1,0],[0,1],[1,1]]  일 때 (마스크 없음) 첫 번째 토큰의 어텐션 가중치와 출력을 구하라.
# ------------------------------------------------------------------
from step3_attention import scaled_dot_product_attention, softmax

X = np.array([[1., 0.], [0., 1.], [1., 1.]])
out, w = scaled_dot_product_attention(X, X, X)
print("[문제1] 첫 토큰 점수 = Q0.K / sqrt(2) =", (X[0] @ X.T / np.sqrt(2)).round(3))   # [0.707 0 0.707]
print("        가중치 =", w[0].round(3), " 출력 =", out[0].round(3))                    # [0.401 0.198 0.401]

# ------------------------------------------------------------------
# 문제 2. 파라미터 수:  d_model = 32, 헤드 4개, 블록 2개, 어휘 28개 (NumPy 버전 모델)
#   Linear(n_in, n_out) = n_in*n_out + n_out
#   어텐션 = Linear 4개(q,k,v,o)      = 4*(32*32+32) = 4224
#   FFN    = 32->128, 128->32          = (32*128+128) + (128*32+32) = 8352
#   LayerNorm 2개 = 2*(32+32) = 128   -> 블록 1개 = 12704
#   전체 = 임베딩 28*32 + 블록 2*12704 + 마지막 LN 64 + 출력층(32*28+28)
# ------------------------------------------------------------------
d, V, L = 32, 28, 2
block = 4 * (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d) + 2 * (2 * d)
total = V * d + L * block + 2 * d + (d * V + V)
print("\n[문제2] 블록 1개 =", block, " 전체 =", total)       # 12704, 27292 (step6 출력과 같다)

# ------------------------------------------------------------------
# 문제 3. NumPy 로 top-k 샘플링 구현:  점수가 높은 k개 글자만 후보로 남기고 뽑는다.
# ------------------------------------------------------------------
def top_k_sample(logits, k, rng, temperature=1.0):
    logits = logits / temperature
    cut = np.sort(logits)[-k]                         # 상위 k번째 점수 (이보다 작으면 탈락)
    logits = np.where(logits < cut, -np.inf, logits)  # 탈락한 글자는 확률 0
    p = softmax(logits)
    return int(rng.choice(len(p), p=p))

rng = np.random.default_rng(0)
logits = np.array([1.0, 3.0, 0.5, 2.5, -1.0])
picks = [top_k_sample(logits, 2, rng) for _ in range(1000)]
print("\n[문제3] top-2 로 1000번 뽑은 결과 번호:", sorted(set(picks)))   # 1번과 3번만 나와야 한다
assert set(picks) <= {1, 3}

# ------------------------------------------------------------------
# 문제 4. 층 수(n_layers)를 1, 2, 4 로 바꿔 학습했을 때 마지막 loss 를 비교하라. (PyTorch, 300 step)
# ------------------------------------------------------------------
import torch
from step5_gpt_train import GPT

text = open(os.path.join(HERE, "tiny_corpus.txt"), encoding="utf-8").read()
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
CTX, BATCH = 32, 32

print("\n[문제4] n_layers 에 따른 마지막 50 step 평균 loss")
for n_layers in (1, 2, 4):
    torch.manual_seed(0)
    model = GPT(len(chars), d_model=64, n_heads=4, n_layers=n_layers, max_len=CTX, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(300):
        s = torch.randint(0, len(data) - CTX - 1, (BATCH,))
        x = torch.stack([data[i:i + CTX] for i in s])
        y = torch.stack([data[i + 1:i + CTX + 1] for i in s])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  n_layers={n_layers}: loss {np.mean(losses[-50:]):.3f}")
