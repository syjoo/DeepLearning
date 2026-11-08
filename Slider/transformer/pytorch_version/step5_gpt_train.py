"""
[Step 5] 전체 모델(GPT) 조립 + 학습 + 글 생성  -  PyTorch 버전
=====================================================================
NumPy 버전 Step 6 과 같은 모델/데이터지만, 실전 도구를 사용한다.
  - 자동미분(loss.backward)          - AdamW 옵티마이저
  - 기울기 클리핑(폭주 방지)          - GPU 자동 사용
  - 학습/평가 모드(model.train/eval)  - 체크포인트 저장
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from step1_embedding_pe import TokenAndPosition
from step4_block import Block


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3, max_len=64, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.embed = TokenAndPosition(vocab_size, d_model, max_len)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, max_len, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embed(idx)                                  # (B, T, D)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                     # (B, T, V)
        loss = None
        if targets is not None:                              # (B*T, V) vs (B*T,) 로 펴서 계산
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()                                         # 생성할 때는 기울기 계산이 필요 없다
    def generate(self, idx, n_new, temperature=0.8, top_k=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.max_len:])         # 최근 max_len 토큰만 입력
            logits = logits[:, -1, :] / temperature          # 마지막 위치의 점수
            if top_k is not None:                            # 확률 높은 k개만 후보로 남긴다
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)                # 확률대로 한 글자 뽑기
            idx = torch.cat([idx, nxt], dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 데이터 ----------
    path = os.path.join(os.path.dirname(__file__), "..", "tiny_corpus.txt")
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], device=device)

    CTX, BATCH, STEPS = 32, 32, 600

    def get_batch():
        s = torch.randint(0, len(data) - CTX - 1, (BATCH,))
        x = torch.stack([data[i:i + CTX] for i in s])           # 입력
        y = torch.stack([data[i + 1:i + CTX + 1] for i in s])   # 정답 = 한 글자 뒤
        return x, y

    # ---------- 모델 / 옵티마이저 ----------
    model = GPT(len(chars), d_model=64, n_heads=4, n_layers=3, max_len=CTX).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    print(f"device={device}, 파라미터 수={sum(p.numel() for p in model.parameters()):,}")

    # ---------- 학습 루프 ----------
    model.train()                                    # dropout 켜기
    for step in range(1, STEPS + 1):
        x, y = get_batch()
        _, loss = model(x, y)                        # 1) 순전파 + 손실
        opt.zero_grad(set_to_none=True)              # 2) 이전 기울기 지우기
        loss.backward()                              # 3) 역전파 (자동미분!)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 4) 기울기가 너무 크면 잘라 줌
        opt.step()                                   # 5) 가중치 갱신
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    # ---------- 생성 ----------
    model.eval()                                     # dropout 끄기
    start = torch.tensor([[stoi[c] for c in "the "]], device=device)
    out = model.generate(start, 150, temperature=0.7, top_k=5)[0].tolist()
    print("\n--- 생성 결과 ---")
    print("".join(itos[i] for i in out))

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "gpt_tiny.pt"))
