"""
[Step 2] 셀프 어텐션 (헤드 1개)  -  PyTorch 버전
=====================================================================
  Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V

  - Q, K, V 를 만드는 가중치는 nn.Linear 가 대신 관리해 준다.
  - 미래를 가리는 마스크(tril: 아래쪽 삼각형)는 buffer 로 한 번만 만든다.
  - 기울기 계산은 PyTorch 가 자동으로 해 주므로 backward 코드는 필요 없다.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, max_len):
        super().__init__()
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        # tril: 1이 아래쪽 삼각형에 있는 행렬 -> 1인 자리는 '볼 수 있는' 칸
        self.register_buffer("causal", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):                                   # x: (B, T, d_model)
        B, T, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)           # 각 (B, T, d_k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])   # (B, T, T)
        scores = scores.masked_fill(self.causal[:T, :T] == 0, float("-inf"))  # 미래 가리기
        w = F.softmax(scores, dim=-1)                       # 각 행 합 = 1
        return w @ v, w                                     # 출력 (B,T,d_k), 가중치 (B,T,T)


if __name__ == "__main__":
    torch.manual_seed(0)
    attn = SelfAttention(d_model=16, d_k=8, max_len=32)
    x = torch.randn(2, 5, 16)
    out, w = attn(x)
    print("출력 shape:", tuple(out.shape), " 가중치 shape:", tuple(w.shape))
    print("첫 번째 문장의 어텐션 가중치:\n", w[0].detach().numpy().round(2))
    assert torch.allclose(w.sum(-1), torch.ones(2, 5))      # 행의 합은 1
    assert torch.all(torch.triu(w, diagonal=1) == 0)        # 미래 칸은 0
    print("검산 통과")
