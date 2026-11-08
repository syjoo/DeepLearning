"""
[Step 3] 멀티헤드 어텐션  -  PyTorch 버전
=====================================================================
NumPy 버전은 for 문으로 헤드를 하나씩 돌렸다.
실전 코드는 헤드를 '새로운 축'으로 만들어 한 번에(병렬로) 계산한다.

  (B, T, D) --view--> (B, T, h, dk) --transpose--> (B, h, T, dk)
  이제 h 가 배치처럼 취급되어 모든 헤드가 동시에 계산된다.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 은 n_heads 로 나누어떨어져야 한다"
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)                  # 헤드를 섞는 출력 가중치
        self.drop = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):                                      # (B, T, D)
        B, T, D = x.shape
        # 1) 선형변환 후 헤드 단위로 쪼갠다 -> (B, h, T, dk)
        split = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))

        # 2) 모든 헤드에서 동시에 어텐션 계산 -> (B, h, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        scores = scores.masked_fill(self.causal[:T, :T] == 0, float("-inf"))
        w = self.drop(F.softmax(scores, dim=-1))

        # 3) 헤드들을 다시 이어 붙이고 (B, T, D), 출력 가중치로 섞는다
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(ctx)


if __name__ == "__main__":
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, n_heads=4, max_len=64)
    x = torch.randn(2, 10, 32)
    y = mha(x)
    print("입력", tuple(x.shape), "-> 출력", tuple(y.shape))

    # PyTorch 내장 함수(F.scaled_dot_product_attention)와 결과가 같은지 검산
    B, T, D = x.shape
    split = lambda t: t.view(B, T, 4, 8).transpose(1, 2)
    q, k, v = split(mha.q(x)), split(mha.k(x)), split(mha.v(x))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = mha.o(ref.transpose(1, 2).reshape(B, T, D))
    assert torch.allclose(y, ref, atol=1e-5)
    print("검산 통과: 직접 만든 어텐션 == 내장 scaled_dot_product_attention")
