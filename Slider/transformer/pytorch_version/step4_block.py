"""
[Step 4] 트랜스포머 블록  -  PyTorch 버전
=====================================================================
블록 = 멀티헤드 어텐션 + 피드포워드, 각각에 LayerNorm 과 잔차연결(residual)을 붙인다.

    x = x + Attention( LayerNorm(x) )   # 단어들끼리 정보를 주고받는다
    x = x + FeedForward( LayerNorm(x) ) # 각 단어가 혼자 정보를 가공한다

  - 잔차연결 : 깊게 쌓아도 기울기가 잘 흐르게 하는 '지름길'
  - LayerNorm: 값의 크기를 안정시켜 학습을 쉽게 한다
  - Dropout  : 학습 중 일부를 무작위로 꺼서 과적합을 막는다
"""
import torch
import torch.nn as nn
from step3_multihead import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # 4배로 넓혔다가
            nn.GELU(),                         # 부드러운 비선형 함수 (ReLU의 친척)
            nn.Linear(4 * d_model, d_model),   # 원래 크기로 되돌린다
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    blk = Block(d_model=32, n_heads=4, max_len=64)
    x = torch.randn(2, 10, 32)
    print("블록 입력", tuple(x.shape), "-> 출력", tuple(blk(x).shape))   # shape 이 같아야 쌓을 수 있다
    print("블록 하나의 파라미터 수:", sum(p.numel() for p in blk.parameters()))
