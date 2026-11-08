"""
[Step 1] 토큰 임베딩 + 위치 인코딩  -  PyTorch 버전
=====================================================================
NumPy 버전(Step 1, 2)과 같은 일을 PyTorch 부품으로 한다.

  - nn.Embedding : '번호 -> 벡터' 조회 표 (학습 가능한 파라미터)
  - 위치 인코딩  : 학습하지 않는 상수 표 -> register_buffer 로 모델에 등록
                   (모델과 함께 GPU로 이동/저장되지만 optimizer는 건드리지 않는다)
  - 배치(batch)  : 문장을 여러 개 한꺼번에 처리하므로 shape 이 (B, T, d_model) 이 된다.
"""
import math
import torch
import torch.nn as nn


class TokenAndPosition(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model)      # (V, d_model) 표

        # --- 사인/코사인 위치 인코딩 표 (max_len, d_model) ---
        pos = torch.arange(max_len).unsqueeze(1)                           # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)                                 # 짝수 열
        pe[:, 1::2] = torch.cos(pos * div)                                 # 홀수 열
        self.register_buffer("pe", pe)                                     # 상수로 등록

    def forward(self, idx):                     # idx: (B, T) 정수 텐서
        T = idx.shape[1]
        x = self.tok(idx) * math.sqrt(self.d_model)   # (B, T, d_model)
        return x + self.pe[:T]                        # (T, d_model) 이 B 방향으로 자동 브로드캐스트


if __name__ == "__main__":
    layer = TokenAndPosition(vocab_size=30, d_model=16, max_len=64)
    idx = torch.randint(0, 30, (2, 10))          # 문장 2개, 각 10토큰
    out = layer(idx)
    print("입력 idx shape :", tuple(idx.shape))
    print("출력 shape     :", tuple(out.shape), " = (배치 B, 토큰 T, 벡터 d_model)")
    print("학습되는 파라미터:", [n for n, _ in layer.named_parameters()])   # pe 는 없다!
