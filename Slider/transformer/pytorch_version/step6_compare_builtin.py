"""
[Step 6] 내가 만든 부품 vs PyTorch 내장 부품  -  검증
=====================================================================
직접 만든 MultiHeadAttention 의 가중치를 PyTorch 공식 nn.MultiheadAttention 에
그대로 복사해 넣으면 출력이 똑같아야 한다.  똑같다 = 내가 제대로 이해하고 구현했다는 증거!

실무에서는 검증된 내장 부품(nn.TransformerEncoderLayer 등)을 쓰는 경우가 많다.
"""
import torch
import torch.nn as nn
from step3_multihead import MultiHeadAttention

torch.manual_seed(0)
D, H, T = 32, 4, 10
mine = MultiHeadAttention(D, H, max_len=64).eval()
ref = nn.MultiheadAttention(D, H, batch_first=True).eval()

with torch.no_grad():
    # 공식 구현은 Q,K,V 가중치를 세로로 붙여 하나의 행렬(in_proj_weight)에 저장한다
    ref.in_proj_weight.copy_(torch.cat([mine.q.weight, mine.k.weight, mine.v.weight], dim=0))
    ref.in_proj_bias.zero_()                       # 우리는 q,k,v 에 bias 가 없다
    ref.out_proj.weight.copy_(mine.o.weight)
    ref.out_proj.bias.copy_(mine.o.bias)

    x = torch.randn(2, T, D)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True = 보면 안 되는 칸
    y_ref, _ = ref(x, x, x, attn_mask=mask)
    y_mine = mine(x)

print("최대 차이:", (y_ref - y_mine).abs().max().item())
assert torch.allclose(y_ref, y_mine, atol=1e-5)
print("검증 통과: 내가 만든 멀티헤드 어텐션 == nn.MultiheadAttention")

# 내장 트랜스포머 블록 한 줄 사용 예
layer = nn.TransformerEncoderLayer(d_model=D, nhead=H, dim_feedforward=4 * D,
                                   batch_first=True, norm_first=True)
print("내장 TransformerEncoderLayer 출력 shape:", tuple(layer(x, src_mask=mask).shape))
