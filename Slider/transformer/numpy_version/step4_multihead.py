"""
[Step 4] 멀티헤드 어텐션(Multi-Head Attention)  -  NumPy 버전
=====================================================================
한 번만 보지 말고, 여러 '관점(head)'으로 동시에 보자.
  - 벡터(d_model)를 h개 조각으로 나누어 각자 어텐션을 한다.
  - 결과를 다시 이어 붙이고(concat) 출력 가중치 Wo 로 섞는다.
"""
import numpy as np
from step3_attention import scaled_dot_product_attention, causal_mask


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """
    x : (T, d_model)
    W*: (d_model, d_model)   -> 모든 헤드의 가중치를 한 행렬에 모아 둔다.
    """
    T, d_model = x.shape
    d_k = d_model // n_heads                    # 헤드 하나가 맡는 길이

    Q, K, V = x @ Wq, x @ Wk, x @ Wv            # 각 (T, d_model)
    mask = causal_mask(T) if causal else None

    heads = []
    for h in range(n_heads):
        s = slice(h * d_k, (h + 1) * d_k)       # h번째 헤드의 열 범위
        out_h, _ = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s], mask)
        heads.append(out_h)                     # 각 (T, d_k)

    concat = np.concatenate(heads, axis=-1)     # (T, d_model) 로 다시 이어 붙임
    return concat @ Wo                          # 헤드들의 결과를 섞어 최종 출력


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    T, d_model, n_heads = 5, 16, 4
    x = rng.normal(size=(T, d_model))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) * 0.3 for _ in range(4))

    y = multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads)
    print("입력 shape:", x.shape, "-> 출력 shape:", y.shape)  # 같아야 한다!

    # 헤드 수가 1이면 Step 3 의 단일 어텐션과 똑같아야 한다 (검산)
    y1 = multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads=1)
    ref, _ = scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, causal_mask(T))
    assert np.allclose(y1, ref @ Wo)
    print("검산 통과: 헤드 1개 == 단일 어텐션")
