"""
[Step 3] 셀프 어텐션(Self-Attention)  -  NumPy 버전   <- 트랜스포머의 심장
=====================================================================
아이디어: 문장 속 각 단어가 "나는 어떤 단어를 얼마나 참고해야 할까?"를 계산한다.

  Q (Query, 질문)   : 내가 찾고 싶은 것
  K (Key,   열쇠)   : 나는 이런 정보를 가지고 있어요 (이름표)
  V (Value, 값)     : 실제로 전달할 내용

  Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V
"""
import numpy as np


def softmax(x, axis=-1):
    """수치적으로 안전한 softmax. 최댓값을 빼도 결과는 같지만 overflow를 막아 준다."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    """미래를 가리는 마스크. True인 자리(대각선 위쪽)가 '보면 안 되는' 칸이다."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V : (T, d_k)   T = 토큰 수, d_k = 한 헤드의 벡터 길이
    mask    : (T, T) bool, True인 칸은 -무한대로 만들어 softmax 후 0이 되게 한다.
    반환    : (출력 (T, d_k), 어텐션 가중치 (T, T))
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)            # (T, T) 모든 단어쌍의 '유사도 점수'
    if mask is not None:
        scores = np.where(mask, -1e9, scores)  # 못 보는 칸은 아주 작은 값
    weights = softmax(scores, axis=-1)         # 각 행의 합이 1이 되는 '참고 비율'
    return weights @ V, weights                # 비율대로 V를 섞는다


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T, d_model, d_k = 4, 8, 4
    x = rng.normal(size=(T, d_model))          # 토큰 4개짜리 입력

    # x에서 Q, K, V를 만드는 가중치 (학습으로 바뀌는 값들)
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) * 0.3 for _ in range(3))
    Q, K, V = x @ Wq, x @ Wk, x @ Wv

    out, w = scaled_dot_product_attention(Q, K, V)
    print("어텐션 가중치(각 행의 합=1):\n", w.round(2))
    print("각 행의 합:", w.sum(axis=1).round(3))
    print("출력 shape:", out.shape)

    # 디코더(글 생성)에서는 '미래 단어'를 훔쳐보면 안 되므로 causal mask 를 쓴다.
    out_c, w_c = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))
    print("\n마스크 적용 후(아래 삼각형만 남는다):\n", w_c.round(2))
    assert np.allclose(np.triu(w_c, k=1), 0)   # 대각선 위쪽은 전부 0
