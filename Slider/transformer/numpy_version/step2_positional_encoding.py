"""
[Step 2] 위치 인코딩(Positional Encoding)  -  NumPy 버전
=====================================================================
문제: 어텐션은 단어를 '한꺼번에' 보기 때문에 순서를 모른다.
      "개가 사람을 물었다" 와 "사람이 개를 물었다" 를 구분하려면
      각 토큰에 '내가 몇 번째인지' 표시를 더해 주어야 한다.

해결(원 논문 방식): 사인/코사인 파형으로 위치마다 고유한 벡터를 만든다.

    PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
"""
import numpy as np


def positional_encoding(max_len, d_model):
    """(max_len, d_model) 크기의 위치 인코딩 표를 만든다. (d_model은 짝수)"""
    pos = np.arange(max_len)[:, None]                  # (max_len, 1)  위치 0,1,2,...
    i = np.arange(0, d_model, 2)[None, :]              # (1, d_model/2) 짝수 차원 번호
    angle = pos / np.power(10000.0, i / d_model)       # (max_len, d_model/2) 각도
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)                        # 짝수 열: sin
    pe[:, 1::2] = np.cos(angle)                        # 홀수 열: cos
    return pe


if __name__ == "__main__":
    pe = positional_encoding(max_len=16, d_model=8)
    print("PE shape:", pe.shape)
    print(pe[:4].round(2))

    # 위치 인코딩을 단어 임베딩에 '더한다' (이어붙이는 것이 아니다!)
    from step1_embedding import x          # (5, 8) "hello"의 임베딩
    x_in = x + pe[: x.shape[0]]            # 같은 shape이므로 원소별 덧셈
    print("임베딩 + 위치 인코딩 shape:", x_in.shape)

    # 가까운 위치일수록 벡터가 비슷하다 (내적이 크다)는 것을 확인
    pe = positional_encoding(64, 32)
    for gap in [0, 1, 4, 16, 48]:
        print(f"위치 0 과 위치 {gap:2d} 의 내적: {pe[0] @ pe[gap]:7.3f}")
