"""
[Step 1] 토큰화(Tokenization)와 임베딩(Embedding)  -  NumPy 버전
=====================================================================
목표: 글자(문자열)를 컴퓨터가 계산할 수 있는 '숫자 벡터'로 바꾼다.

흐름:  문자열 --(토큰화)--> 정수 번호(id) --(임베딩 표 조회)--> 벡터
"""
import numpy as np

# ------------------------------------------------------------------
# 1) 토큰화: 글자 하나하나를 토큰으로 쓰는 가장 단순한 방식(문자 단위)
# ------------------------------------------------------------------
text = "hello transformer"

# set(text)   : 중복을 제거한 글자 집합
# sorted(...) : 항상 같은 순서가 되도록 정렬 -> 글자마다 고정된 번호를 줄 수 있다
chars = sorted(set(text))
vocab_size = len(chars)                       # 어휘 크기 V (서로 다른 토큰의 개수)

stoi = {ch: i for i, ch in enumerate(chars)}  # 글자 -> 번호 (string to int)
itos = {i: ch for ch, i in stoi.items()}      # 번호 -> 글자 (int to string)


def encode(s):
    """문자열을 정수 번호 리스트로 바꾼다."""
    return [stoi[c] for c in s]


def decode(ids):
    """정수 번호 리스트를 다시 문자열로 되돌린다."""
    return "".join(itos[i] for i in ids)


# ------------------------------------------------------------------
# 2) 임베딩 표(Embedding table): 크기 (V, d_model)인 숫자 표
#    i번째 행 = i번 토큰을 나타내는 길이 d_model짜리 벡터
#    처음에는 랜덤이지만, 학습을 하면서 '의미 있는 값'으로 바뀐다.
# ------------------------------------------------------------------
d_model = 8                                   # 벡터 길이 (모델의 폭)
rng = np.random.default_rng(0)                # 시드를 고정하면 매번 같은 결과
E = rng.normal(0, 1, size=(vocab_size, d_model))

ids = np.array(encode("hello"))               # 예) [3, 2, 4, 4, 5]
x = E[ids]                                    # 행 번호로 조회 -> shape (T, d_model)

# ------------------------------------------------------------------
# 3) 같은 결과를 'one-hot 벡터 x 표' 행렬곱으로도 얻을 수 있다.
#    (조회는 사실 행렬곱의 빠른 버전이라는 것을 확인)
# ------------------------------------------------------------------
one_hot = np.eye(vocab_size)[ids]             # shape (T, V)
assert np.allclose(one_hot @ E, x)

if __name__ == "__main__":
    print("어휘(vocab):", chars)
    print("vocab_size :", vocab_size)
    print("encode('hello') ->", encode("hello"))
    print("decode(...)     ->", decode(encode("hello")))
    print("임베딩 표 E의 shape :", E.shape)
    print("입력 x의 shape      :", x.shape, " (토큰 수 T, 벡터 길이 d_model)")
    print(x.round(2))
