"""Step 2. 레이블 일관성: 어떤 변환이 '정답을 바꾸는가'를 숫자로 확인한다.
'각 숫자의 평균 이미지'와 가장 닮은 쪽을 정답으로 보는 아주 단순한 판정기를 쓴다.
실행: python step2_label.py
"""
import numpy as np
from step1_transforms import load, rotate, flip_h, shift

X, y = load()
means = np.stack([X[y == k].mean(0) for k in range(10)])          # 클래스별 평균 이미지
def guess(img): return int(np.argmin(((means - img) ** 2).sum((1, 2))))   # 가장 가까운 평균

def keep_rate(T, digit=None):
    """변환 후에도 판정이 원래 정답과 같은 비율"""
    idx = np.where(y == digit)[0] if digit is not None else np.arange(len(y))
    return np.mean([guess(T(X[i])) == y[i] for i in idx])

tests = {"변환 없음": lambda a: a, "이동 1칸": lambda a: shift(a, 0, 1), "회전 10도": lambda a: rotate(a, 10),
         "회전 15도": lambda a: rotate(a, 15), "회전 45도": lambda a: rotate(a, 45), "회전 90도": lambda a: rotate(a, 90),
         "좌우 뒤집기": flip_h, "회전 180도": lambda a: rotate(a, 180)}
if __name__ == "__main__":
    for k, T in tests.items(): print(f"{k:10s} 정답 유지율 {keep_rate(T):.2f}")
    six = np.where(y == 6)[0]
    after = [guess(rotate(X[i], 180)) for i in six]
    print("6을 180도 돌리면 판정:", {int(k): after.count(k) for k in set(after)})
    print("숫자 2를 좌우로 뒤집으면 판정:", {int(k): [guess(flip_h(X[i])) for i in np.where(y == 2)[0]].count(k) for k in range(10)})
