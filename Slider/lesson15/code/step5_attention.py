"""Step 5. 어텐션 가중치 들여다보기 — 식 15.5  Attention(i,j) = softmax_j( q_i·k_j / √d_k )
손으로 설계한 아주 작은 어텐션으로 (1) 가중치 행렬을 직접 계산·시각화하고,
(2) '어텐션 가중치가 크다 ≠ 그 토큰이 결과에 중요하다' 를 숫자로 확인한다. 마지막에 PyTorch 와 같은 값인지 대조.
실행: python step5_attention.py
"""
import numpy as np

TOK = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
# 토큰 특징 6칸: [동물인가, 사물인가, 대명사인가, 서술어인가, 문장 첫 토큰 표시, 상수1]
E = np.array([[0,0,0,0,1,1],[1,0,0,0,0,1],[0,0,0,1,0,1],[0,0,0,1,0,1],[0,0,0,0,0,1],[0,1,0,0,0,1],
              [0,0,0,0,0,1],[0,0,1,0,0,1],[0,0,0,1,0,1],[0,0,0,0,0,1],[0,0,0,1,0,1]], float)
d = E.shape[1]
# 가중치 행렬 (q = Wq·e, k = Wk·e, v = Wv·e):
Wq = np.zeros((d, d)); Wq[0, 2] = 4.4; Wq[0, 3] = 3.0; Wq[1, 2] = 2.0; Wq[4, 5] = 2.0   # 대명사는 '동물'(강)·'사물'(약)을 찾고, 서술어는 '동물'을 찾는다. 모두 첫 토큰을 약간 찾는다
Wk = np.zeros((d, d)); Wk[0, 0] = 2.0; Wk[1, 1] = 2.0; Wk[4, 4] = 3.0                  # 키: 동물/사물 표지, 첫 토큰 표지
Wv = np.zeros((d, d)); Wv[:4, :4] = np.eye(4)                                            # 값: 의미 특징 4칸만 전달. 첫 토큰은 의미가 없어 값이 0 벡터

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def attention(E, mask=None):
    """어텐션 행렬 A (T,T) 와 출력 O (T,d).  mask[j]=False 인 토큰은 '없는 셈' 친다."""
    Q, K, V = E @ Wq.T, E @ Wk.T, E @ Wv.T
    S = Q @ K.T / np.sqrt(d)                              # 점수 q_i·k_j / √d_k
    if mask is not None: S = np.where(mask[None, :], S, -np.inf)
    A = softmax(S)                                        # 각 행의 합이 1 (i가 누구를 얼마나 볼지)
    return A, A @ V

def contribution(i):
    """값 가려 보기(value ablation): 토큰 j 의 값 벡터만 0으로 만들면 출력 O_i 가 얼마나 변하는가.
    O_i = Σ_j A_ij·v_j 이므로 변화량 = ‖A_ij · v_j‖ : '얼마나 봤나(A)' 뿐 아니라 '무엇을 가져왔나(v)'도 곱해진다."""
    A, _ = attention(E); V = E @ Wv.T
    return np.linalg.norm(A[i][:, None] * V, axis=1)

def removal(i):
    """참고: 토큰 j 를 문장에서 아예 빼고(마스크) 다시 계산. 나머지 토큰의 가중치가 다시 커지므로 값 가리기와 다르다."""
    A, O = attention(E); out = []
    for j in range(len(TOK)):
        m = np.ones(len(TOK), bool); m[j] = False
        out.append(np.linalg.norm(attention(E, m)[1][i] - O[i]))
    return np.array(out)

def ranks(a): r = np.empty(len(a)); r[np.argsort(-a)] = np.arange(len(a)); return r

if __name__ == "__main__":
    A, O = attention(E); i = TOK.index("it")
    assert np.allclose(A.sum(1), 1), "각 행의 합은 1"
    print('"it" 가 보는 곳 (어텐션 가중치):')
    for t, a in sorted(zip(TOK, A[i]), key=lambda p: -p[1])[:4]: print(f"   {t:8s} {a:.2f}")
    print("→ 대명사 'it'이 'animal'을 가장 크게 본다 (공참조 해결)")
    con, rem = contribution(i), removal(i)
    print('\n"it" 의 출력에 대한 토큰별 기여: 어텐션 가중치 vs 값 가려 보기 vs 문장에서 빼 보기')
    for j in np.argsort(-A[i])[:4]: print(f"   {TOK[j]:8s} 어텐션 {A[i, j]:.2f} | 값 가리면 변화 {con[j]:.2f} | 토큰 빼면 변화 {rem[j]:.2f}")
    print("→ 첫 토큰 'The'는 어텐션이 %.2f 로 둘째로 크지만, 값(v)이 0 벡터라 출력에는 기여가 %.2f (어텐션 싱크)" % (A[i, 0], con[0]))
    assert A[i].argmax() == 1 and A[i, 0] > 0.1 and con[0] < 1e-12
    corr = np.corrcoef(ranks(A[i]), ranks(con))[0, 1]
    print("어텐션 순위와 기여 순위의 상관(순위상관) = %.2f  (1이면 완전히 같은 순서)" % corr)
    loo = con
    np.save("_attn.npy", dict(A=A, loo=loo, i=i, tok=TOK), allow_pickle=True)

    # ---------- PyTorch 대조: nn.MultiheadAttention 에 같은 가중치를 넣으면 같은 A, 같은 O ----------
    import torch, torch.nn as nn
    mha = nn.MultiheadAttention(d, 1, bias=False, batch_first=True).double()
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.tensor(np.vstack([Wq, Wk, Wv])))       # [Wq;Wk;Wv] 를 쌓은 모양
        mha.out_proj.weight.copy_(torch.eye(d, dtype=torch.double))            # 출력 투영은 항등
        o, a = mha(*(torch.tensor(E)[None],) * 3, need_weights=True)           # a: (1,T,T) 평균된 헤드 가중치
    print("\nPyTorch MultiheadAttention 과 일치: 가중치 %s, 출력 %s" % (np.allclose(a[0].numpy(), A), np.allclose(o[0].numpy(), O)))
