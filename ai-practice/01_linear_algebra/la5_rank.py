"""
[실전선대] 5장 — §5.3~5.4 계수(rank): 속성·특수 행렬·연산 규칙·이동·tol 추정·확장·독립성

핵심: rank = 행렬이 담은 실질 정보 축의 개수(독립 열=행 최대 수 = 열공간 차원).
     r ≤ min{M,N}; 이동하면 보통 full rank(정보 거의 불변); 컴퓨터는 tol로 '추정'.
     확장(augmenting): rank 불변 ⇔ v∈C(A). 독립성: r=N 독립 / r<N 종속.
실행: python la5_rank.py
노트: 수학 원자 [2.7 행렬 계수(rank) — 정보 차원·최대계수·이동·확장] / 렌즈 [13]
"""
import numpy as np

rk = np.linalg.matrix_rank

# ── 각주 2: 눈 추정 6종 검산 — r(A)=1, r(B)=1, r(C)=2, r(D)=3, r(E)=1, r(F)=0 ──
A = np.array([[1.], [2.], [4.]])
B = np.array([[1., 3.], [2., 6.], [4., 12.]])          # 2열 = 3×1열 → 1
C = np.array([[1., 3.1], [2., 6.], [4., 12.]])         # 3.1이 종속을 깸 → 2
D = np.array([[1., 3., 2.], [6., 6., 1.], [4., 2., 0.]])
E = np.ones((3, 3))                                    # 아홉 칸, 정보는 한 열 → 1
F = np.zeros((3, 3))
assert [rk(M) for M in (A, B, C, D, E, F)] == [1, 1, 2, 3, 1, 0]

# ── 속성: 상한 r≤min{M,N} · 스칼라곱 불변(0 예외) ─────────────
tall = np.random.default_rng(0).standard_normal((5, 3))
assert rk(tall) == min(5, 3) == 3                      # randn → 사실상 full rank
assert rk(3.7 * D) == rk(D) and rk(0 * D) == 0

# ── 특수 행렬의 계수 (§5.3.1) ────────────────────────────────
assert rk(np.eye(4)) == 4                              # 단위 I_N → N
assert rk(np.diag([3., 0., 2., 0.])) == 2              # 대각 → 0 아닌 대각 원소 수
T_ok  = np.array([[1., 5.], [0., 2.]])                 # 대각에 0 없음 → full
T_bad = np.array([[0., 5.], [0., 2.]])                 # 대각에 0 → 축소
assert rk(T_ok) == 2 and rk(T_bad) == 1
v, w = np.array([4., 2., 3.]), np.array([3., 1., 1., 3., 1.])
assert rk(np.outer(v, w)) == 1                         # 외적 = 계수-1 (병목=1, 각주 4)

# ── 연산 규칙 (§5.3.2): 상한 부등식 ──────────────────────────
P = np.outer([1., 2.], [1., 0.])                       # rank 1
Q = np.outer([0., 1.], [0., 1.])                       # rank 1
assert rk(P + Q) == 2 <= rk(P) + rk(Q)                 # 덧셈: 개별보다 커질 수 있음
assert rk(P @ Q) <= min(rk(P), rk(Q))                  # 곱셈: 병목이 상한

# ── 이동된 행렬 (§5.3.3): rank 2 → 3, 정보는 거의 불변 ────────
M = np.array([[1., 3., 2.], [5., 7., 2.], [2., 2., 0.]])
assert np.allclose(M[:, 2], M[:, 1] - M[:, 0]) and rk(M) == 2   # 3열 = 2열 − 1열
Ms = M + 0.01 * np.eye(3)
assert rk(Ms) == 3                                     # 이동 → full rank(가역 획득)
rho = np.corrcoef(M.ravel(), Ms.ravel())[0, 1]
assert rho > 0.99997                                   # 책: ρ = 0.999972 (정보 보존)
assert rk(np.zeros((3, 3)) + np.eye(3)) == 3           # 0 + I = full rank
# 잡음 계수(그림 5-6): 수학적 rank-2 평면 + 잡음 → rank 3 (tol이 거를 대상)
noisy = M + 1e-9 * np.random.default_rng(1).standard_normal((3, 3))
assert rk(noisy, tol=1e-6) == 2 and rk(noisy, tol=1e-12) == 3   # rank는 tol 정책의 함수

# ── 확장 augmenting (§5.4.1): rank 비교로 소속 판별 ───────────
base = np.array([[1., 1.], [3., 2.], [1., 1.]])        # 3×2, rank 2 → C = R³ 속 평면
v_in  = 2 * base[:, 0] - 1 * base[:, 1]                # 열 결합 → 열공간 안
v_out = np.array([0., 0., 5.])                         # 평면 밖
assert rk(np.column_stack([base, v_in])) == rk(base) == 2       # 불변 → v ∈ C(A)
assert rk(np.column_stack([base, v_out])) == rk(base) + 1       # +1  → v ∉ C(A)
aug = np.column_stack([np.array([[4.,5.,6.],[0.,1.,2.],[9.,9.,4.]]), [1., 2., 3.]])
assert aug.shape == (3, 4)                             # 책 예제: M×N ⊔ M×1 → M×(N+1)

# ── 독립성 판별 (§5.4.2): r=N 독립 / r<N 종속 (책 'r<M'은 오기 의심) ──
ind = np.array([[1., 0.], [0., 1.], [1., 1.]])         # r=2=N → 독립
dep = np.array([[1., 2.], [2., 4.], [3., 6.]])         # r=1<N → 종속
assert rk(ind) == ind.shape[1] and rk(dep) < dep.shape[1]

if __name__ == "__main__":
    print("✅ §5.3~5.4 검증 통과 — 각주2 6종 · 특수행렬 · 상한 부등식 · 이동(ρ>0.99997) · tol 추정 · 확장 소속 · 독립성")
