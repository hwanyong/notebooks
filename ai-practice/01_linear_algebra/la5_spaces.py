"""
[실전선대] 5장 — §5.2~5.2.1 행렬 공간: 열공간 C(A)와 소속 판별

핵심: C(A) = 열들의 모든 선형 가중 결합(=span). v∈C(A) ⇔ Ax=v 해 존재.
     N개 열 ≠ N차원 — 차원 = 독립 열 수(rank 예고).
실행: python la5_spaces.py
노트: 수학 원자 [2.6 행렬 공간 — 열공간·행공간·영공간] / 렌즈 [12]
"""
import numpy as np

# ── 예 1: 열 1개 — 소속 판별 3례 (책 예제) ──────────────────
A = np.array([[1.], [3.]])
for v, expect in [([1, 3], True), ([-2, -6], True), ([1, 4], False)]:
    v = np.array(v, float)
    lam = v[0] / A[0, 0]                        # 첫 성분으로 λ 후보
    member = np.allclose(A.ravel() * lam, v)
    assert member == expect
    print(f"{v.tolist()} ∈ C(A)? {member}")

# ── 예 2: 독립 2열 → R² 전체 — 책 수치 [-4,3] = 11·c1 − 15·c2 ─
c1, c2 = np.array([1., 3.]), np.array([1., 2.])
assert np.allclose(11*c1 - 15*c2, [-4, 3])      # 책이 예고한 가중치 검증

# ── 예 3: 종속 2열 → 여전히 1D (차원 ≠ 열 수) ────────────────
D = np.array([[1., 2.], [3., 6.]])              # col2 = 2·col1
assert np.linalg.matrix_rank(D) == 1            # 독립 열 수 = 1 (rank 예고)

# ── 예 4: R³ 속 독립 2열 → 2D 평면 ("얇은 종이") ─────────────
E = np.array([[3., 0.], [5., 2.], [1., 2.]])
assert np.linalg.matrix_rank(E) == 2            # 2차원 열공간 ⊂ R³
# 소속 판별 일반형: Ax=v 최소제곱 잔차 0 ⇔ v ∈ C(A)
v_in  = 2*E[:, 0] + 1*E[:, 1]                   # 열공간 안 점
x, res, *_ = np.linalg.lstsq(E, v_in, rcond=None)
assert np.allclose(E @ x, v_in)                 # 잔차 0 → 소속
v_out = v_in + np.cross(E[:, 0], E[:, 1])       # 법선 방향으로 밀어낸 점(평면 밖)
x2, *_ = np.linalg.lstsq(E, v_out, rcond=None)
assert not np.allclose(E @ x2, v_out)           # 재구성 불가 → 비소속

# ── §5.2.2 행공간: R(A) = C(Aᵀ) — 전치 재사용 ────────────────
assert np.linalg.matrix_rank(E.T) == np.linalg.matrix_rank(E)   # 행공간 차원 = 열공간 차원(rank 예고)
S = np.array([[2., 1.], [1., 3.]])              # 대칭 행렬 → R(S) = C(S)
assert np.allclose(S, S.T)

# ── §5.2.3 영공간: N(A) — 책 예제 2개 ───────────────────────
A1 = np.array([[1., -1.], [-2., 2.]])           # 종속 열(rank 1) → N(A) 존재
B1 = np.array([[1., -1.], [-2., 3.]])           # 독립 열(det=1, 가역) → N(B) = {}
try:
    from scipy.linalg import null_space         # 책 코드 그대로
    ns, ns_b = null_space(A1), null_space(B1)
except ImportError:                             # scipy 없으면 SVD로 동일 계산
    def null_space(M, tol=1e-10):
        _, s, vh = np.linalg.svd(M)
        return vh[s.size - np.sum(s > tol):].T if np.sum(s > tol) < M.shape[1] else np.empty((M.shape[1], 0))
    ns, ns_b = null_space(A1), null_space(B1)
assert ns.shape[1] == 1 and ns_b.shape[1] == 0            # A1은 1D 영공간, B1은 빈 집합
assert np.allclose(np.abs(ns.ravel()), np.sqrt(1/2))      # 0.70710678 = 단위벡터(√½)
assert np.isclose(np.linalg.norm(ns), 1.0)                # 노름 1 — 단위 기저 벡터 관행
assert np.allclose(A1 @ ns, 0)                            # Ay = 0
assert np.isclose(A1[0] @ ns.ravel(), 0) and np.isclose(A1[1] @ ns.ravel(), 0)  # 영공간⊥각 행
assert np.allclose(A1 @ np.array([7.34, 7.34]), 0)        # 저자의 [7.34, 7.34]도 성립(λ 자유 변수)
# 독립 열 ⇔ 빈 영공간 (rank로 재확인)
assert np.linalg.matrix_rank(A1) == 1 and np.linalg.matrix_rank(B1) == 2

if __name__ == "__main__":
    print("✅ §5.2 검증 통과 — 소속 3례 · [-4,3]=11c1−15c2 · 종속=1D · R³ 평면 · R(A)=C(Aᵀ) · N(A) 단위벡터/빈집합/직교")
