"""
[실전선대] 5장 — 행렬 파트2 (§5.1 행렬 노름: 프로베니우스·ℓ2·행렬 거리)

핵심: 행렬 노름은 하나가 아니다(원소별 vs 유도). 프로베니우스 = √(전 원소 제곱합)
     = 행렬을 펴서 벡터 ℓ2. 행렬 거리 = ‖A−B‖F. 정규화 = 노름을 비용 함수에.
실행: python la5_norms.py
노트: 수학 원자 [2.5 행렬 노름 — 프로베니우스·ℓ2·행렬 거리] / 렌즈 [11]
"""
import numpy as np

# ── 식 5-1: 프로베니우스 노름 — 삼중 일치 검증 ──────────────
A = np.array([[1., 2.], [3., 4.]])
manual = np.sqrt((A**2).sum())                      # 식 5-1 직접
assert np.isclose(manual, np.linalg.norm(A, 'fro'))         # NumPy 'fro'
assert np.isclose(manual, np.linalg.norm(A.ravel(), 2))     # "펴서 벡터 ℓ2"와 동일
print("‖A‖F =", round(manual, 4))                   # 5.4772

# ── 원소별 p-노름 (p=2 가 프로베니우스) ────────────────────
p = 3
lp = (np.abs(A)**p).sum()**(1/p)
assert np.isclose((np.abs(A)**2).sum()**(1/2), manual)      # p=2 ⇒ 프로베니우스
print("‖A‖p=3 =", round(lp, 4))                     # 4.6416

# ── 행렬 거리: A → C=A−B 치환만 ────────────────────────────
B = np.array([[1., 2.], [3., 5.]])
assert np.isclose(np.linalg.norm(A - B, 'fro'), 1.0)        # 원소 하나 1 차이
assert np.linalg.norm(A - A, 'fro') == 0.0                  # 동일 행렬 = 거리 0

# ── 정규화 감각: 노름 = 가중치 크기 벌금 ────────────────────
W_small, W_big = np.full((3, 3), 0.1), np.full((3, 3), 10.0)
assert np.linalg.norm(W_big, 'fro') > np.linalg.norm(W_small, 'fro') * 50
# L2 페널티 λ‖W‖² 가 W_big 에 훨씬 큰 비용을 매김 → 매개변수 비대 방지(릿지)

# ── §5.1.1 대각합(trace): 정방 전용 · ‖A‖F = √tr(AᵀA) ──────
M1 = np.array([[4, 5, 6], [0, 1, 4], [9, 9, 9]])
M2 = np.array([[0, 0, 0], [0, 8, 0], [1, 2, 6]])
assert np.trace(M1) == np.trace(M2) == 14          # 책 예제: 둘 다 14 (대각 밖은 무시)
assert np.isclose(np.sqrt(np.trace(A.T @ A)), np.linalg.norm(A, 'fro'))  # √tr(AᵀA) = ‖A‖F
# 이유: (AᵀA)의 (i,i) = A의 i열 자기 내적(제곱합) → 대각합 = 전 원소 제곱합
col_sq = [(A[:, j]**2).sum() for j in range(A.shape[1])]
assert np.isclose(sum(col_sq), np.trace(A.T @ A))
# ⚠️ 오독 방지: "노름 = A의 대각 구하기"가 아니다 — A 자신의 대각만이면 √17 ≠ √30
assert not np.isclose(np.sqrt((np.diag(A)**2).sum()), np.linalg.norm(A, 'fro'))
# 비정방(3×2)도 ‖·‖F 존재; tr(R)은 정의 불가 — 우회로는 항상 정방인 RᵀR의 대각
R = np.array([[1., 2.], [3., 4.], [5., 6.]])
assert np.isclose(np.sqrt(np.trace(R.T @ R)), np.linalg.norm(R, 'fro'))
# 두 우회로(P03): 열로 묶든(RᵀR, 2×2) 행으로 묶든(RRᵀ, 3×3) tr = 전 원소 제곱합 = 91
assert np.trace(R.T @ R) == np.trace(R @ R.T) == 91
assert np.trace(R) == 5.0  # ⚠️ 비정방 np.trace: 에러 없이 1+4 — 수학적 대각합 아님(함정)

# ── 왜 '대각' = 주대각선(i=j)만인가: 기저 불변 vs 반대각선 ────
# ① tr 은 기저 변환에 불변: tr(P⁻¹AP) = tr(A)
rng = np.random.default_rng(0)
P = rng.standard_normal((3, 3)) + 3*np.eye(3)       # 가역 보장용 대각 강화
assert np.isclose(np.trace(np.linalg.inv(P) @ M1 @ P), np.trace(M1))
# ② 반대각합(i+j=N+1)은 불변 아님 — 책 두 행렬조차 16 vs 9
anti = lambda X: np.fliplr(X).trace()               # 좌우 뒤집으면 반대각→주대각
assert anti(M1) == 16 and anti(M2) == 9
# ③ 축 1↔2 재번호(치환 유사변환): 주대각합 보존, 반대각합 깨짐
Q = np.eye(3)[[1, 0, 2]]                            # 치환 행렬 (Q⁻¹ = Qᵀ)
M1p = Q.T @ M1 @ Q
assert np.trace(M1p) == np.trace(M1)                # 주대각: 원소가 자리만 바꿔 잔류
assert anti(M1p) != anti(M1)                        # 반대각: 다른 원소들로 교체됨

# ── 이동량 λ 기준 회수(2.1 예고): 노름 비례 이동 ─────────────
M = np.array([[4., 5., 1.], [0., 1., 11.], [4., 9., 7.]])
lam = 0.01 * np.linalg.norm(M, 'fro')               # 행렬 스케일에 비례하는 "적은 양"
shifted = M + lam * np.eye(3)
assert np.allclose(np.diag(shifted) - np.diag(M), lam)      # 대각만 +λ

if __name__ == "__main__":
    print("✅ §5.1 검증 통과 — 식 5-1 = 'fro' = 벡터화 ℓ2 · p-노름 · 거리 · λ=노름 비례 이동")
