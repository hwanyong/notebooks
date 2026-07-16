"""실전선대 §2.2~2.3 — 선형 가중 결합 + 선형 독립 판별(rank)."""
import numpy as np

# 선형 가중 결합 (식 2-2)
l1, l2, l3 = 1, 2, -3
v1, v2, v3 = np.array([4, 5, 1]), np.array([-4, 0, -4]), np.array([1, 3, 2])
w = l1*v1 + l2*v2 + l3*v3
print("w =", w)                              # [ -7  -4 -13]

# 선형 독립 판별 = rank (열로 쌓은 행렬)
indep = np.array([[1, 3], [2, 7]]).T          # 열 = 벡터
dep   = np.array([[1, 3], [2, 6]]).T          # s2 = 2·s1 → 종속
print("indep rank =", np.linalg.matrix_rank(indep))   # 2
print("dep   rank =", np.linalg.matrix_rank(dep))     # 1

# 4×4 종속 예 (처음 세 벡터 합 = 네 번째의 2배)
T = np.array([[8,-4,14,6],[4,6,0,3],[14,2,4,7],[13,2,9,8]]).T
print("T rank =", np.linalg.matrix_rank(T), "(<4 → 종속)")

# 영벡터 포함 → 항상 종속
Z = np.array([[1,0],[0,0]]).T
print("zero-included rank =", np.linalg.matrix_rank(Z), "(<2 → 종속)")
