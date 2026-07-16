import numpy as np
from numpy.linalg import inv, det
# 연립방정식: 4x+2y+4z=44, 5x+3y+7z=56, 9x+3y+6z=72  →  AX=B,  X=A⁻¹B
A = np.array([[4,2,4],[5,3,7],[9,3,6]], dtype=float)
B = np.array([44,56,72], dtype=float)
print("det(A) =", round(det(A),3), " (≠0 → 가역)")
Ainv = inv(A)
print("A⁻¹ =\n", np.round(Ainv,4))
print("A⁻¹A =\n", np.round(Ainv @ A))         # 항등 I
X = Ainv @ B
print("X = A⁻¹B =", np.round(X,6))             # [2, 34, -8]
assert np.allclose(X, [2,34,-8]), X
# 역행렬 = 변환 되돌리기: A로 변형한 점을 A⁻¹로 복원
v = np.array([1,2,3], dtype=float)
print("복원 검증 A⁻¹(Av) =", np.round(Ainv @ (A @ v)))   # == v
print("OK")
