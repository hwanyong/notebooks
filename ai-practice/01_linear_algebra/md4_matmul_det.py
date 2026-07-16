"""필수수학 §4.3 행렬곱(변환 합성·비교환) + §4.4 행렬식."""
import numpy as np
from numpy.linalg import det

# §4.3 행렬곱 = 변환 합성 (예제 4-11/4-12: 순서가 결과를 바꿈)
T1 = np.array([[0,-1],[1,0]])   # 회전
T2 = np.array([[1,1],[0,1]])    # 전단
v = np.array([1,2])
print("T2·T1 적용:", (T2 @ T1) @ v)     # [-1  1]
print("T1·T2 적용:", (T1 @ T2) @ v)     # [-2  3]  → 순서 다르면 결과 다름(비교환)
print("교환 안 됨:", not np.array_equal(T2@T1, T1@T2))   # True

# §4.4 행렬식 = 면적 배율 / 방향 / 선형종속
def basis(i_hat, j_hat):           # 열로 세운 변환 행렬
    return np.array([i_hat, j_hat]).T
print("확대 det:", round(det(basis([3,0],[0,2]))))        # 6   (면적 6배)
print("전단 det:", round(det(basis([1,0],[1,1]))))        # 1   (면적 불변)
print("반전 det:", round(det(basis([-2,1],[1,2]))))       # -5  (음수=방향 뒤집힘)
print("선형종속 det:", round(det(basis([3,-1.5],[-2,1])))) # 0   (차원 붕괴=종속)
