"""
[실전선대] §1.3 — 벡터 크기와 단위벡터 (노름)

크기(노름, ‖v‖) = 꼬리→머리 거리 = 표준 유클리드 거리(원소 제곱합의 제곱근).
단위벡터 = 크기가 1인 벡터. v̂ = (1/‖v‖)·v.
⚠️ len() = 수학적 '차원'(원소 수) / np.linalg.norm() = 기하학적 '크기(길이)'. 둘은 다르다.
실행: python la1_3_norm.py
"""
import numpy as np

v = np.array([1, 2, 3, 7, 8, 9])

# 식 1-7: ‖v‖ = sqrt(Σ vᵢ²)
v_dim = len(v)                       # 수학의 '차원'(원소 수) = 6
v_mag = np.linalg.norm(v)            # 기하학적 '크기/노름'
print(f"차원(len)      : {v_dim}")
print(f"크기(norm ‖v‖) : {v_mag:.4f}")
print(f"수동 노름       : {np.sqrt(np.sum(v**2)):.4f}")   # 검증: 동일

# 제곱 크기 ‖v‖² (제곱근 제거 — 일부 응용에서 사용, sqrt 생략으로 더 쌈)
print(f"제곱 크기 ‖v‖² : {np.dot(v, v)}  (= norm² {v_mag**2:.1f})")

# 식 1-8: 단위벡터 = 노름의 역수를 스칼라 곱
v_hat = v / np.linalg.norm(v)        # = (1/‖v‖)·v
print(f"단위벡터 v̂ 크기 : {np.linalg.norm(v_hat):.6f}  (≈ 1)")
print(f"방향 동일 확인  : {np.allclose(v_hat * v_mag, v)}")  # v̂·‖v‖ == v

# ── 노름 수식의 세 표기는 같은 연산 (Σ = for loop, vᵀv = 행렬곱) ──
loop_sum = 0
for x in v:                          # 시그마 Σᵢ₌₁ⁿ vᵢ² = for 루프
    loop_sum += x ** 2               #   바디: 각 원소 제곱해 누적
norm_loop = np.sqrt(loop_sum)        # √(Σvᵢ²)   기하: √(x²+y²+...)
norm_vec  = np.sqrt(np.sum(v ** 2))  # NumPy 벡터화
norm_mat  = np.sqrt(v @ v)           # 행렬곱 √(vᵀv) = np.dot(v.T, v)
print(f"\n세 표기 동일?   loop={norm_loop:.4f} vec={norm_vec:.4f} mat={norm_mat:.4f}",
      "→", np.allclose([norm_loop, norm_vec, norm_mat], v_mag))
print(f"vᵀv (제곱합)   : {v @ v}  = v1²+v2²+...+vn²")

# ── 명시적 (N,1) 열벡터로 vᵀv = (1,N)@(N,1) → (1,1) (→ 탐구 T04 규칙) ──
col = v.reshape(-1, 1)               # (6,1) 명시적 열벡터  [내 2D]
sq_mat = (col.T @ col)               # (1,6)@(6,1) = (1,1)
print(f"(N,1) vᵀv 형태  : col.T@col = {sq_mat.ravel()[0]} shape{sq_mat.shape}",
      "(1D v@v는 같은 값이나 스칼라=내적; 2D는 행렬곱 규격 1:1)")

# ⚠️ 각주: '모든 비단위벡터가 단위벡터를 가진다'의 예외 = 영벡터(zero vector)
zero = np.zeros(3)
print(f"\n영벡터 노름     : {np.linalg.norm(zero)}  → 1/0 불가라 단위벡터 없음")
