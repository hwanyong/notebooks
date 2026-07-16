"""
[실전선대] §1.4 내적(점곱)·분배·기하·직교 + §1.5 아다마르곱·외적
실행: python la1_4_dot.py
"""
import numpy as np

# ── §1.4 내적 = Σ aᵢbᵢ ────────────────────────────────
v, w = np.array([1,2,3,4]), np.array([5,6,7,8])
print("v·w =", np.dot(v, w))            # 70
print("(10v)·w =", np.dot(10*v, w))     # 700  (스칼라 선형)
print("(-v)·w =", np.dot(-1*v, w))      # -70  (부호만 반전)

# ── §1.4.1 분배 법칙 aᵀ(b+c)=aᵀb+aᵀc ─────────────────
a,b,c = np.array([0,1,2]), np.array([3,5,8]), np.array([13,21,34])
print("분배:", np.dot(a,b+c), "==", np.dot(a,b)+np.dot(a,c),
      np.dot(a,b+c)==np.dot(a,b)+np.dot(a,c))   # 110 == 110 True

# ── §1.4.2 기하: δ = cosθ·‖v‖‖w‖, 직교 → 0 ───────────
e1,e2 = np.array([1,0]), np.array([0,1])
print("직교 e1·e2 =", np.dot(e1,e2))    # 0 (각 90°)
# cosθ 복원
cos = np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w))
print("cosθ(v,w) =", round(float(cos),4))   # ~0.968 (예각, +)

# ── §1.5.1 아다마르곱(원소별) vs §1.5.2 외적(rank-1) ──
print("아다마르 a⊙b:", np.array([5,4,8,2])*np.array([1,0,.5,-1]))  # [5. 0. 4. -2.]
o = np.outer([1,2,3],[4,5])
print("외적 shape:", o.shape, "| rank:", np.linalg.matrix_rank(o))  # (3,2) rank 1

# ── 내적 vs 외적: 전치 위치 (→ T04) ───────────────────
col = np.array([[1],[2],[3]]); row = np.array([[4],[5],[6]]).T   # (3,1),(1,3)
print("vᵀw shape:", (col.T@col).shape, "| vwᵀ shape:", (col@row).shape)  # (1,1) (3,3)
