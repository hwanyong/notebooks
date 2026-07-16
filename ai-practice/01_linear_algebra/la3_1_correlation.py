"""
[실전선대] §3.1 상관관계와 코사인 유사도 — 식3-1(직접)·식3-2(선대 표기)·numpy 내장 3중 검증.
목표: 세 계산법이 같은 rho를 주는지, 코사인 유사도와 왜 다른지 수치로 확인.
검증: rho_direct == rho_la == rho_builtin ≈ 1.0, cos_sim ≈ 0.808 (y=x+100 예제).
"""
import numpy as np

x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([100, 101, 102, 103], dtype=float)

# 식 3-1: 피어슨 상관계수 (직접, 통계 표기)
xbar, ybar = x.mean(), y.mean()
num = np.sum((x - xbar) * (y - ybar))
den = np.sqrt(np.sum((x - xbar)**2)) * np.sqrt(np.sum((y - ybar)**2))
rho_direct = num / den

# 식 3-2: 선형대수 표기 (평균중심화 벡터의 코사인)
xt, yt = x - xbar, y - ybar
rho_la = (xt @ yt) / (np.linalg.norm(xt) * np.linalg.norm(yt))

# numpy 내장 교차검증
rho_builtin = np.corrcoef(x, y)[0, 1]

# 코사인 유사도 (평균중심화 없음 — 원본 그대로)
cos_sim = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

print("피어슨(식3-1 직접):", rho_direct)
print("피어슨(식3-2 선대):", rho_la)
print("numpy corrcoef:   ", rho_builtin)
print("코사인 유사도:      ", cos_sim)

print("\n평균중심화 후 x̃, ỹ:", xt, yt)

assert np.isclose(rho_direct, rho_la) and np.isclose(rho_direct, rho_builtin)
assert np.isclose(rho_direct, 1.0)
assert np.isclose(cos_sim, 0.8083174787557303)
assert np.array_equal(xt, yt), "y=x+100은 순수 이동이므로 평균중심화하면 완전히 같은 벡터가 되어야 함"
print("\n검증 통과: 식3-1=식3-2=corrcoef≈1.0, cos≈0.808, 평균중심화 후 x̃=ỹ")
