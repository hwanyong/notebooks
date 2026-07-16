"""
[실전선대] §3.3 부속 — "제곱근을 생략해도 되는가"의 순위 불변성(모노토닉 논증) 직접 검증.
목표: 제곱거리로 뽑은 argmin과 실제 거리(sqrt 포함)로 뽑은 argmin이 100% 일치하는지 실증.
근거: sqrt는 [0,∞)에서 순증가함수 → a<b ⟺ √a<√b → argmin(순위)이 안 바뀜 (T08 모노토닉 논증).
"""
import numpy as np

rng = np.random.default_rng(0)
data = rng.normal(0, 5, (150, 2))       # 150개 관측치, 2특징
centroids = rng.normal(0, 5, (3, 2))    # 임의의 중심 3개

# 제곱 거리(책의 코드 그대로 — 제곱근 생략)
sq_dists = np.zeros((150, 3))
for ci in range(3):
    sq_dists[:, ci] = np.sum((data - centroids[ci, :])**2, axis=1)

# 실제 거리(제곱근 포함)
real_dists = np.sqrt(sq_dists)

# 두 방식의 argmin이 전부 일치하는가?
g_squared = np.argmin(sq_dists, axis=1)
g_real = np.argmin(real_dists, axis=1)
print("150개 점 전부 같은 클러스터에 배정?", np.array_equal(g_squared, g_real))

assert np.array_equal(g_squared, g_real), "sqrt 유무가 argmin 순위를 바꾸면 안 됨 (모노토닉 논증 위반)"
print(f"검증 통과: {len(data)}개 점 × {centroids.shape[0]}개 중심 = {data.shape[0]*centroids.shape[0]}개 거리쌍 전부 순위 일치")
