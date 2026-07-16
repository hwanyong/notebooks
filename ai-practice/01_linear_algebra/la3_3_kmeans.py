"""
[실전선대] §3.3 k-평균 클러스터링 — 거리 계산(브로드캐스팅)·argmin 배정·중심 갱신 재현.
목표: 5단계 알고리즘(초기화→거리계산→argmin배정→중심갱신→반복) 구현, 합성 3-클러스터 데이터로 수렴 검증.
검증: 정답을 알고 만든 중심([2,2],[6,6],[2,7]) 근처로 수렴 + 클러스터 크기 50/50/50 정확히 일치.
"""
import numpy as np

rng = np.random.default_rng(0)

# 합성 데이터 — 세 개의 자연스러운 무리(정답을 알고 만듦, 검증용)
c1 = rng.normal([2, 2], 0.6, (50, 2))
c2 = rng.normal([6, 6], 0.6, (50, 2))
c3 = rng.normal([2, 7], 0.6, (50, 2))
data = np.vstack([c1, c2, c3])
rng.shuffle(data)

# ① 중심 초기화
k = 3
ridx = rng.choice(range(len(data)), k, replace=False)
centroids = data[ridx, :].copy()

for iteration in range(3):
    # ② 거리 계산 (브로드캐스팅, 제곱근 생략 — T08 모노토닉 논증: argmin 결과는 거리·거리제곱에서 동일)
    dists = np.zeros((data.shape[0], k))
    for ci in range(k):
        dists[:, ci] = np.sum((data - centroids[ci, :])**2, axis=1)
    # ③ 최근접 중심 배정 — argmin은 인덱스(위치)를 반환, min은 값을 반환(혼동 주의)
    groupidx = np.argmin(dists, axis=1)
    # ④ 중심 갱신 — 배정된 점들의 특징별 평균
    for ki in range(k):
        centroids[ki, :] = [np.mean(data[groupidx == ki, 0]),
                             np.mean(data[groupidx == ki, 1])]

print("수렴한 중심:\n", centroids)                                   # [2,2] [6,6] [2,7] 근처로 수렴
print("클러스터 크기:", [np.sum(groupidx == ki) for ki in range(k)])  # [50, 50, 50]

# ── 검증 assert (책의 "결과가 같다" 주장을 직접 확인) ──
real_dists = np.sqrt(dists)
g_real = np.argmin(real_dists, axis=1)
assert np.array_equal(groupidx, g_real), "sqrt 유무가 argmin 결과를 바꾸면 안 됨 (T08 모노토닉 논증 위반)"
assert sorted([np.sum(groupidx == ki) for ki in range(k)]) == [50, 50, 50]
print("검증 통과: sqrt 생략 여부와 무관하게 argmin 배정 동일, 클러스터 크기 50/50/50 정확 일치")
