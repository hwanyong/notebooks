"""[필수수학AI] §2.5 시뮬레이션 데이터 예시 — 키-체중 선형 모델 재현.

목표:
  1) weight = -314.5 + 7.07*height 에서 height=60 → 109.7 검산
  2) 그림 2-4 재현: 54~79인치 키 5,000개 → 노이즈 없는 완벽한 직선
  3) 그림 2-5 재현: 정규 분포 키 샘플링 + 정규 노이즈 → 유사-실데이터(5분 시뮬)

노트: 1단계/19 실제 vs 시뮬레이션 데이터 · 20 수학 모델
그림: 1단계/attachments/MA그림2-4·2-5 (재구성)
"""
import numpy as np

rng = np.random.default_rng(42)
w0, w1 = -314.5, 7.07

# ① (60, 109.7) 검산
assert abs((w0 + w1 * 60) - 109.7) < 1e-9
print(f"height=60 → weight = {w0 + w1 * 60:.1f}")

# ② 그림 2-4: 노이즈 없는 완벽한 직선 — 데이터가 함수 그 자체
h = rng.uniform(54, 79, 5000)
w_clean = w0 + w1 * h
resid = w_clean - (w0 + w1 * h)
print(f"무노이즈 잔차 최대 = {np.abs(resid).max()}")     # 0.0 — 완벽한 직선

# ③ 그림 2-5: 종 모양 키 + 정규 노이즈 (노이즈의 분포도 '선택'이다 — 균등도 가능)
h2 = rng.normal(63.5, 3, 5000)
noise = rng.normal(0, 4, 5000)
w_noisy = w0 + w1 * h2 + noise
corr = np.corrcoef(h2, w_noisy)[0, 1]
print(f"노이즈 시뮬 상관계수 = {corr:.4f}")               # 강한 선형(1 미만)
assert 0.95 < corr < 1.0

# 균등 노이즈 버전(책의 대안 언급) — 무작위성의 분포 선택 비교
noise_u = rng.uniform(-7, 7, 5000)
w_noisy_u = w0 + w1 * h2 + noise_u
print(f"균등 노이즈 상관계수 = {np.corrcoef(h2, w_noisy_u)[0, 1]:.4f}")

print("통과 — 시뮬 데이터의 특징: 데이터 자체가 함수의 기능을 수행한다.")
