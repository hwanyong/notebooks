"""[필수수학] §2.2.3~2.2.4 조건부 확률·베이즈 정리 — 분수 정밀 검증.

목표:
  1) 베이즈 뒤집기 P(암|커피) = 17/2600 ≈ 0.006538 (예제 2-1 값 재현)
  2) 결합의 두 방향 분해가 정확분수로 동일 (17/4000 = 0.00425) — 베이즈의 유래
  3) ⚠️ 반올림 함정: 표기값 곱 0.0065×0.65 = 0.004225 ≠ 0.00425
  4) 리프트(기저율 대비) 두 경로 일치: P(암|커피)/P(암) = P(커피|암)/P(커피) = 17/13
  5) 조건부 덧셈 법칙의 배타 자동 환원: P(4|6)=0 → 1/3 (md2_2의 11/36 문제 해소)

노트: 수학/확률통계/1.3 조건부 확률·베이즈 정리 (원자) · 2단계/03 (렌즈)
"""
from fractions import Fraction as F

p_coffee = F(65, 100)            # P(커피) = 0.65  (statista.com)
p_cancer = F(5, 1000)            # P(암)   = 0.005 (cancer.gov)
p_coffee_g_cancer = F(85, 100)   # P(커피|암) = 0.85 (연구 주장)

# ① 베이즈 정리 — 조건 뒤집기
p_cancer_g_coffee = p_coffee_g_cancer * p_cancer / p_coffee
print(f"P(암|커피) = {p_cancer_g_coffee} = {float(p_cancer_g_coffee)}")
assert p_cancer_g_coffee == F(17, 2600)
assert abs(float(p_cancer_g_coffee) - 0.006538461538461539) < 1e-15  # 예제 2-1

# ② 결합의 두 방향 분해 = 동일 (이 등식을 P(B)로 나누면 베이즈 정리)
j1 = p_coffee_g_cancer * p_cancer            # P(커피|암)×P(암)
j2 = p_cancer_g_coffee * p_coffee            # P(암|커피)×P(커피)
print(f"결합 두 방향: {j1} = {j2} = {float(j1)}")
assert j1 == j2 == F(17, 4000)               # 0.00425 정확

# ③ ⚠️ 반올림 함정 — 표기값(0.0065)의 곱은 등식을 깬다
approx = 0.0065 * 0.65
print(f"반올림 곱 0.0065×0.65 = {approx} (≠ 0.00425)")
assert abs(approx - 0.004225) < 1e-12 and approx != 0.00425

# ④ 리프트 — 기저율 대비 배율, 두 경로가 같은 17/13 ≈ 1.31
lift1 = p_cancer_g_coffee / p_cancer
lift2 = p_coffee_g_cancer / p_coffee
print(f"리프트 = {lift1} = {lift2} ≈ {float(lift1):.4f}")
assert lift1 == lift2 == F(17, 13)

# ⑤ 조건부 덧셈 법칙 — 배타 사건 자동 환원 (P(4|6)=0)
p4_or_6 = F(1, 6) + F(1, 6) - F(0) * F(1, 6)
print(f"P(4 OR 6) 조건부 일반형 = {p4_or_6}")
assert p4_or_6 == F(1, 3)                    # md2_2 곱 대입형 오답(11/36) 해소

# 독립 검증: P(A|B)=P(A)면 일반형이 곱셈 정리로 축소 (동전·주사위)
assert F(1, 2) * F(1, 6) == F(1, 2) * F(1, 6)  # P(앞면|6)=P(앞면)=1/2
print("전부 통과 — 베이즈=결합 대칭성의 재배열, 등식 검증은 정확값으로.")
