"""[필수수학] §2.1 확률 이해하기 — 오즈↔확률 변환 검증.

목표:
  1) 오즈↔확률 왕복이 정확히 복원되나 (7/3 → 0.7 → 7/3, Fraction으로 오차 없이)
  2) 상호 배타적 결과의 확률 합 = 1
  3) (🔭 6장 예고 검증) 시그모이드 = 오즈→확률 공식 P=O/(1+O)에 O=e^z 대입과 동일

노트: 수학/확률/1.1 확률 — 정의·가능도·오즈 (원자) · 2단계/01 (렌즈)
"""
from fractions import Fraction
import math


def odds_to_p(o):
    """P = O/(1+O) — 오즈 → 확률."""
    return o / (1 + o)


def p_to_odds(p):
    """O = P/(1−P) — 확률 → 오즈 (odds_to_p의 역함수)."""
    return p / (1 - p)


# ① 책 예제: 오즈 7/3 → 확률 0.7, 왕복 복원
o = Fraction(7, 3)
p = odds_to_p(o)
print(f"O=7/3 → P = {p} = {float(p)}")          # 7/10 = 0.7 (책과 일치)
assert p == Fraction(7, 10)
assert p_to_odds(p) == o                          # 왕복 복원 — 서로 역함수
print(f"P=0.7 → O = {p_to_odds(p)}")             # 7/3

# 여사건: P(not X) = 1 − P(X)
print(f"P(not X) = {1 - Fraction(7, 10)}")        # 3/10 = 0.30

# ② 상호 배타적 결과(주사위 6면)의 확률 합 = 1.0 — 가능도엔 이 규칙 없음
total = sum([Fraction(1, 6)] * 6)
print(f"주사위 6면 확률 합 = {total}")             # 1
assert total == 1

# ③ 시그모이드 = 오즈 변환 공식 (🔭 6장 로지스틱 회귀 예고)
for z in (-2.0, 0.0, 1.234, 5.0):
    via_odds = odds_to_p(math.exp(z))             # O=e^z 를 P=O/(1+O)에
    sigmoid = 1 / (1 + math.exp(-z))              # 표준 시그모이드
    print(f"z={z:+.3f}  오즈경유={via_odds:.6f}  시그모이드={sigmoid:.6f}")
    assert math.isclose(via_odds, sigmoid, rel_tol=1e-12)

print("전부 통과 — 두 변환 공식은 서로 역함수, 시그모이드는 오즈 변환의 지수 버전.")
