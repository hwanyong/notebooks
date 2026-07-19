"""[필수수학] §2.2~2.2.2 확률 계산 — 결합(곱셈 정리)·합(덧셈 법칙) 검증.

목표:
  1) 전수 열거(동전×주사위 12조합)와 곱셈 정리·덧셈 법칙이 일치하나
  2) 단순 덧셈의 오답(0.666, 1.333 — 공리 P(Ω)=1 위반) 재현
  3) 배타 사건에 곱 대입형(-P(A)×P(B), 독립 전용)을 쓰면 오답(11/36)임을 확인

노트: 수학/확률통계/1.2 확률 계산 (원자) · 2단계/02 (렌즈)
"""
from fractions import Fraction
from itertools import product

F = Fraction
space = list(product("HT", range(1, 7)))        # 카티전 곱 — 12조합 전수 열거
assert len(space) == 12
print("전 조합:", " ".join(f"{c}{d}" for c, d in space))

# ① 결합(AND): 열거 vs 곱셈 정리
joint = F(sum(c == "H" and d == 6 for c, d in space), 12)
print(f"P(앞면 AND 6) 열거 = {joint} = {float(joint):.5f}")
assert joint == F(1, 2) * F(1, 6) == F(1, 12)   # 곱셈 정리와 일치

# ② 합(OR): 열거 vs 덧셈 법칙
union = F(sum(c == "H" or d == 6 for c, d in space), 12)
print(f"P(앞면 OR 6) 열거 = {union} = {float(union):.5f}")        # 7/12 = 0.58333
assert union == F(1, 2) + F(1, 6) - F(1, 2) * F(1, 6)             # 덧셈 법칙 일치

naive = F(1, 2) + F(1, 6)
print(f"단순 덧셈(오답) = {naive} = {float(naive):.4f}")           # 2/3 = 0.666...
overflow = F(1, 2) + F(5, 6)
print(f"앞면 OR 1~5 단순 덧셈 = {overflow} = {float(overflow):.4f} → P>1.0, 공리 위반")
assert overflow > 1                                                # 버그 신호 재현

# ③ 배타 사건 4 OR 6: 일반형 ✓ vs 곱 대입형(독립 전용) ✗
general = F(1, 6) + F(1, 6) - 0                  # 교집합 0 — 일반형
misuse = F(1, 6) + F(1, 6) - F(1, 6) * F(1, 6)   # 곱 대입형 오용
print(f"P(4 OR 6) 일반형 = {general}, 곱 대입형 = {misuse}")
assert general == F(1, 3) and misuse == F(11, 36) and general != misuse

print("전부 통과 — 공식은 열거의 지름길, 곱 대입형은 독립 전용(배타≠독립).")
