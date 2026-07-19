"""[필수수학] §2.3~2.4 이항 분포·베타 분포 — 수치 전수 검증.

목표:
  1) 예제 2-2 pmf 재현 + 공식 C(n,k)·p^k·(1-p)^(n-k) 일치
  2) P(k<=8) 두 경로(막대 합 vs 1-P(9)-P(10)) = 0.2639
  3) 예제 2-3~2-6 베타 CDF 4수치 (0.7748 / 0.2252 / 0.1316 / 0.3386)
  4) ⚠️ 책 매핑 Beta(a=성공, b=실패)의 정점 = 0.875 ≠ 관측률 0.8 (Beta(9,3)이 0.8)

노트: 수학/확률통계/1.4 이항·베타 (원자) · 2단계/04 (렌즈)
그림: 수학/확률통계/attachments/MD그림2-1~2-9·보강 (재구성 스크립트는 그림 생성 시 사용)
"""
from math import comb, isclose

from scipy.stats import binom, beta

n, p = 10, 0.9

# ① 예제 2-2: pmf 전체 출력 + 공식 일치
for k in range(n + 1):
    pk = binom.pmf(k, n, p)
    assert isclose(pk, comb(n, k) * p**k * (1 - p) ** (n - k))
    print(f"{k} - {pk}")
assert isclose(binom.pmf(8, n, p), 0.19371024449999982)   # k=8
assert isclose(binom.pmf(1, n, p), 8.999999999999981e-09) # k=1 (막대 안 보임)
assert max(range(n + 1), key=lambda k: binom.pmf(k, n, p)) == 9  # 최빈 9회

# ② P(k<=8) 두 경로 — 막대 합(상호 배타) vs 여사건
le8 = sum(binom.pmf(k, n, p) for k in range(9))
assert isclose(le8, 1 - binom.pmf(9, n, p) - binom.pmf(10, n, p))
assert isclose(le8, 0.2639010708999999)
print(f"P(k<=8) = {le8}")

# ③ 베타 CDF 4수치 (예제 2-3 ~ 2-6)
assert isclose(beta.cdf(0.90, 8, 2), 0.7748409780000002)
assert isclose(1.0 - beta.cdf(0.90, 8, 2), 0.22515902199999982)
assert isclose(1.0 - beta.cdf(0.90, 30, 6), 0.13163577484183697)
assert isclose(beta.cdf(0.90, 8, 2) - beta.cdf(0.80, 8, 2), 0.33863336199999994)
print("beta cdf(0.9;8,2) =", beta.cdf(0.90, 8, 2))
print("P(p>=0.9 | 8,2)  =", 1.0 - beta.cdf(0.90, 8, 2))
print("P(p>=0.9 | 30,6) =", 1.0 - beta.cdf(0.90, 30, 6))   # 22.5% → 13.16% 압축
print("P(0.8<=p<=0.9)   =", beta.cdf(0.90, 8, 2) - beta.cdf(0.80, 8, 2))

# ④ ⚠️ 정점(모드) — 책 매핑 vs 표준 사후 Beta(s+1, f+1)
mode = lambda a, b: (a - 1) / (a + b - 2)
assert mode(8, 2) == 0.875 and mode(9, 3) == 0.8
print(f"Beta(8,2) mode = {mode(8,2)} (책 매핑) vs Beta(9,3) mode = {mode(9,3)} (관측률)")

print("전부 통과 — 점수(8/10)는 점이 아니라 분포다; 정점은 0.8이 아니라 0.875.")
