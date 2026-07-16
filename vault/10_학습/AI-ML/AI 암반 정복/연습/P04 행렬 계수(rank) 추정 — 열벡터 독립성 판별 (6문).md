---
유형: 연습
코스: AI 암반 정복
단계: 1
관련노트: "[[17 행렬 계수(rank) — 정보 차원·최대계수·이동·확장|렌즈 17]] · [[2.7 행렬 계수(rank) — 정보 차원·최대계수·이동·확장|원자 2.7]] · [[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|원자 1.8 (선형독립)]]"
출처: "[실전선대] §5.3 각주 2 (책 자체 연습문제 — 정답은 각주에 이미 공개)"
풀이일: "2026-07-13"
성적: "6/6 (각주로 정답 기지 — 이 세트는 방법론 복기·시각화용)"
재시도: "🔁 권장 (정답 가리고 방법만 다시)"
태그:
  - ai-ml
  - 연습
  - rank
  - 계수
---
# P04 행렬 계수(rank) 추정 — 열벡터 독립성 판별 (6문)

> ✏️ **연습 세트** (책 §5.3 각주 2, 정답 공개형). 정답·방법은 접이식 — 펼치기 전에 손으로 먼저 추정할 것. 운영 규칙: [[_연습 MOC]]
> 🎯 **이 세트가 시험하는 것:** ① 벡터 1개/스칼라배/거의-스칼라배/완전동일/영벡터, 5가지 전형적 rank 패턴 ② "쌍끼리 다르면 독립"의 함정을 3열 이상에서 재확인 ③ **rank는 열공간·행공간을 "각각 구해서 합치는" 게 아니라 둘 중 하나만 구하면 된다**는 절차상의 오해 교정.

## ❗ 방법론 먼저 — "열공간 + 행공간을 각각 구한다"는 것은 오해다

> [!question] 🎯 질문 → 정정
> 질문: "계수는 열공간, 행공간을 각각 구해서 구하는 것 아닌가?"
> **아니다 — 정확히는 "둘 중 아무거나 하나만" 구하면 충분하다.**
> - 정리(2.7 핵심): $\mathrm{rank}(A)=\dim C(A)=\dim R(A)$ — 열공간의 차원과 행공간의 차원은 **항상 같다.** 서로 다른 공간(열공간은 $\mathbb{R}^M$ 속, 행공간은 $\mathbb{R}^N$ 속)에 살면서도 차원 숫자는 우연이 아니라 **정리**로 일치한다.
> - 그래서 실전 절차는: **열을 보고 독립인 열의 최대 개수를 센다 (또는) 행을 보고 독립인 행의 최대 개수를 센다 — 둘 중 계산이 편한 쪽 하나만.** "열공간 구하고, 행공간도 구하고, 그다음 뭘 어떻게 합친다"는 단계는 존재하지 않는다. 두 값을 비교·평균·결합하는 절차가 아니라, 애초에 **같은 값이 나오도록 보장된** 두 개의 서로 다른 셈법 중 하나를 고르는 것뿐이다.
> - 그럼 왜 책([[2.7 행렬 계수(rank) — 정보 차원·최대계수·이동·확장|2.7]] 그림 5-5)은 열·행 둘 다 보여주는가? — 계산법을 알려주려는 게 아니라, **"공간은 다른데 차원은 같다"는 정리 자체를 눈으로 증명**하기 위해서다. 실제로 rank 하나를 구할 땐 벡터 개수가 더 적은 쪽($\min(M,N)$ 방향)을 고르는 게 보통 더 쉽다.
> - 아래 A~F는 전부 3행 행렬이라 **열벡터**($\mathbb{R}^3$)로 그려서 판별한다 — 열이 3개 이하라 눈으로 보기 편해서 고른 관점일 뿐, 행벡터로 판별해도(각 행을 $\mathbb{R}^2$ 또는 $\mathbb{R}^3$ 벡터로) 항상 같은 답이 나온다(D에서 교차검증 코드로 확인).

## 문제

**P1 (A).** $A=\begin{bmatrix}1\\2\\4\end{bmatrix}$ — $r(A)$?
- 시험 개념: 벡터 1개의 rank — [[2.7 행렬 계수(rank) — 정보 차원·최대계수·이동·확장|2.7 §5.3.1]]

**P2 (B).** $B=\begin{bmatrix}1&3\\2&6\\4&12\end{bmatrix}$ — $r(B)$?
- 시험 개념: 열이 정확한 스칼라배 관계 → 종속 — [[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|1.8 §2.3]]

**P3 (C).** $C=\begin{bmatrix}1&3.1\\2&6\\4&12\end{bmatrix}$ — $r(C)$? (B와 딱 숫자 하나만 다르다)
- 시험 개념: "거의 배수"와 "정확히 배수"의 차이 — 선형독립은 엄밀한 0/1 판정이지 정도의 문제가 아님

**P4 (D).** $D=\begin{bmatrix}1&3&2\\6&6&1\\4&2&0\end{bmatrix}$ — $r(D)$?
- 시험 개념: 3열 이상에서 "쌍끼리 방향이 다 다르다"만으론 독립을 보장 못 함 — [[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|1.8 "기울기 다르다=독립"의 함정]] 회수. 정방행렬의 지름길(행렬식)도 함께 확인.

**P5 (E).** $E=\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$ — $r(E)$?
- 시험 개념: 완전히 동일한 벡터의 극단적 종속

**P6 (F).** $F=\begin{bmatrix}0&0&0\\0&0&0\\0&0&0\end{bmatrix}$ — $r(F)$?
- 시험 개념: 영벡터는 독립 방향을 만들지 못함 — [[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|1.8 "영벡터 포함 집합은 항상 종속"]] 회수

---

> [!quote]- ✅ P1(A) 정답·방법 — $r(A)=1$
> **방법:** 열이 1개뿐이면 그 벡터가 **영벡터가 아닌 한** 항상 rank 1이다 — 벡터 하나는 원점을 지나는 1차원 직선 하나를 만들 뿐이고, "독립인지 아닌지" 비교할 다른 벡터 자체가 없다. $[1,2,4]\neq\mathbf{0}$ → $r(A)=1$.
>
> ```plotly
> data:
>   - {type: scatter3d, mode: lines+markers+text, x: [0,1], y: [0,2], z: [0,4], line: {color: "#1c7ed6", width: 8}, marker: {size: [0,5]}, text: ["","col1"], textposition: "top center", name: "col1=[1,2,4]"}
> layout:
>   margin: {l: 0, r: 0, t: 0, b: 0}
>   scene: {aspectmode: cube}
> ```

> [!quote]- ✅ P2(B) 정답·방법 — $r(B)=1$
> **방법:** 두 열이 스칼라배 관계인지 **성분별 비율**로 확인한다: $col_2/col_1 = 3/1,\ 6/2,\ 12/4 = 3,3,3$ — **셋 다 정확히 3으로 동일** → $col_2=3\cdot col_1$, 완전히 같은 직선 위. 독립 방향은 1개뿐 → $r(B)=1$.
>
> ```plotly
> data:
>   - {type: scatter3d, mode: lines+markers+text, x: [0,3], y: [0,6], z: [0,12], line: {color: "#e8590c", width: 9}, marker: {size: [0,5]}, text: ["","col2=3·col1"], textposition: "top center", name: "col2=[3,6,12]"}
>   - {type: scatter3d, mode: lines+markers+text, x: [0,1], y: [0,2], z: [0,4], line: {color: "#1c7ed6", width: 6}, marker: {size: [0,5]}, text: ["","col1"], textposition: "top center", name: "col1=[1,2,4]"}
> layout:
>   margin: {l: 0, r: 0, t: 0, b: 0}
>   scene: {aspectmode: cube}
> ```

> [!quote]- ✅ P3(C) 정답·방법 — $r(C)=2$
> **방법:** 같은 비율 계산: $3.1/1,\ 6/2,\ 12/4 = 3.1,\ 3,\ 3$ — **3.1 하나가 나머지(3)와 다르다.** 즉 $col_2$는 $col_1$의 어떤 스칼라배도 아니다 → 두 열은 독립 → $\mathbb{R}^3$ 안에서 2차원 평면을 생성 → $r(C)=2$.
> 회색 점선(정확히 3배 지점, B의 $col_2$와 같은 자리)과 실제 주황 벡터의 미세한 어긋남을 회전시켜 보면 눈으로도 확인된다 — **0.1의 차이가 "완전히 겹친 직선"과 "분명한 평면"을 가른다.**
>
> ```plotly
> data:
>   - {type: scatter3d, mode: lines, x: [0,3], y: [0,6], z: [0,12], line: {color: "#adb5bd", width: 4, dash: "dot"}, name: "정확히 3배 지점(참고선)"}
>   - {type: scatter3d, mode: lines+markers+text, x: [0,3.1], y: [0,6], z: [0,12], line: {color: "#e8590c", width: 8}, marker: {size: [0,5]}, text: ["","col2=3.1×[1,2,4]?"], textposition: "top center", name: "col2=[3.1,6,12]"}
>   - {type: scatter3d, mode: lines+markers+text, x: [0,1], y: [0,2], z: [0,4], line: {color: "#1c7ed6", width: 6}, marker: {size: [0,5]}, text: ["","col1"], textposition: "top center", name: "col1=[1,2,4]"}
> layout:
>   margin: {l: 0, r: 0, t: 0, b: 0}
>   scene: {aspectmode: cube}
> ```

> [!quote]- ✅ P4(D) 정답·방법 — $r(D)=3$ (최대계수)
> **방법 ① (개념, 3열이라 쌍비교로는 부족):** 세 열 중 하나가 나머지 둘의 선형결합인지를 확인해야 한다 — "두 열씩 짝지어 봤을 때 방향이 다 다르다"는 것만으론 부족하다([[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|1.8의 함정]]: $a,b,c{=}a{+}b$ 처럼 쌍마다 달라도 종속일 수 있다). 직접 $\lambda_1 col_1+\lambda_2 col_2=col_3$ 을 풀어 무모순 해가 있는지 봐야 하는데, 3열이면 손으로 귀찮다.
> **방법 ② (정방행렬 전용 지름길):** $\det(D)$ 를 계산 — 0이 아니면 자동으로 세 열 모두 독립(최대계수).
> $$\det(D)=1(6{\cdot}0-1{\cdot}2)-3(6{\cdot}0-1{\cdot}4)+2(6{\cdot}2-6{\cdot}4)=-2+12-24=-14\neq0$$
> → 최대계수, $r(D)=3$.
> **교차검증(방법론 재확인):** 행 관점 $\mathrm{rank}(D^{\mathsf{T}})$ 을 계산해도 3이 나온다(아래 코드) — "열로 세나 행으로 세나 같다"는 정리를 이 예시로 직접 확인한 것이지, 둘 다 계산해야 답이 나오는 게 아니다.
>
> ```plotly
> data:
>   - {type: scatter3d, mode: lines+markers+text, x: [0,1], y: [0,6], z: [0,4], line: {color: "#1c7ed6", width: 8}, marker: {size: [0,5]}, text: ["","col1"], textposition: "top center", name: "col1=[1,6,4]"}
>   - {type: scatter3d, mode: lines+markers+text, x: [0,3], y: [0,6], z: [0,2], line: {color: "#e8590c", width: 8}, marker: {size: [0,5]}, text: ["","col2"], textposition: "top center", name: "col2=[3,6,2]"}
>   - {type: scatter3d, mode: lines+markers+text, x: [0,2], y: [0,1], z: [0,0], line: {color: "#2f9e44", width: 8}, marker: {size: [0,5]}, text: ["","col3"], textposition: "top center", name: "col3=[2,1,0]"}
> layout:
>   margin: {l: 0, r: 0, t: 0, b: 0}
>   scene: {aspectmode: cube}
> ```

> [!quote]- ✅ P5(E) 정답·방법 — $r(E)=1$
> **방법:** $col_1=col_2=col_3=[1,1,1]$ — 비율이 전부 $1:1:1$, 즉 세 벡터가 **완전히 같다.** 몇 개를 늘어놔도 새로운 방향이 하나도 추가되지 않는다 → $r(E)=1$.
>
> ```plotly
> data:
>   - {type: scatter3d, mode: lines+markers+text, x: [0,1], y: [0,1], z: [0,1], line: {color: "#2f9e44", width: 10}, marker: {size: [0,6]}, text: ["","col1=col2=col3"], textposition: "top center", name: "col1=col2=col3=[1,1,1]"}
> layout:
>   margin: {l: 0, r: 0, t: 0, b: 0}
>   scene: {aspectmode: cube}
> ```

> [!quote]- ✅ P6(F) 정답·방법 — $r(F)=0$
> **방법:** 모든 열이 영벡터. 영벡터는 "방향"이 없어 어떤 독립 방향도 만들지 못한다([[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|1.8 "영벡터 포함 집합은 항상 종속"]] 회수) → 독립 방향 0개 → $r(F)=0$.
>
> ```plotly
> data:
>   - {type: scatter3d, mode: markers+text, x: [0], y: [0], z: [0], marker: {size: 8, color: "#868e96"}, text: ["origin only"], textposition: "top center", name: "col1=col2=col3=[0,0,0]"}
> layout:
>   margin: {l: 0, r: 0, t: 0, b: 0}
>   scene: {aspectmode: cube}
> ```

## 코드 검산
```python
import numpy as np
A = np.array([[1],[2],[4]], dtype=float)
B = np.array([[1,3],[2,6],[4,12]], dtype=float)
C = np.array([[1,3.1],[2,6],[4,12]], dtype=float)
D = np.array([[1,3,2],[6,6,1],[4,2,0]], dtype=float)
E = np.ones((3,3))
F = np.zeros((3,3))

ranks = [np.linalg.matrix_rank(M) for M in (A,B,C,D,E,F)]
assert ranks == [1,1,2,3,1,0], ranks

# B: col2/col1 비율 전부 3 (정확한 배수) / C: 3.1 하나가 다름
assert np.allclose(B[:,1]/B[:,0], 3.0)
assert not np.allclose(C[:,1]/C[:,0], 3.0)

# D: 행렬식으로 최대계수 확인 + 열/행 rank 교차검증
assert np.isclose(np.linalg.det(D), -14.0)
assert np.linalg.matrix_rank(D) == np.linalg.matrix_rank(D.T) == 3   # 방법론: 열로 세나 행으로 세나 같다
```

## 연결
- 시험 대상: [[17 행렬 계수(rank) — 정보 차원·최대계수·이동·확장|렌즈 17]] · [[2.7 행렬 계수(rank) — 정보 차원·최대계수·이동·확장|원자 2.7]]
- 토대: 선형독립·함정 [[1.8 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|1.8]] · 행렬식 지름길 [[2.2 행렬 곱셈·행렬식|2.2]]
- 관리: [[_연습 MOC]] ｜ 오답 시 → 원자 2.7 `막힌 점` 역기입
