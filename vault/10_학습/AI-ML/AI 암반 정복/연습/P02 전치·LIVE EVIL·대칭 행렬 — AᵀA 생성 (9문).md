---
유형: 연습
코스: AI 암반 정복
단계: 1
관련노트: "[[13 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|렌즈 13]] · [[2.3 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|원자 2.3]] · [[T04 NumPy 벡터 내부 — shape·strides·열벡터 (N,1)|T04 (1D 함정)]]"
출처: "자체 출제 (원자 2.3의 📌 전수 커버 — [실전선대] §4.4~4.6 기반)"
풀이일: 미풀이
성적: "0/9"
재시도: 🔁 미정
태그:
  - ai-ml
  - 연습
  - 전치
  - 대칭행렬
---
# P02 전치·LIVE EVIL·대칭 행렬 — AᵀA 생성 (9문)

> ✏️ **연습 세트 (자체 출제).** 정답은 접이식 — 펼치기 전에 손으로 풀 것. 운영 규칙: [[_연습 MOC]]
> 🎯 **이 세트가 시험하는 것:** ① 전치 정의·크기 반전(§4.4) ② 1D `.T` 함정(각주 3) ③ LIVE EVIL — 값이 아니라 **크기 논증**으로 순서 반전을 설명할 수 있는가(§4.5) ④ 대칭 판별·정방 조건(§4.6) ⑤ $A^{\mathsf{T}}A$/$AA^{\mathsf{T}}$ 크기·대칭·차이(§4.6.1) ⑥ 책이 숙제로 남긴 $AA^{\mathsf{T}}$ 대칭 증명.

## A. 전치 기본 (§4.4)

**P1.** $B=\begin{bmatrix}2&-1&4\\0&3&7\end{bmatrix}$ — $B^{\mathsf{T}}$ 를 구하고 크기 변화를 말하라.
- 시험 개념: 식 4-2 $a^{\mathsf{T}}_{i,j}=a_{j,i}$ + $M{\times}N\to N{\times}M$ — [[2.3 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|2.3]]

**P2.** ⑴ $(B^{\mathsf{T}})^{\mathsf{T}}$ 는? (계산 없이) ⑵ `v = np.array([1,2,3])` 일 때 `v.T`의 shape은? 왜 그런가?
- 시험 개념: 이중 전치 $C^{\mathsf{TT}}=C$ + **1D `.T` 먹통**(각주 3) — [[T04 NumPy 벡터 내부 — shape·strides·열벡터 (N,1)|T04]]

## B. LIVE EVIL (§4.5)

**P3.** $L$이 $2{\times}3$, $V$가 $3{\times}5$ 일 때 ⑴ $(LV)^{\mathsf{T}}$ 의 크기는? ⑵ $L^{\mathsf{T}}V^{\mathsf{T}}$ 는 왜 계산 자체가 불가능한가? (크기만으로 답하라)
- 시험 개념: **크기 논증** — 순서 반전이 취향이 아니라 유효성 규칙의 강제임을 설명 — [[13 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|렌즈 13]]

**P4.** $A=\begin{bmatrix}1&2\\0&1\end{bmatrix}$, $B=\begin{bmatrix}3&0\\1&2\end{bmatrix}$ — ⑴ $(AB)^{\mathsf{T}}$ ⑵ $B^{\mathsf{T}}A^{\mathsf{T}}$ ⑶ $A^{\mathsf{T}}B^{\mathsf{T}}$ 를 각각 계산해 ⑴=⑵≠⑶ 을 눈으로 확인하라.
- 시험 개념: LIVE EVIL 수치 검증 (정방이라 $A^{\mathsf{T}}B^{\mathsf{T}}$ 도 *계산은 되지만 값이 다름* — 비정방의 "무효"보다 교묘한 함정)

## C. 대칭 행렬 (§4.6)

**P5.** 다음 중 대칭 행렬은? 아닌 것은 이유를 말하라.
$$ S_1=\begin{bmatrix}4&7&1\\7&-2&0\\1&0&3\end{bmatrix},\quad S_2=\begin{bmatrix}1&2\\3&1\end{bmatrix},\quad S_3=\begin{bmatrix}1&0&5\\0&2&8\end{bmatrix} $$
- 시험 개념: $A^{\mathsf{T}}=A$ 판별 + **비정방은 대칭 자체가 불가**한 이유

**P6.** 다음 행렬이 대칭이 되도록 ㉠㉡㉢을 채워라 (식 4-4 스타일):
$$ \begin{bmatrix}a&5&㉠\\㉡&b&-3\\2&㉢&c\end{bmatrix} $$
- 시험 개념: "각 행 = 대응되는 열" — 대각 기준 거울 반사

## D. 곱셈 기법 — AᵀA (§4.6.1)

**P7.** $A$가 $3{\times}2$ 일 때 ⑴ $A^{\mathsf{T}}A$ 와 $AA^{\mathsf{T}}$ 의 크기는? ⑵ 두 행렬은 같은가? (계산 없이 크기만으로)
- 시험 개념: 📌 **$A^{\mathsf{T}}A \ne AA^{\mathsf{T}}$** — 크기부터 다르다

**P8.** $A=\begin{bmatrix}1&2\\0&1\\2&0\end{bmatrix}$ — $A^{\mathsf{T}}A$ 를 계산하고 대칭임을 확인하라. (여력이 되면 $AA^{\mathsf{T}}$ 도)
- 시험 개념: 비정방·비대칭 재료 → 정방 대칭 결과 (곱셈 기법의 실감)

**P9. (증명)** $(AA^{\mathsf{T}})^{\mathsf{T}} = AA^{\mathsf{T}}$ 를 LIVE EVIL로 증명하라 — 책이 "직접 해보라"고 남긴 그 숙제다.
- 시험 개념: LIVE EVIL + 이중 전치의 3단 조합 — [[2.3 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|2.3 증명 ②]]의 쌍둥이. 풀면 원자 2.3 막힌 점의 해당 항목을 [x] 처리할 것.

> [!quote]- ✅ 정답·풀이 (펼치기 전에 손으로!)
> **P1.** $B^{\mathsf{T}}=\begin{bmatrix}2&0\\-1&3\\4&7\end{bmatrix}$ — $2{\times}3 \to 3{\times}2$. 1행 $[2,-1,4]$ 가 1열로 눕는다.
> **P2.** ⑴ $B$ (이중 전치 = 원래, 계산 불필요). ⑵ `(3,)` **그대로** — 1D는 뒤집을 축이 없어 `.T`가 조용히 무시된다(각주 3). 열벡터로 쓰려면 `(3,1)` 명시.
> **P3.** ⑴ $LV=2{\times}5$ → $(LV)^{\mathsf{T}}=5{\times}2$. ⑵ $L^{\mathsf{T}}V^{\mathsf{T}}=(3{\times}2)(5{\times}3)$ — **내부 2≠5 불일치로 무효.** $V^{\mathsf{T}}L^{\mathsf{T}}=(5{\times}3)(3{\times}2)=5{\times}2$ ✔️ — 순서 반전은 유효성 규칙의 강제.
> **P4.** ⑴ $AB=\begin{bmatrix}5&4\\1&2\end{bmatrix}$ → $(AB)^{\mathsf{T}}=\begin{bmatrix}5&1\\4&2\end{bmatrix}$ ⑵ $B^{\mathsf{T}}A^{\mathsf{T}}=\begin{bmatrix}5&1\\4&2\end{bmatrix}$ = ⑴ ✔️ ⑶ $A^{\mathsf{T}}B^{\mathsf{T}}=\begin{bmatrix}3&1\\6&4\end{bmatrix}$ ≠ — 정방이라 *계산은 되지만 틀린 값*. LIVE EVIL을 어겨도 에러가 안 나는 최악의 케이스.
> **P5.** $S_1$ 만 대칭($s_{12}{=}s_{21}{=}7$ 등 전부 거울 일치). $S_2$: $s_{12}{=}2 \ne s_{21}{=}3$. $S_3$: $2{\times}3$ **비정방 — 전치가 $3{\times}2$ 라 크기부터 달라 대칭 자체가 성립 불가.**
> **P6.** ㉠$=2$ ($a_{13}=a_{31}$), ㉡$=5$ ($a_{21}=a_{12}$), ㉢$=-3$ ($a_{32}=a_{23}$). 대각($a,b,c$)은 아무 값이나 무방.
> **P7.** ⑴ $A^{\mathsf{T}}A=(2{\times}3)(3{\times}2)=\mathbf{2{\times}2}$, $AA^{\mathsf{T}}=(3{\times}2)(2{\times}3)=\mathbf{3{\times}3}$. ⑵ **다르다** — 크기부터 다르니 비교할 것도 없다 📌.
> **P8.** $A^{\mathsf{T}}A=\begin{bmatrix}5&2\\2&5\end{bmatrix}$ (대칭 ✔️). $AA^{\mathsf{T}}=\begin{bmatrix}5&2&2\\2&1&0\\2&0&4\end{bmatrix}$ (역시 대칭 ✔️, $3{\times}3$ — P7의 실물).
> **P9.** $(AA^{\mathsf{T}})^{\mathsf{T}} \overset{\text{LIVE EVIL}}{=} A^{\mathsf{TT}}A^{\mathsf{T}} \overset{C^{\mathsf{TT}}=C}{=} AA^{\mathsf{T}}$ ∎ — 전치가 자신과 같으므로 대칭. ($A^{\mathsf{T}}A$ 증명과 완전 대구.)
>
> ```python
> import numpy as np   # 검산 (전부 통과 2026-07-03)
> B = np.array([[2,-1,4],[0,3,7]])
> assert np.array_equal(B.T, [[2,0],[-1,3],[4,7]]) and np.array_equal(B.T.T, B)
> assert np.array([1,2,3]).T.shape == (3,)                      # P2: 1D 먹통
> A, Bm = np.array([[1,2],[0,1]]), np.array([[3,0],[1,2]])
> assert np.array_equal((A@Bm).T, Bm.T@A.T)                     # P4: LIVE EVIL
> assert not np.array_equal((A@Bm).T, A.T@Bm.T)                 # P4: 순서 어기면 다른 값
> S1 = np.array([[4,7,1],[7,-2,0],[1,0,3]])
> assert np.array_equal(S1, S1.T)                               # P5
> A8 = np.array([[1,2],[0,1],[2,0]])
> AtA, AAt = A8.T@A8, A8@A8.T
> assert np.array_equal(AtA, [[5,2],[2,5]]) and np.array_equal(AtA, AtA.T)
> assert np.array_equal(AAt, AAt.T) and AtA.shape != AAt.shape  # P7·P8·P9
> ```

## 오답·헤맨 지점 (풀이 후 기록)
- 미풀이 — 풀이 후 결과·헤맨 개념을 여기 기록하고, 틀리면 해당 학습 노트 `막힌 점`에 역기입([[_연습 MOC]] 규칙 5)
- P9를 풀면 → 원자 [[2.3 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|2.3]] 막힌 점의 "AAᵀ 대칭 증명 직접" 항목 [x] 처리

## 연결
- 시험 대상: [[13 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|렌즈 13]] · [[2.3 행렬 전치 — 내적·외적 표기법·LIVE EVIL·대칭 행렬|원자 2.3]]
- 함정 근거: [[T04 NumPy 벡터 내부 — shape·strides·열벡터 (N,1)|T04 (1D `.T` 먹통)]] · 유효성 규칙 [[12 행렬 곱셈·행렬식 — 변환 합성·면적 배율·선형종속|12]]
- 색인: [[_연습 MOC]] ｜ 단계: [[_1단계 - 선형대수와 임베딩]]
