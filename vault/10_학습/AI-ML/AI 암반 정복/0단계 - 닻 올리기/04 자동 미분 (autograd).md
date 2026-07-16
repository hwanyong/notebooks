---
유형: 개념노트
코스: AI 암반 정복
단계: 0
도서: "[밑바닥LLM] 부록 A.4 (자동 미분)"
이해도: 🟡
태그:
  - ai-ml
  - autograd
---

# 자동 미분 (autograd([ˈɔːtoʊɡræd], 오토그래드, 오토그래드))

> `requires_grad=True`인 텐서가 있으면 forward([ˈfɔːrwərd], 포워드, 포워드) 그래프가 기록되고, `backward()`가 **연쇄 법칙**으로 가중치별 그레이디언트를 자동 계산한다.

> 🗺️ [[_0단계 - 닻 올리기|0단계 허브]] ｜ ⬅️ 이전: [[03 계산 그래프|계산 그래프]] ｜ ➡️ 다음: [[05 퍼셉트론·편향·은닉층|퍼셉트론·편향·은닉층]]

> 🎯 **핵심 (TL;DR)**
> - `requires_grad=True`면 backward 시 **연쇄 법칙**으로 grad 자동 계산
> - `loss.backward()` 한 줄이면 `.grad`에 ∂L/∂param 저장 → 직접 미분 불필요
> - `grad()`(콕 집어, 디버깅) vs `backward()`(전부, 실전) — 결과 동일

> [!note] 🗺️ 이 노트 읽는 법
> 책 A.4(자동 미분)를 *척추*로, 내가 직접 해본 실습·검산을 가지로. 앞 노트 [[03 계산 그래프|계산 그래프]]에서 만든 그래프를 미분한다.

---

> [!question] 🎯 왜 이걸 배우나 — 필요성 사슬 (결핍 → 필요 → 발명)
> *출발 문제(구체적 목표):* 신경망을 학습시키려면 손실을 줄이는 방향(각 가중치의 기울기)이 필요한데, 수천만 파라미터를 손으로 미분하는 건 불가능하다.
> 1. 가중치를 어느 방향으로 고칠지 모른다 → 손실의 파라미터별 기울기가 필요 → **편도함수(∂L/∂param)**
> 2. 파라미터가 수천만 개라 손으로 못 푼다 → 그래프 따라 기울기를 자동 계산할 엔진이 필요 → **autograd(후진 모드 자동 미분)**
> 3. 그 엔진을 어떻게 부르나 → 한 줄로 모든 leaf에 기울기를 채워야 → **`loss.backward()` → `.grad`**
> 4. 디버깅 땐 하나만 보고 싶다 → 콕 집어 구하는 길도 → **`grad()` (vs backward, 결과 동일)**
> **한 줄:** 기울기가 있어야 학습되는데 손미분은 불가능 → 그래프를 따라 자동으로 미분하는 엔진이 autograd → `backward()` 한 줄로 끝난다.

> [!abstract] 📘 책 핵심 — 부록 A.4 자동 미분
> - 종단 노드 중 하나라도 `requires_grad=True`면 내부에 계산 그래프 생성. 역전파 = **연쇄 법칙(chain([tʃeɪn], 체인, 체인) rule([ruːl], 룰, 룰))** 을 이 그래프에 적용 = *후진 모드 자동 미분*(출력 loss → 입력 방향으로 거꾸로).
> - 편도함수·연쇄법칙을 몰라도 OK — *고수준에서 "연쇄법칙이 파라미터별 손실 그레이디언트를 계산한다"* 만 알면 됨.
> - 코드 A-3: `grad(loss, w1, retain_graph=True)` 수동, 또는 **`loss.backward()`** → 모든 leaf의 `.grad`에 저장. 결과 `∂L/∂w1 ≈ -0.0898`, `∂L/∂b ≈ -0.0817`.
> - **핵심:** `.backward()`가 미분을 처리 → *직접 편도함수를 계산할 필요가 전혀 없다.*

![[그림 A-8 backward 연쇄법칙.png|560]]

> 💭 **여기서 내가 직접 해본 것 ↓**

## 🧩 직접 구해보기 — `grad()` vs `backward()` + 검산
> 코드: `~/LOCAL/03-00_STUDIES/AI/00_setup/a4_autograd.py`

forward 그래프를 만든 뒤, 연쇄 법칙(역전파)으로 가중치별 그레이디언트를 두 방법으로 구한다.

**방법 1 — `grad()` (원하는 것만 콕 집어, 디버깅용):**
```python
from torch.autograd import grad
grad_L_w1 = grad(loss, w1, retain_graph=True)   # (tensor([-0.0898]),)
grad_L_b  = grad(loss, b,  retain_graph=True)   # (tensor([-0.0817]),)
```
- `retain_graph=True`: `grad()`/`backward()`는 기본적으로 계산 후 그래프를 **지운다**(메모리 해제). 그래프를 한 번 더 쓸 거라 유지시킴.

**방법 2 — `loss.backward()` (모든 leaf의 `.grad` 자동 채움, 실전용):**
```python
loss.backward()
print(w1.grad)   # tensor([-0.0898])
print(b.grad)    # tensor([-0.0817])
```

**검산(연쇄 법칙):** $z = x_1w_1+b = 2.42,\quad a=\sigma(z)\approx0.9184$
$$\frac{\partial L}{\partial w_1}=\underbrace{\frac{\partial z}{\partial w_1}}_{x_1}\frac{\partial a}{\partial z}\frac{\partial L}{\partial a}=-(1-a)\,x_1=-0.0816\times1.1\approx-0.0898$$
$$\frac{\partial L}{\partial b}=-(1-a)\approx-0.0817$$
→ 두 방법 + 손계산이 모두 일치. **이게 역전파의 실체.**

> 💡 실전에선 `grad()`를 손으로 부를 일은 거의 없다. `loss.backward()` 한 줄이면 파이토치가 그래프의 모든 미분을 알아서 처리 → 편도 함수를 직접 계산할 필요가 전혀 없다.

## 🌉 시스템적 확장
- 이 "연산을 기록해 두고 거꾸로 미분" 메커니즘이 **모든 딥러닝 학습의 엔진**. 트랜스포머·LoRA([ˈlɔːrə], 로라, 로라)도 결국 같은 autograd 위에서 돈다.

> [!tip] 🌱 탐구 · [[T01 경사 하강법과 훈련 루프|경사 하강법과 훈련 루프]]
> 방금 `backward()`로 구한 `grad`(∂L/∂w)를 **옵티마이저가 받아 `w ← w − lr·grad`로 반복 갱신**하면 손실이 점점 줄어든다 = **학습 루프**. 그 한계(지역 최솟값·오버슈팅)와 보완책 **Adam([ˈædəm], 애덤, 아담)**까지가 탐구 내용.
> _↳ 역전파(grad)가 처음 등장한 이 지점에서 갈라져 나온 챕터-독립 탐구 · 전체_ → [[_탐구 MOC]]

## 막힌 점 · 다시 볼 것
- [x] `loss.backward()` 실행 후 `w1.grad` 값 직접 확인 (A.4) → `-0.0898`, `-0.0817` (책·손계산 일치)
- [ ] `binary_cross_entropy` 가 수학적으로 어떻게 손실을 계산하는지는 2단계에서 다시 (지금은 "벌점 매기는 함수"로만)

## 📌 암기 카드
> 앞면(cue)을 보고 뒷면을 떠올린 뒤 확인. (→ [[_핵심 암기장]])

| 앞면 (cue) | 뒷면 (외울 내용) |
|---|---|
| 역전파의 수학 원리? | 연쇄 법칙(chain rule)을 계산 그래프에 적용 = 후진 모드 자동 미분(출력 loss → 입력 방향) |
| autograd 코딩 핵심 3단계? | ① `requires_grad=True` 표시 ② `loss.backward()` ③ `w.grad`/`b.grad`로 결과 확인 |
| `loss.backward()`가 하는 일? | 모든 leaf 텐서의 `.grad`에 ∂L/∂param 자동 저장 → 직접 미분 불필요 |
| `grad()` vs `backward()`? | `grad()`=원하는 것만 콕(디버깅), `backward()`=모든 leaf(실전); 결과 동일 |
| `retain_graph=True`는 왜? | grad()/backward()는 기본적으로 계산 후 그래프를 지움(메모리 해제) → 한 번 더 쓸 때 유지 |

## 연결
- 이전: [[03 계산 그래프|계산 그래프]] (여기서 만든 forward 그래프를 미분)
- 다음: [[05 퍼셉트론·편향·은닉층|퍼셉트론·편향·은닉층]]
- 단계: [[_0단계 - 닻 올리기]]
