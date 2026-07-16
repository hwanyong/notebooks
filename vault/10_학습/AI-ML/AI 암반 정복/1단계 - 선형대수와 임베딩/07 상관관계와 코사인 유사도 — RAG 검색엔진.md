---
유형: 적용렌즈
코스: AI 암반 정복
단계: 1
도서: "[실전선대] 3장 §3.1 — 개념 본체는 [[1.9 상관관계와 코사인 유사도 — 피어슨 상관계수·평균중심화·분할정규화]](수학 원자)"
상태: 학습중
순서: 7
이해도: 🟡
태그:
  - ai-ml
  - 코사인유사도
  - 상관계수
  - 적용렌즈
---
# 상관관계와 코사인 유사도 — RAG 검색엔진 (AI 렌즈)

> 🎓 **AI 적용 렌즈.** 개념 본체(정의·유도·증명·그림·코드)는 수학 원자 [[1.9 상관관계와 코사인 유사도 — 피어슨 상관계수·평균중심화·분할정규화]]에 있고 아래에 그대로 끌어와 보여준다. 이 노트는 *AI에서 어떻게 쓰이고 왜 필요한가* 만 더한다.

> 🗺️ [[_1단계 - 선형대수와 임베딩|1단계 허브]] ｜ ⬅️ 이전: [[06 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|벡터 확장]] ｜ ➡️ 다음: [[08 벡터 응용 — 시계열 필터링과 특징 탐지|시계열 필터링·특징탐지]]
> 📐 개념 원자(SSoT): [[1.9 상관관계와 코사인 유사도 — 피어슨 상관계수·평균중심화·분할정규화]] ｜ 🔗 토대: [[04 벡터 내적 — 점곱·기하·직교 + 아다마르곱·외적|내적·코사인 유사도]]

> [!question] 🎯 왜 AI에서 — 필요성 사슬 (AI 앵커)
> *1단계 최종 목표(RAG·시맨틱 검색):* [[04 벡터 내적 — 점곱·기하·직교 + 아다마르곱·외적|04 렌즈]]에서 "코사인 유사도 = RAG의 심장"이라 예고했다 — 이 노트가 그 약속을 실제로 갚는다.
> 1. 문서·쿼리를 벡터(임베딩)로 바꿨다 → 이제 **"이 쿼리와 가장 비슷한 문서는?"** 을 답할 엔진이 필요 → 전 문서와 쿼리의 **코사인 유사도**를 계산해 순위를 매긴다(아래 하드코딩 엔진).
> 2. 그런데 원자 1.9는 "관련성"을 재는 방법이 코사인 유사도 **말고도 상관계수**가 있다고 가르쳤다 → RAG는 왜 상관계수(평균중심화)가 아니라 코사인을 쓰는가? → **상관계수는 "같은 축 위에서 나란히 움직이는 두 데이터열"(시계열·특징 컬럼처럼 인덱스가 맞물린 쌍)을 잴 때 자연스럽다. 반면 쿼리 벡터와 문서 벡터는 애초에 "인덱스로 짝지어진 두 계열"이 아니라 각자 독립적으로 존재하는 한 점(점 대 점)이다** — 평균을 뺄 "짝지어진 축"이 없으므로 코사인이 구조적으로 맞는 도구.
> 3. 문서가 수백만 개면 매번 전부 비교는 느리다 → 인덱싱(사전 벡터화)·근사 최근접 탐색(ANN)이 필요 → 지금은 "정확한 코사인 전수 비교"까지만, ANN은 🔭 예고.
> **한 줄:** 임베딩 하나하나는 짝지어진 시계열이 아니라 독립된 점 → 그래서 평균중심화(상관계수) 아닌 순수 각도(코사인)로 비교 → 전 문서 스캔+순위화가 RAG 검색의 최소 골격. 수학 본체 ↓

---

![[1.9 상관관계와 코사인 유사도 — 피어슨 상관계수·평균중심화·분할정규화]]

---

## 🧠 AI 특화 연결 (이 노트 고유)

> [!quote] 🌉 RAG 파이프라인 = "인덱싱 + 쿼리 임베딩 + 코사인 랭킹" 세 단계   `#연결/AI`
> 실제 RAG 시스템(벡터DB·pgvector·FAISS 등)도 뼈대는 아래 하드코딩 엔진과 동일하다: **① 인덱싱** — 문서마다 임베딩을 미리 계산해 저장(`doc_vectors`에 대응). **② 쿼리 임베딩** — 사용자 질문도 같은 임베딩 함수로 벡터화. **③ 코사인 랭킹** — 쿼리 벡터와 모든 문서 벡터의 코사인 유사도를 구해 상위 k개 반환. 실무 시스템이 다른 건 ①의 임베딩이 (아래처럼 단어 카운트가 아니라) **학습된 신경망**이고, ③의 전수 비교가 **근사 최근접 탐색(ANN)** 으로 가속된다는 점뿐 — 수학 연산 자체(코사인 유사도)는 완전히 같다. 전역 다리 → [[_지식 연결 허브]]

> [!warning] ⚠️ 정확성 — 아래 임베딩은 "장난감"이다, 진짜 의미 이해가 아니다   `#정확성`
> 아래 코드의 `embed()`는 **단어 등장 횟수(bag-of-words)** 로 벡터를 만든다 — 철자가 겹쳐야 벡터가 가까워지는, 순전히 **어휘적(lexical)** 유사도다. 진짜 RAG는 신경망(3단계 트랜스포머)이 문맥으로 학습한 **의미(semantic)** 임베딩을 쓴다 — "cat"과 "feline"처럼 철자가 달라도 의미가 같으면 벡터가 가깝다. 여기서 확인하는 건 **"임베딩을 어떻게 만드는가"가 아니라 "임베딩이 있을 때 검색 엔진이 어떻게 동작하는가"**(코사인 랭킹) — 이 절의 진짜 주제.

## 🧩 코사인 유사도 RAG 엔진 — 하드코딩 (§3.1 목표 코어)
> 🎯 목표: numpy만으로 "문서 인덱싱 → 쿼리 임베딩 → 코사인 랭킹" 검색 엔진을 처음부터 구현. sklearn·벡터DB 없이 엔진 자체의 수학을 직접 짠다.
> 코드: `~/LOCAL/03-00_STUDIES/AI/01_linear_algebra/la3_1_rag_engine.py`

```python
import numpy as np
import re

# ---- 토이 문서 저장소 (실제 RAG의 "지식 베이스") ----
docs = [
    "the cat sat on the mat",
    "dogs are loyal and friendly animals",
    "a cat is a small furry animal that likes to sleep",
    "the dog barked loudly at the mailman",
    "linear algebra uses vectors and matrices to represent data",
    "a vector has both magnitude and direction in space",
]

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

# ---- 어휘 사전(vocabulary) 구축: 코퍼스 전체 단어 집합 ----
vocab = sorted(set(w for d in docs for w in tokenize(d)))
word_to_idx = {w: i for i, w in enumerate(vocab)}

def embed(text):
    """카운트 기반 임베딩(bag-of-words) — 진짜 RAG는 신경망 임베딩을 쓰지만,
    '벡터공간 + 코사인 유사도'라는 검색 엔진의 뼈대는 동일하다."""
    vec = np.zeros(len(vocab))
    for w in tokenize(text):
        if w in word_to_idx:
            vec[word_to_idx[w]] += 1.0
    return vec

# ---- ① 인덱싱: 문서 임베딩을 미리 계산해 저장 ----
doc_vectors = np.array([embed(d) for d in docs])   # (n_docs, |vocab|)

def cosine_similarity(u, v):
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return (u @ v) / denom   # 평균중심화 없음 — 원자 1.9의 cos θ 그대로

def search(query, k=3):
    """② 쿼리 임베딩 → ③ 전 문서와 코사인 유사도 → 상위 k개. 이게 검색 엔진의 전부다."""
    q = embed(query)
    scores = np.array([cosine_similarity(q, dv) for dv in doc_vectors])
    ranked = np.argsort(-scores)          # 내림차순
    return [(docs[i], scores[i]) for i in ranked[:k]]

print("쿼리: 'small furry pets like cats and dogs'")
for doc, score in search("small furry pets like cats and dogs"):
    print(f"  {score:.3f}  {doc}")

print("\n쿼리: 'vectors and matrices in math'")
for doc, score in search("vectors and matrices in math"):
    print(f"  {score:.3f}  {doc}")
```

실행 결과 (검증됨):
```
쿼리: 'small furry pets like cats and dogs'
  0.408  dogs are loyal and friendly animals
  0.277  a cat is a small furry animal that likes to sleep
  0.167  linear algebra uses vectors and matrices to represent data

쿼리: 'vectors and matrices in math'
  0.500  linear algebra uses vectors and matrices to represent data
  0.333  a vector has both magnitude and direction in space
  0.204  dogs are loyal and friendly animals
```

관찰 포인트:
- **레이블·학습 없이 코사인 유사도 하나만으로 주제 클러스터가 분리된다** — "동물" 쿼리는 동물 문서를, "수학" 쿼리는 수학 문서를 정확히 상위에 올림. 이게 RAG의 핵심 트릭: 벡터 공간에 잘 배치하면(임베딩), 나머지는 순수 기하(코사인)로 풀린다.
- `cosine_similarity`가 **평균중심화를 하지 않는다**는 점이 원자 1.9의 예고를 그대로 실증한다 — 쿼리·문서는 "짝지어진 시계열"이 아니라 독립된 점이므로 뺄 평균이 없다.
- 이 엔진의 병목은 `search()`의 `for dv in doc_vectors` 전수 스캔 — 문서 수가 커지면 이 부분이 ANN 인덱스로 대체된다(🔭 예고).

> [!tip] 🔭 예고 — 진짜 임베딩(3단계) · ANN 인덱스(대규모 검색)   `#안내`
> 이 엔진의 `embed()`를 **트랜스포머 인코더**로 바꾸면(3단계) 어휘가 겹치지 않아도 의미가 통하는 문서를 찾는다. 문서가 수백만 개면 `search()`의 전수 비교가 너무 느려 **근사 최근접 탐색(ANN, 예: HNSW·IVF)** 으로 대체된다 — 정확도를 살짝 포기하고 로그 시간에 가까운 검색을 얻는 절충. 지금은 "정확한 코사인 전수 비교"라는 기준선(baseline)을 손으로 짜본 것.

## 막힌 점 · 다시 볼 것
- [x] 상관계수·코사인 유사도 정의·평균중심화/분할정규화·ρ=1 vs cos=0.808 예제 → 원자 [[1.9 상관관계와 코사인 유사도 — 피어슨 상관계수·평균중심화·분할정규화]] · `la3_1_correlation.py`
- [x] **코사인 유사도 엔진 하드코딩**(§3.1 RAG 코어) → 위 `la3_1_rag_engine.py` — [[04 벡터 내적 — 점곱·기하·직교 + 아다마르곱·외적|04 렌즈]]의 예고 이행 완료
- [ ] 진짜 신경망 임베딩으로 `embed()` 교체 → 3단계 트랜스포머
- [ ] ANN 인덱스(HNSW·IVF) 구현·성능 비교 → 4단계 AI엔지니어링 후보
- [ ] 공분산 행렬(covariance matrix) — 이 절의 스칼라 ρ를 다변량으로 확장 → 6.1~6.2

## 연결
- 📐 개념 원자(수학): [[1.9 상관관계와 코사인 유사도 — 피어슨 상관계수·평균중심화·분할정규화]] — 정의·유도·증명·그림·코드 (SSoT)
- 이전: [[06 벡터 확장 — 집합·선형결합·선형독립·부분공간·생성·기저|벡터 확장]] ｜ 다음: [[08 벡터 응용 — 시계열 필터링과 특징 탐지|시계열 필터링·특징탐지]]
- 토대: [[04 벡터 내적 — 점곱·기하·직교 + 아다마르곱·외적|내적·코사인 유사도]] — 이 렌즈가 갚은 예고의 출처
- 단계: [[_1단계 - 선형대수와 임베딩]]
