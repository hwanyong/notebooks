"""
[실전선대] §3.1 코사인 유사도 RAG 코어 — "문서 인덱싱 → 쿼리 임베딩 → 코사인 랭킹" 검색 엔진 하드코딩.
목표: numpy만으로 bag-of-words 임베딩 + 코사인 랭킹 검색기를 처음부터 구현.
검증: 동물 쿼리는 동물 문서를, 수학 쿼리는 수학 문서를 정확히 상위에 랭킹.
"""
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
res1 = search("small furry pets like cats and dogs")
for doc, score in res1:
    print(f"  {score:.3f}  {doc}")

print("\n쿼리: 'vectors and matrices in math'")
res2 = search("vectors and matrices in math")
for doc, score in res2:
    print(f"  {score:.3f}  {doc}")

assert res1[0][0] == "dogs are loyal and friendly animals"
assert res2[0][0] == "linear algebra uses vectors and matrices to represent data"
print("\n검증 통과: 동물 쿼리→동물 문서 최상위, 수학 쿼리→수학 문서 최상위")
