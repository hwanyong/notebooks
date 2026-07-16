# AI 암반 정복 — 실습 코드

> 5권 융합 커리큘럼 **"AI 암반 정복"** 의 실습 코드 저장소.
> 📓 **개념·정리 노트는 Obsidian 볼트**(`notebooks/10_학습/AI-ML/AI 암반 정복/`)에 있습니다. 여기는 **코드만** 둡니다.

---

## 📁 폴더 구조 (커리큘럼 단계와 1:1)

| 폴더 | 커리큘럼 단계 | 내용 |
|---|---|---|
| `00_setup/` | 0단계 닻 올리기 | 환경 확인·PyTorch 텐서 기초 |
| `01_linear_algebra/` | 1단계 선형대수·임베딩 | 벡터/행렬, 코사인 유사도, NumPy |
| `02_optimization/` | 2단계 최적화·역전파 | 미적분, 회귀/분류, 역전파 생코딩 |
| `03_transformer/` | 3단계 트랜스포머·LoRA | SVD, 어텐션, GPT 블록, LoRA |
| `04_career/` | 4단계 커리어 분기 | 리서처(생성모델/GNN) · 엔지니어(RAG/서빙) |
| `data/` | 공용 | 데이터셋 (git 미추적) |

---

## 🚀 환경 세팅 (최초 1회)

> torch 2.4.0이 지원하는 **Python 3.12** + 공유 가상환경 1개를 루트에 둡니다.

```bash
cd ~/LOCAL/03-00_STUDIES/AI

# 1. 가상환경 생성 (반드시 3.12)
python3.12 -m venv .venv

# 2. 활성화
source .venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt
```

설치 확인:
```bash
python 00_setup/version_pytorch.py   # -> 2.4.0
python 00_setup/check_spec.py        # -> True (Apple Silicon MPS 사용 가능)
```

> 이후엔 작업 시작할 때 `source .venv/bin/activate` 만 하면 됩니다.
> ⚠️ 설치는 항상 활성화 후 `pip install ...` (또는 `python -m pip install ...`). 자세한 환경 트러블슈팅은 볼트의 *PyTorch 설치 트러블슈팅* 노트 참고.

---

## 🔗 노트 ↔ 코드 연결 규칙
- 볼트 노트에는 **핵심 스니펫만 발췌** + 이 저장소의 **파일 경로**를 적습니다.
  예: `> 코드: ~/LOCAL/03-00_STUDIES/AI/01_linear_algebra/cosine_similarity.py`
- 중요한 시점은 **git 커밋 해시**도 함께 적어 "그때 그 코드"를 되짚습니다.

## 🌱 git 시작 (선택)
```bash
cd ~/LOCAL/03-00_STUDIES/AI
git init
git add .
git commit -m "init: AI 암반 정복 실습 구조"
```
