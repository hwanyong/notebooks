---
유형: 개념노트
코스: AI 암반 정복
단계: 0
도서: "[밑바닥LLM] 부록 A.6 (데이터 로더)"
상태: 학습중
순서: 7
이해도: 🟡
태그:
  - ai-ml
  - 데이터로더
  - dataset
---

# 데이터 로더 (Dataset · DataLoader)

> 모델에 데이터를 *어떻게 떠먹일까*를 책임지는 두 부품 — **`Dataset`**(레코드 한 건 꺼내는 법 정의) + **`DataLoader`**(셔플·배치·병렬로 묶어 공급).

> 🗺️ [[_0단계 - 닻 올리기|0단계 허브]] ｜ ⬅️ 이전: [[06 가중치 — 상태·초기화·학습 실패|가중치]] ｜ ➡️ 다음: [[08 훈련 루프 (Training Loop)|훈련 루프]]

> 🎯 **핵심 (TL;DR)**
> - **`Dataset`** = 3개 메서드(`__init__`·`__getitem__`·`__len__`)만 구현하는 *인터페이스*. "index([ˈɪndɛks], 인덱스, 인덱스) 주면 (특성, 레이블) 한 쌍 반환".
> - **`DataLoader`** = 그 Dataset을 받아 **배치 묶기 + 셔플 + 병렬 로드**를 자동화한 이터레이터.
> - 핵심 인자: `batch_size`(묶음 크기) · `shuffle`(에포크마다 순서 섞기) · `drop_last`(불완전 마지막 배치 버리기) · `num_workers`(병렬 로드 프로세스 수).
> - **에포크** = 전체 데이터를 한 번 다 도는 것. 매 에포크 셔플로 반복 주기 갇힘 방지.

> [!note] 🗺️ 이 노트 읽는 법
> 책 A.6(데이터 로더)를 *척추*로 한다. `nn.Module`로 모델은 만들었으니(05·06), 이제 그 모델에 **데이터를 공급하는 파이프**를 붙이는 단계. 다음 A.7 훈련 루프에서 이 로더를 `for`로 돌려 학습한다.

---

> [!question] 🎯 왜 이걸 배우나 — 필요성 사슬 (결핍 → 필요 → 발명)
> *출발 문제(구체적 목표):* 만든 모델(05·06)에 데이터를 먹여 학습시켜야 한다. 그런데 수십만~수억 건을 한 번에 메모리에 올려 한꺼번에 넣을 수는 없다.
> 1. 데이터 출처가 메모리·CSV·이미지 폴더로 제각각 → 모델은 "한 건 꺼내는 법"만 알면 되도록 통일할 표준 필요 → **`Dataset`(`__getitem__`·`__len__` 인터페이스)**
> 2. 전체를 한 번에 못 올림 → 잘게 끊어 묶음으로 공급할 장치 필요 → **`DataLoader`(배치 묶기)**
> 3. 매 에포크 같은 순서면 모델이 순서 패턴을 외워 편향됨 → 순서를 섞을 장치 필요 → **`shuffle`**
> 4. 데이터 준비(CPU)가 느려 GPU가 다음 배치를 기다리며 굶음 → 백그라운드 선공급 필요 → **`num_workers`(병렬 prefetch)**
> **한 줄:** 큰 데이터를 한 번에 못 먹임 → 한 건 꺼내는 법 통일(`Dataset`) → 배치로 잘라 공급(`DataLoader`) → 섞고(`shuffle`) 미리 준비(`num_workers`).

> [!abstract] 📘 책 핵심 — 부록 A.6 데이터 로더
> 파이토치는 데이터 공급을 **`Dataset` + `DataLoader`** 두 클래스로 분리한다(그림 A-10). `Dataset`은 *레코드 한 건이 로드되는 방법*을 정의하고, `DataLoader`는 *셔플링·배치 묶기·병렬 로드*를 처리한다.
>
> ![[그림 A-10 데이터 로더 구조.png|540]]
> ▲ 그림 A-10 — `Dataset`(레코드 로드법 정의)으로 훈련/테스트 데이터셋 생성 → `DataLoader`(셔플·배치)에 주입.
>
> **① 사용자 정의 `Dataset` (코드 A-6) — 3개 메서드**
> - `__init__` : 속성 준비(파일 경로·텐서 등). 여기선 메모리 텐서 `X`·`y`를 보관.
> - `__getitem__(index)` : index에 해당하는 **(특성 1건, 레이블 1건)** 반환. ← *DataLoader가 이 index를 넘겨준다.*
> - `__len__` : 데이터셋 길이(`labels.shape[0]`). → `print(len(train_ds))` = 5.
> - `train_ds = ToyDataset(X_train, y_train)` 처럼 생성. **목적은 DataLoader에 주입할 객체를 만드는 것.**
>
> **② `DataLoader` (코드 A-7)** — `DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)`
> - `for idx, (x, y) in enumerate(train_loader):` 로 **배치 단위**로 순회. 5샘플·배치2 → 배치 3개(2+2+1).
> - `shuffle=True` + `manual_seed(123)` → 셔플 순서 재현 가능. 두 번째 순회부턴 순서가 또 바뀜(반복 업데이트 주기 방지). **테스트 로더는 `shuffle=False`.**
>
> **③ `drop_last=True` (코드 A-8)** — 마지막 불완전 배치(여기선 1샘플)를 버린다. 에포크 끝 배치가 너무 작으면 훈련 수렴을 방해할 수 있어서.
>
> **④ `num_workers` (그림 A-11)** — 데이터 로드를 **몇 개 프로세스로 병렬화**할지.
> - `0` = 메인 프로세스가 로드 → GPU([dʒiː piː juː], 지피유, 지피유)가 다음 배치를 *기다리는 병목* 가능.
> - `>0` = 워커들이 백그라운드에서 다음 배치를 미리 준비 → GPU가 훈련에만 집중.
> - 단, **작은 데이터셋·주피터 노트북에선 `0`** 권장(프로세스 생성 오버헤드·자원 공유 오류). 실전 대형은 **`num_workers=4`** 가 최적인 경우가 많음(하드웨어·로딩 코드에 따라 다름).
>
> ![[그림 A-11 num_workers 병렬화.png|600]]
> ▲ 그림 A-11 — 워커 없음(`0`): GPU가 다음 배치 로드를 *기다리는 병목*. 워커 사용(`>0`): 백그라운드에서 다음 배치를 미리 준비해 GPU가 안 굶음.
>
> > [!quote]- 📐 NOTE — 클래스 레이블 규칙 (책)
> > 레이블은 **0부터** 시작(파이썬 인덱스). 최대 레이블 값 ≤ *출력 노드 수 − 1*. 레이블이 0~4면 출력층은 5개 노드여야 한다.

> 💭 **여기서 내 사고가 분기됨 ↓** (복습 시 같은 흐름 재현)

## 🧠 근본 이해 · 융합 · 통찰

### 🧱 "코드 A-5가 글이랑 따로 논다?" — 재료 선언 ≠ 파이프라인 장착
> 처음 헷갈린 지점: 글은 `Dataset`·`DataLoader`라는 *거대한 흐름*(그림 A-10)을 말하는데, **코드 A-5는 그냥 텐서 4개를 선언하고 끝났다.** 왜 안 이어지지?
> → 코드 A-5는 그림 A-10의 **맨 왼쪽 바깥 '개별 데이터 레코드'(순수 재료)를 만든 단계일 뿐**, 아직 `Dataset`에 넣지도 `DataLoader`로 묶지도 않았다. 그래서 글과 코드가 따로 노는 것처럼 보였던 것.

**글 설명 ↔ 코드 A-5 1:1 매칭 (재료 준비 단계)**

| 글 설명 | 코드 | 의미 |
|---|---|---|
| "5개 샘플, 2개 특성" | `X_train` (5행 × 2열) | 입력 특성 행렬 |
| "3개는 클래스 0, 2개는 클래스 1" | `y_train = [0,0,0,1,1]` | 정답 레이블(앞 3개 0, 뒤 2개 1) |
| "2개 샘플 테스트셋" | `X_test` · `y_test = [0,1]` | 평가용 |

**데이터 파이프라인 3단계 — 코드 A-5는 ①만 끝낸 상태**
1. **재료 선언 (코드 A-5):** raw 텐서 `X`·`y` 생성. ← *지금 여기*
2. **규격 박스에 포장 (코드 A-6):** `ToyDataset(Dataset)`로 감싸 "index로 한 건 꺼내는" 표준 형태로.
3. **트럭에 싣기 (코드 A-7):** `DataLoader`가 그 박스를 받아 셔플·배치·병렬로 공급.

> [!tip] 🪂 사이드 · 심화 — 왜 굳이 3단계로 쪼개나   `#사이드/심화`
> raw 텐서를 바로 `for`로 돌려도 되지만, **Dataset/DataLoader로 분리하면** 데이터 출처(메모리·CSV·이미지)는 `Dataset`만, 공급 방식(배치·셔플·병렬)은 `DataLoader`만 손대면 된다 = **관심사 분리**. 그래서 '재료(A-5)'와 '설비(A-6·A-7)'를 따로 선언하는 것. (= 공장 비유: 원재료 더미를 먼저 쌓아두고 → 규격 상자에 담고 → 컨베이어에 싣는 순서)

### 🔍 Dataset = '인터페이스 구현', DataLoader = '이터레이터'
전통 개발자 눈으로 보면 이 둘은 익숙한 패턴이다.

- **`Dataset` = 추상 인터페이스 구현.** 파이토치가 정한 규약(`__getitem__`·`__len__`)을 오버라이드하면, 내 데이터가 *파일이든 DB([diː biː], 디비, 디비)든 메모리 텐서든* DataLoader 입장에선 똑같이 "index로 한 건 꺼내는 시퀀스"로 보인다. → **데이터 출처를 추상화**(다형성). CSV([siː ɛs viː], 시에스브이, 시에스브이)든 이미지 폴더든 `__getitem__`만 다시 짜면 끝.
- **`DataLoader` = 제너레이터/이터레이터.** `for ... in loader`가 내부적으로 index를 뽑아 `__getitem__`을 호출하고 배치로 쌓아 `yield`. 반복 로직(셔플·배치·병렬)을 **호출부에서 감춘다**.

> [!example] 🧭 사이드 · 비유   `#사이드/비유`
> - **`Dataset` = 식자재 보관법** — "재료 하나를 어떻게 꺼내는가"만 정의(레시피 아님).
> - **`DataLoader` = 주방 자동화 라인** — 재료를 정해진 묶음(batch([bætʃ], 배치, 배치))으로, 매번 순서 섞어(shuffle([ˈʃʌfəl], 셔플, 셔플)), 여러 일손(workers([ˈwɜːrkərz], 워커즈, 워커스))으로 미리 손질해 셰프(모델)에게 올림.
> 허브: [[_지식 연결 지도]]

> [!tip] 🪂 사이드 · 심화 — `TensorDataset` 지름길 vs 밑바닥 Custom([ˈkʌstəm], 커스텀, 커스텀) Dataset   `#사이드/심화`
> 사실 메모리 텐서뿐이면 `from torch.utils.data import TensorDataset` → `train_ds = TensorDataset(X_train, y_train)` **한 줄**이면 똑같이 된다. 그런데 책이 굳이 `__init__`·`__getitem__`·`__len__`을 손으로 짜는 이유:
> - **밑바닥 원리를 보려고** — DataLoader가 내부에서 `__getitem__(index)`을 호출하는 메커니즘을 직접 눈으로.
> - **실전 데이터는 지름길이 안 통해서** — 이미지 폴더·CSV·DB는 `TensorDataset`로 못 싸고, *반드시* Custom Dataset의 `__getitem__` 안에서 "파일 1건 열어 텐서로 변환"을 직접 구현해야 한다.
> → **`TensorDataset`은 장난감, Custom Dataset이 진짜 무기.** (이 노트 예제는 메모리 텐서라 둘 다 가능하지만, 책은 일부러 어려운 길로 원리를 보여준 것)

> [!example]- 💭 이해용 예시 — 지름길(`TensorDataset`)로 본 전체 흐름 (책 본문 아님)
> A-5의 재료가 A-10 파이프라인에 어떻게 장착되는지 한눈에. *Custom `ToyDataset` 대신 `TensorDataset` 지름길로 압축한 버전.*
> ```python
> from torch.utils.data import TensorDataset, DataLoader
>
> # ① 재료(텐서) → Dataset 박스   (그림 A-10: 훈련/테스트 데이터셋 생성)
> train_dataset = TensorDataset(X_train, y_train)
> test_dataset  = TensorDataset(X_test,  y_test)
>
> # ② Dataset → DataLoader 기계   (그림 A-10: 훈련/테스트 데이터 로더 생성)
> #    여기서 배치 크기·셔플 여부를 결정
> train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
> test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False)
> ```
> 책은 ①의 `TensorDataset`을 `ToyDataset`(Custom)으로 손수 구현한 것일 뿐, 그림 A-10 흐름은 동일. (→ 책 정식 코드는 아래 🧩 코드 실습)

### 🏭 `train_ds = ToyDataset(...)` 한 줄의 의미 — 그림 A-10의 '화살표' 실현
- 3대 메서드(**입고** `__init__` · **재고 수** `__len__` · **출고** `__getitem__`)는 *설계도*일 뿐. `train_ds = ToyDataset(X_train, y_train)`을 **실행하는 순간**, 그림 A-10의 `[사용자 정의 Dataset 클래스] → [훈련 데이터셋]` 화살표가 실제로 동작해 **메모리에 데이터셋 객체가 생긴다.**
- 이 `train_ds`가 다음 단계 `DataLoader(train_ds, …)`에 주입되어 배치·셔플로 공급된다. (= 파이프라인 ②→③ 이음새)

### ⚙️ 인자 4개의 '왜'
- **`batch_size`** — 한 번에 모델에 넣는 샘플 수. 그래디언트를 *여러 샘플 평균*으로 계산해 노이즈를 줄이고 GPU 병렬을 채운다. (행렬 첫 차원 = 배치 → [[02 텐서 이해하기|텐서]] shape의 그 `[Batch, …]`)
- **`shuffle`** — 매 에포크 순서를 섞어 **반복 업데이트 주기**(항상 같은 순서로 학습→편향)를 깬다. SGD([ɛs dʒiː diː], 에스지디, 에스지디)가 가정하는 "샘플이 독립·무작위로 온다"를 만족시키는 장치.
- **`drop_last`** — 5÷2의 나머지 1처럼 **반쪽 배치**는 그래디언트가 튀어 수렴을 흔든다 → 버린다. (배치 통계에 민감한 BatchNorm에서 특히 중요)
- **`num_workers`** — 데이터 준비(CPU)와 연산(GPU)의 **생산자–소비자 파이프라이닝**. 0이면 직렬(GPU 유휴), N이면 워커가 다음 배치를 *prefetch([ˈpriːfɛtʃ], 프리페치, 프리페치)* 해 GPU를 굶기지 않음.

> [!quote] 🌉 사이드 · 타 분야 연결 — 생산자–소비자 / 파이프라이닝   `#연결/시스템`
> `num_workers`는 OS([oʊ ɛs], 오에스, 오에스)의 **멀티프로세싱 + 큐(생산자–소비자)** 그 자체. 워커=생산자(배치 준비), 메인 학습 루프=소비자(GPU 연산). 버퍼가 비면 소비자가 굶고(GPU 유휴), 워커가 너무 많으면 컨텍스트 스위칭·메모리 오버헤드. 스트리밍/ETL([iː tiː ɛl], 이티엘, 이티엘) 파이프라인의 **백프레셔·prefetch**와 같은 트레이드오프. 허브: [[_지식 연결 지도]]

> [!tip] 🪂 사이드 · 심화 — 에포크 · 이터레이션 · 배치   `#사이드/심화`
> **1 에포크** = 전체 데이터 1회 순회. **1 이터레이션** = 배치 1개 처리(= 가중치 1회 갱신). 5샘플·batch 2·drop_last=False면 1에포크 = 3 이터레이션. 학습량은 보통 *에포크 수 × 배치당 갱신*으로 잡는다. → 다음 A.7 훈련 루프에서 이 순회가 곧 학습 루프의 바깥 골격이 된다.

### 🎲 왜 '꼬리 배치'를 버리나 — 배치·셔플·drop_last 깊이 파기
> 문제집 비유: 5문제 문제집을 `batch_size=2`로 "2문제씩 풀고 채점(가중치 갱신)". **에포크**=문제집 한 바퀴, **이터레이션**=채점 1회. 5문제는 `2+2+1` → 마지막 **꼬리 배치 1개**가 남는다.

**① 꼬리 배치(1개)가 수렴을 방해하는 이유 — gradient는 '평균'이라서**
- 모델은 배치 안 샘플들의 **오답(gradient) 평균**으로 학습 방향을 정한다. 2개면 둘을 평균 내 "대략 이 방향"으로 *부드럽게* 내려간다.
- 꼬리 배치는 **단 1개** → 평균 낼 짝이 없다. 그 1개가 괴짜(아웃라이어)면 모델이 **과민 반응해 방향을 휙 틀어** 엉뚱한 곳으로 튄다. 매 에포크 끝마다 이 덜컹거림이 반복되면 최적점에 정착 못 하고 주변을 맴돈다(= **수렴 실패**).
- → 그래서 훈련은 `drop_last=True`로 꼬리를 버려 **안정적 하강**을 지킨다.

**② drop_last가 '데이터 손실'이 아닌 이유 — shuffle과의 결합**
- "데이터를 버리면 손해 아닌가?" → 아니다. `shuffle=True`라 **매 에포크 순서가 새로 섞이고, 버려지는 꼬리 1개도 매번 달라진다.**
  - 1에포크 `[A,B / C,D / E]` → E 버림 · 2에포크 `[B,E / A,C / D]` → D 버림 · 3에포크 → C 버림 …
- 수십~수백 에포크를 돌면 **모든 데이터가 골고루 학습에 참여.** 통째로 잃는 게 아니라, *매 에포크 끝의 불안정한 짜투리 학습*만 막는 안전장치.

**③ 훈련은 버리고, 테스트는 안 버린다 — 차이는 `no_grad`(역전파 유무)**

| | 훈련(train) | 테스트(test) |
|---|---|---|
| 내부 | forward([ˈfɔːrwərd], 포워드, 포워드) → loss([lɒs], 로스, 로스) → **backward([ˈbækwərd], 백워드, 백워드)(가중치 수정)** | forward → **개수(metric([ˈmɛtrɪk], 메트릭, 메트릭))만 기록** (`no_grad`, 가중치 동결) |
| 꼬리 1개 | 가중치를 휙 틀어 **수렴 방해** → 버림(`drop_last=True`) | 가중치 안 건드림 → **무해** |
| 평가 | 무관 | 1개라도 빼면 점수 왜곡 → **전부 평가해야** |

- 그래서 **테스트 로더는 `drop_last=False`(기본)** — 홀수라 꼬리가 남아도 버리면 안 된다. (테스트 = 최종 시험: 전 문항을 채점해야 진짜 실력. 5개 중 1개 빼면 80%만 채점한 꼴)
- 연결: 테스트가 가중치를 안 건드리는 그 장치가 [[06 가중치 — 상태·초기화·학습 실패|torch.no_grad()]]. 꼬리 배치가 '수렴'을 망치는 그 학습 과정이 [[T01 경사 하강법과 훈련 루프|경사 하강법]].

**④ 셔플이 막는 것 — '순서 규칙성(사이클)'에 갇히기**
- 매 에포크 같은 순서면, 모델이 데이터 본질이 아니라 *"A 다음 B가 온다"는 순서 패턴*을 외운다 → 특정 오차 파도에 갇혀 얕은 웅덩이(지역 최솟값)에 빠짐 = **반복 업데이트 주기에 갇힘**.
- `shuffle=True`가 매 에포크 순서를 깨 이를 막는다. (`manual_seed`는 그 무작위를 *재현*만 시킬 뿐, 무작위성 자체를 없애진 않음 → 책과 같은 순서가 나오는 이유)

### 🏗️ DataLoader 내부 = '프로토콜 계약' + 숨은 OS 최적화
> 한발 더: DataLoader는 "데이터 로드·연산을 직렬/병렬로 제어하는 **인터페이스**"이고, Dataset은 "그 인터페이스에 맞춰 데이터를 규격화하는 **프로토콜**"이다. 둘을 나눈 건 전형적인 **관심사 분리(Separation of Concerns([ˌsɛpəˈreɪʃən əv kənˈsɜːrnz], 세퍼레이션 오브 컨선즈, 세퍼레이션 오브 컨선스))** — 이 분리가 깨지면 파이토치 특유의 유연함도 사라진다.

**관심사 분리 — 누가 무엇을 책임지나**

| | 책임 | 비유 |
|---|---|---|
| `Dataset` | **무엇(What)** — "`len()`하면 개수, `[idx]`하면 (X,y)" 약속만 | 규격 **포장지(Protocol([ˈproʊtəkɒl], 프로토콜, 프로토콜))** |
| `DataLoader` | **어떻게(How)** — 섞고·배치로 묶고·병렬로 실어 나름 | 지능형 **수송 트럭(Orchestrator([ˈɔːrkɪstreɪtər], 오케스트레이터, 오케스트레이터))** |

DataLoader 내부는 대략 이렇게 돈다 (Dataset 프로토콜이 작동하는 실증):
```python
# DataLoader 내부 (개념)
def __next__(self):
    indices = self.sampler.get_next_indices()       # __len__ 기반 인덱스 샘플링 (셔플도 여기)
    batch = [self.dataset[idx] for idx in indices]  # [idx] = __getitem__ 호출 (= 프로토콜!)
    return self.collate_fn(batch)                   # 제각각 샘플을 균일 배치 텐서로 묶기
```
→ 엔진은 데이터가 이미지·텍스트·3D든 **관심 없다.** `len()`·`[idx]` 약속만 지키면 GPU로 실어 보낸다 = [[파이썬 OOP — self · super() · 매직 메서드|매직 메서드(프로토콜)]]의 힘.

> [!quote] 🌉 타 영역 연결 — 겉은 단순, 속은 OS레벨 최적화   `#연결/CS`
> "DataLoader 내부도 단순할 줄?" → 인터페이스는 우아하지만, 수면 아래엔 복잡한 **OS·하드웨어 최적화**가 숨어 있다(겉이 단순한 게 곧 관심사 분리의 성공).
> - **GIL([ɡɪl], 길, 길) 우회 + 공유 메모리:** 파이썬 GIL 탓에 `num_workers=4`는 스레드가 아니라 **프로세스 4개를 fork([fɔːrk], 포크, 포크)**. 프로세스 간 대용량 텐서 전달의 직렬화 병목을 **공유 메모리(Shared Memory([ʃɛrd ˈmɛməri], 셰어드 메모리, 셰어드 메모리))** 로 우회(복사 없이 포인터만). → CS/운영체제
> - **`pin_memory` + DMA([diː ɛm eɪ], 디엠에이, 디엠에이):** 일반 RAM([ræm], 램, 램)은 OS가 스왑으로 쫓아낼 수 있음. `pin_memory=True`는 RAM 물리주소에 **고정(page-lock([peɪdʒ lɒk], 페이지 락, 페이지 록))** → GPU가 CPU([siː piː juː], 시피유, 시피유) 개입 없이 **DMA**로 VRAM([viː ræm], 브이램, 브이램)에 초고속 비동기 복사. → 컴퓨터 구조
> - **`collate_fn`:** 가변 길이 텍스트·크기 제각각 이미지를 **패딩·정렬**해 균일 배치 텐서로 묶는 숨은 수집 함수.
> 한 줄: DataLoader = *"너는 모양·개수만 알려줘(Dataset). 프로세스 찢고·공유메모리 열고·RAM 압류(`pin_memory`)해 GPU 앞까지 비동기 대령하는 건 내가."*
> 📎 깊이 파기(캐시 히트·페이지 캐시·IPC([aɪ piː siː], 아이피시, 아이피시)·False Sharing([fɔːls ˈʃɛrɪŋ], 폴스 셰어링, 폴스 셰어링)): [[메모리·캐시·멀티프로세스 (시스템 원리)|CS/메모리·캐시 노트]] · 전역 다리: 🌐 [[_지식 연결 허브]]
> 🎯 *이 옵션들이 왜 존재하나*는 A.7 훈련 루프에서 드러난다 — `num_workers`(이중 버퍼링)·`pin_memory`(DMA)는 매 루프 RAM→VRAM 전송에서 생기는 **호스트–디바이스 병목(GPU 굶주림)** 의 해법: [[08 훈련 루프 (Training Loop)|훈련 루프]] 🔬 하드웨어 심화.

> [!tip] 🌱 탐구 · [[T03 PyTorch 데이터 로딩 성능 최적화|PyTorch 데이터 로딩 성능]]
> 책 A.6은 `num_workers`까지. 그 너머 **로딩을 더 빠르게** 하는 PyTorch([ˈpaɪtɔːrtʃ], 파이토치, 파이토치) 실전 — `prefetch_factor`·`persistent_workers`·RAM 캐시·캐시 친화 `Dataset`·`pin_memory`. (적용 HOW는 탐구 노트, 원리 WHY는 [[메모리·캐시·멀티프로세스 (시스템 원리)|CS 노트]])
> _↳ A.6 DataLoader에서 갈라진 성능 탐구_ · [[_탐구 MOC]]

## 🧩 코드 실습 (A.6)
> 🎯 목표: `Dataset`을 직접 구현해 `DataLoader`로 감싸고, **배치가 어떻게 나뉘고 셔플되는지**·`drop_last`·`num_workers`의 효과를 눈으로 확인.
> 코드: `~/LOCAL/03-00_STUDIES/AI/00_setup/a6_dataloader.py`

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 코드 A-5: 예시 데이터셋 (5 train / 2 test, 특성 2개)
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])          # 클래스 0·1 (0부터 시작!)
X_test  = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test  = torch.tensor([0, 1])

# 코드 A-6: 사용자 정의 Dataset — 3개 메서드만 구현
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels   = y
    def __getitem__(self, index):                 # DataLoader가 index를 넘김
        return self.features[index], self.labels[index]   # (특성 1건, 레이블 1건)
    def __len__(self):
        return self.labels.shape[0]               # 데이터셋 길이

train_ds = ToyDataset(X_train, y_train)
test_ds  = ToyDataset(X_test,  y_test)
print(len(train_ds))                              # 5

# 코드 A-7: DataLoader 초기화
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True,  num_workers=0)
test_loader  = DataLoader(dataset=test_ds,  batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"배치 {idx+1}:", x, y)
# 배치 1·2 = 2샘플, 배치 3 = 1샘플 (5÷2 나머지)

# 코드 A-8: 마지막 반쪽 배치 버리기
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True,
                          num_workers=0, drop_last=True)
for idx, (x, y) in enumerate(train_loader):
    print(f"배치 {idx+1}:", x, y)                 # 배치 1·2만 (배치 3 제외)
```

> [!info] 🔗 타 영역 연결 — 개발/파이썬 · [[파이썬 OOP — self · super() · 매직 메서드|self · super · 매직 메서드]]   `#연결/파이썬`
> `class ToyDataset(Dataset)`의 `self`가 *상속 때문*에 들어가는지 헷갈렸던 지점 → **아니다.** `self`는 상속과 무관한 "자기 자신"(점 앞 객체가 첫 인자로 자동 주입), 부모 접근은 `super()`, 던더(`__init__`·`__getitem__`·`__len__`)는 파이썬이 자동 호출 → 그래서 `DataLoader`가 `dataset[i]`로 한 건씩 꺼낼 수 있다.
> _↳ 파이썬 OOP 기본 문법(개발 영역) · 영역 간 다리_ → 🌐 [[_지식 연결 허브]]

관찰 포인트:
- **3개 메서드면 끝** — `__getitem__`만 바꾸면 CSV·이미지 폴더 등 어떤 출처도 같은 인터페이스가 된다.
- **배치 3개 → drop_last로 2개** — `5 = 2 + 2 + 1` 에서 나머지 1샘플 배치가 사라짐.
- **shuffle 순서** — `manual_seed(123)`라 첫 순회는 책과 동일, 두 번째 순회부턴 순서가 바뀜.
- **num_workers** — 이 예제는 작아서 `0`이 맞다(워커 오버헤드 > 이득). 실전 대형 데이터에서 `4` 등으로 올려 GPU 병목 해소.

## 🆕 새 용어 정리
| 용어 | 한 줄 정의 |
|---|---|
| **`Dataset`** | `__init__`·`__getitem__`·`__len__` 3개로 "index→(특성,레이블)"을 정의하는 인터페이스 |
| **`DataLoader`** | Dataset을 배치·셔플·병렬로 공급하는 이터레이터 |
| **배치** (batch) | 한 번에 모델에 넣는 샘플 묶음. `batch_size`로 지정 |
| **에포크** (epoch) | 전체 데이터를 한 번 다 도는 것 |
| **이터레이션** | 배치 1개 처리 = 가중치 1회 갱신 |
| **`shuffle`** | 매 에포크 샘플 순서를 섞어 반복 주기·편향 방지 |
| **`drop_last`** | 에포크 끝 불완전(반쪽) 배치를 버려 수렴 안정화 |
| **`num_workers`** | 데이터 로드를 병렬화하는 워커 프로세스 수(0=메인, 실전 4 흔함) |
| **prefetch / 병목** | 워커가 다음 배치를 미리 준비해 GPU 유휴(병목)를 막는 것 |

## 막힌 점 · 다시 볼 것
- [x] `Dataset` 3개 메서드 구현 + `DataLoader`로 배치·`drop_last` 확인 (A.6) → `00_setup/a6_dataloader.py`
- [ ] `num_workers`를 0 vs 4로 바꿔 실제 로딩 속도 차이 체감 (대형 데이터 필요, A.9~A.10 GPU에서)
- [ ] 이미지/CSV 등 *실제 파일*을 읽는 `__getitem__` 작성 — 4단계 실전에서

## 📌 암기 카드
> 앞면(cue)을 보고 뒷면을 떠올린 뒤 확인. (→ [[_핵심 암기장]])

| 앞면 (cue) | 뒷면 (외울 내용) |
|---|---|
| `Dataset`이 구현하는 3개 메서드 | `__init__`(속성 준비) · `__getitem__(index)`(특성·레이블 1쌍 반환) · `__len__`(길이) |
| `Dataset` vs `DataLoader` 역할 | Dataset=레코드 한 건 꺼내는 법 정의 / DataLoader=배치·셔플·병렬 공급 이터레이터 |
| `DataLoader` 핵심 인자 4개 | `batch_size`(묶음 크기) · `shuffle`(에포크마다 순서 섞기) · `drop_last`(불완전 배치 버리기) · `num_workers`(병렬 워커 수) |
| 에포크 vs 이터레이션 | 에포크=전체 데이터 1회 순회 / 이터레이션=배치 1개 처리=가중치 1회 갱신 |
| `shuffle`을 켜는 이유 | 매 에포크 순서를 섞어 반복 주기·편향(순서 패턴 암기) 방지 |
| `drop_last=True`의 효과 | 에포크 끝 불완전(반쪽) 배치를 버려 수렴 안정화 |
| 테스트 로더 설정 | `shuffle=False`, `drop_last=False`(전부 평가해야 점수 정확) |
| `num_workers=0` vs `>0` | 0=메인 프로세스 로드(GPU 병목 가능) / `>0`=워커가 다음 배치 prefetch. 토이·주피터는 0 권장 |
| 클래스 레이블 규칙 | 레이블은 0부터 시작, 최대 레이블 ≤ 출력 노드 수 − 1 |

## 연결
- 이전: [[06 가중치 — 상태·초기화·학습 실패|가중치]] ｜ 다음: [[08 훈련 루프 (Training Loop)|훈련 루프]]
- 토대: [[02 텐서 이해하기|텐서]] (배치 = 텐서 첫 차원 `[Batch, …]`)
- 다음 합류: [[T01 경사 하강법과 훈련 루프|경사 하강법과 훈련 루프]] (이 로더를 `for`로 돌리는 게 훈련 루프)
- 단계: [[_0단계 - 닻 올리기]]
