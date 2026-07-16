---
유형: 개념노트
코스: AI 암반 정복
단계: 0
도서: "[밑바닥LLM] 부록 A.7 (일반적인 훈련 루프)"
상태: 학습중
순서: 8
이해도: 🟡
태그:
  - ai-ml
  - 훈련루프
  - 손실함수
---

# 훈련 루프 (Training Loop)

> 앞서 만든 부품을 다 합치는 단계 — 모델([[05 퍼셉트론·편향·은닉층|nn.Module]])에 데이터([[07 데이터 로더 (Dataset · DataLoader)|DataLoader]])를 흘려, `forward→loss→backward→step`을 반복해 가중치를 학습시킨다. 부록 A의 종착점.

> 🗺️ [[_0단계 - 닻 올리기|0단계 허브]] ｜ ⬅️ 이전: [[07 데이터 로더 (Dataset · DataLoader)|데이터 로더]] ｜ ➡️ 다음: [[09 모델 저장과 로드 (state_dict)|모델 저장과 로드]]
> 💡 왜·한계·비유(파생 통찰): [[T01 경사 하강법과 훈련 루프|경사 하강법과 훈련 루프]] — *이 노트는 그 의사코드의 책 실제 구현*

> 🎯 **핵심 (TL;DR)**
> - **훈련 루프 5단계:** `forward(logits) → loss(F.cross_entropy) → zero_grad → backward → step` 을 배치마다 반복, 그걸 에포크만큼.
> - **`optimizer.zero_grad()` 필수:** 안 하면 grad가 이전 반복분과 **누적**돼 엉뚱한 갱신.
> - **`F.cross_entropy`:** 로짓을 그대로 받아 내부에서 softmax+음의로그가능도 처리(수치 안정).
> - **`model.train()`/`model.eval()`:** 드롭아웃·배치정규화 같은 *훈련/추론 동작이 다른 층*을 위한 모드 전환.
> - **예측·정확도:** `eval`+`no_grad` → `argmax(logits)` → 정답과 비교.

> [!note] 🗺️ 이 노트 읽는 법
> 책 A.7(훈련 루프)를 *척추*로 한다. [[T01 경사 하강법과 훈련 루프|T01]]에서 "왜 grad로 학습하나·한계(지역 최솟값)·5단계 의사코드"를 미리 팠고, 여기선 **책의 실제 동작 코드**(코드 A-9·A-10)를 본다.

---

> [!question] 🎯 왜 이걸 배우나 — 필요성 사슬 (결핍 → 필요 → 발명)
> *출발 문제(구체적 목표):* 모델(05)·데이터로더(07)·미분(04)은 다 만들었다. 이제 이 부품들을 묶어 가중치를 *실제로* 정답 쪽으로 갱신해야 한다. 그런데 언제·무엇을·어떤 순서로 갱신하나?
> 1. 모델이 얼마나 틀렸는지 숫자로 알아야 함 → 예측을 뽑아 정답과 비교할 손실 필요 → **forward → `F.cross_entropy`**
> 2. 손실만으론 가중치를 어디로 움직일지 모름 → 각 가중치의 책임(기울기) 계산 필요 → **`loss.backward()`**
> 3. PyTorch는 grad를 누적해서, 안 비우면 이전 배치 것이 섞임 → 매번 초기화 필요 → **`optimizer.zero_grad()`**
> 4. 기울기를 알아도 누가 실제로 값을 바꿔주진 않음 → 갱신 실행자 필요 → **`optimizer.step()`**
> 5. 한 번 갱신으론 부족 → 배치마다·에포크마다 반복할 골격 필요 → **이중 `for` 루프**
> **한 줄:** 얼마나 틀렸나(loss) → 어디로 고치나(backward) → 찌꺼기 비우고(zero_grad) → 실제 갱신(step) → 만족까지 반복(epoch).

> [!abstract] 📘 책 핵심 — 부록 A.7 일반적인 훈련 루프
> **① 훈련 루프 (코드 A-9)**
> ```python
> import torch.nn.functional as F
> torch.manual_seed(123)
> model = NeuralNetwork(num_inputs=2, num_outputs=2)
> optimizer = torch.optim.SGD(model.parameters(), lr=0.5)   # 옵티마이저에 파라미터 전달
> num_epochs = 3
> for epoch in range(num_epochs):
>     model.train()
>     for batch_idx, (features, labels) in enumerate(train_loader):
>         logits = model(features)              # forward
>         loss = F.cross_entropy(logits, labels)  # 손실
>         optimizer.zero_grad()                 # grad 초기화(누적 방지)
>         loss.backward()                       # grad 계산
>         optimizer.step()                      # 파라미터 갱신
>     model.eval()
> ```
> - 출력: 에포크가 돌수록 훈련 손실이 `0.75 → … → 0.00` 으로 **수렴**. 3에포크 만에 손실 0 = 훈련셋에 완전 적합(데이터가 작아서).
> - `lr=0.5`(학습률), `num_epochs=3` 은 **하이퍼파라미터**(손실 보고 실험으로 조정).
> - 여기 `model = NeuralNetwork(2, 2)` 는 [[05 퍼셉트론·편향·은닉층|퍼셉트론 노트]]와 같은 클래스지만 입력·출력만 2·2 (그 노트 예시는 50·3). 즉 코드 A-9의 모델 = `NeuralNetwork(2,2)`.
>
> > [!quote]- 📐 연습문제 A.3 — 코드 A-9 신경망의 파라미터 개수 = **752개**
> > `nn.Linear`마다 파라미터 = 가중치(입력×출력) + 편향(출력). `NeuralNetwork(2,2)` 기준:
> > - 1층 `Linear(2,30)`: 2×30 + 30 = **90**
> > - 2층 `Linear(30,20)`: 30×20 + 20 = **620**
> > - 출력 `Linear(20,2)`: 20×2 + 2 = **42**
> > - 합계 = 90 + 620 + 42 = **752** (검증: `sum(p.numel() for p in model.parameters())`)
>
> **② 모드 전환 — `model.train()` / `model.eval()`**
> - 드롭아웃·배치 정규화처럼 *훈련과 추론에서 동작이 다른 층*을 위한 스위치. 이 모델엔 그런 층이 없어 불필요해 보이지만, **재사용·확장 시 사고 방지**로 넣는 게 좋다.
>
> **③ 손실 — `F.cross_entropy`**
> - 로짓을 그대로 전달하면 내부에서 **softmax([ˈsɒftmæks], 소프트맥스, 소프트맥스) + 음의 로그 가능도**를 처리(효율·수치 안정). 그래서 모델 마지막 층에 softmax를 안 붙인다(→ [[05 퍼셉트론·편향·은닉층|softmax]]).
> - 이어 `loss.backward()`(계산 그래프 grad 계산) → `optimizer.step()`(grad로 파라미터 갱신; SGD는 `param -= lr·grad`).
>
> > [!quote]- 📐 NOTE — `zero_grad()`를 왜 매번?
> > PyTorch는 `backward()` 시 grad를 **더한다(누적)**. 반복마다 `optimizer.zero_grad()`로 0을 안 만들면 이전 배치 grad가 쌓여 원치 않은 갱신이 된다.
>
> **④ 예측 & 정확도**
> - `model.eval()` + `with torch.no_grad():` → `outputs = model(X)` → `softmax(dim=1)`로 확률, **`argmax(dim=1)`** 로 클래스 레이블 예측. (확률 없이 로짓에 바로 argmax도 가능)
> - 정확도: `predictions == y` (True/False 텐서) → `torch.sum`으로 맞은 개수. 일반화 함수 **코드 A-10 `compute_accuracy(model, dataloader)`** — 배치를 돌며 `correct/total` 반환. 훈련·테스트셋 모두 `1.0`.
> > [!quote]- 📐 검증셋(validation) — 책 보충
> > 하이퍼파라미터 튜닝엔 **검증셋**(세 번째 데이터)을 쓴다. 테스트셋은 평가 편향 방지를 위해 *딱 한 번만* 쓴다(검증셋은 여러 번).

> 💭 **여기서 내 사고가 분기됨 ↓**

## 🧠 근본 이해 · 융합 · 통찰

### 🔗 T01 의사코드가 '실제 코드'로 — 1:1 매핑
[[T01 경사 하강법과 훈련 루프|T01]]에서 정리한 5단계가 코드 A-9에 그대로 박혀 있다.

| T01 직관 (왜) | 코드 A-9 (어떻게) |
|---|---|
| ① 일단 쏴 보고 점수 | `logits = model(features)` → `loss = F.cross_entropy(...)` |
| ② 거꾸로 역산해 힌트 | `loss.backward()` (← [[04 자동 미분 (autograd)|autograd]]) |
| ③ 나사 살짝 조절 | `optimizer.step()` (SGD: `param -= lr·grad`) |
| (준비) 힌트 지우기 | `optimizer.zero_grad()` |
| ⑤ 만족까지 반복 | `for epoch … for batch …` |

→ T01이 "왜·한계(지역 최솟값·오버슈팅)"를 다뤘다면, 08은 "PyTorch가 실제로 어떻게 그 5단계를 도는가". 둘이 **개념 ↔ 구현** 쌍.

### ⚙️ 부품이 다 모였다 — 0단계의 합류점
이 한 루프에 0단계 전체가 합류한다:
- **모델** = [[05 퍼셉트론·편향·은닉층|nn.Module]] (`model(features)`)
- **데이터** = [[07 데이터 로더 (Dataset · DataLoader)|DataLoader]] (`for … in train_loader`)
- **grad** = [[04 자동 미분 (autograd)|autograd]]·[[03 계산 그래프|계산 그래프]] (`backward()`)
- **추론 모드** = [[06 가중치 — 상태·초기화·학습 실패|no_grad]] (`eval()`+`no_grad`)
- **확률/예측** = [[05 퍼셉트론·편향·은닉층|softmax]]·argmax([ɑːrɡmæks], 아그맥스, 아그맥스)

> [!tip] 🪂 사이드 · 심화 — 손실 0 = 좋기만 한가?   `#사이드/심화`
> 3에포크 만에 훈련 손실 0은 데이터가 5개뿐이라 **외운** 것. 실전에선 train([treɪn], 트레인, 트레인) 정확도 100%·test([tɛst], 테스트, 테스트) 낮음 = **과적합**(→ [[05 퍼셉트론·편향·은닉층|일반화]]). 그래서 검증셋으로 train/val 손실을 함께 본다.

## 🧩 코드 실습 (A.7)
> 🎯 목표: 05 모델 + 07 데이터로더를 합쳐 실제로 학습시키고, 손실 수렴·정확도(`compute_accuracy`)를 확인.
> 코드: `~/LOCAL/03-00_STUDIES/AI/00_setup/a7_training_loop.py`

```python
import torch
import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(3):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        loss = F.cross_entropy(model(x), y)   # forward + 손실
        optimizer.zero_grad()                 # grad 0 (누적 방지)
        loss.backward()                       # grad 계산
        optimizer.step()                      # 파라미터 갱신
        print(f"epoch {epoch+1} batch {batch_idx} loss {loss:.2f}")
    model.eval()

# 정확도 (코드 A-10)
def compute_accuracy(model, loader):
    model.eval(); correct = 0.0; total = 0
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += torch.sum(y == preds); total += len(y)
    return (correct / total).item()
# compute_accuracy(model, train_loader) → 1.0
```

관찰 포인트:
- **`zero_grad`→`backward`→`step`** 순서가 핵심. zero_grad 빼면 grad 누적으로 학습 망가짐.
- **`F.cross_entropy(model(x), y)`** — 로짓 직접 전달(softmax 내장).
- **`compute_accuracy`** 가 로짓→argmax→비교를 배치 단위로 일반화.

### 🍎 Apple([ˈæpəl], 애플, 애플) Silicon([ˈsɪlɪkən], 실리컨, 실리콘)(MPS([ɛm piː ɛs], 엠피에스, 엠피에스)) 전용 — 디바이스-애그노스틱 리팩토링
> 🎯 목표: 내 장비(UMA([juː ɛm eɪ], 유엠에이, 유엠에이)·통합 메모리)에 맞춰 위 코드를 *디바이스-애그노스틱*으로 정리하고, x86+NVIDIA([ɛnˈvɪdiə], 엔비디아, 엔비디아) 전용 군더더기를 걷어낸다.
> 코드: `~/LOCAL/03-00_STUDIES/AI/00_setup/a7_training_loop_mps.py` (검증: 손실→0, 예측 `[0,0,0,1,1]`, acc 1.0)

핵심은 **디바이스-애그노스틱 패턴** — 코드 한 벌로 `mps`·`cuda`·`cpu` 어디서나 돈다.

```python
def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")  # Apple GPU(Metal)
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()
model = NeuralNetwork(2, 2).to(device)                         # 회로(가중치)를 GPU 영토에 상주
for features, labels in train_loader:
    features, labels = features.to(device), labels.to(device)  # UMA: PCIe 이주 아닌 포인터 바인딩(zero-copy)
    ...
```

**NVIDIA 전용 병목 방지 코드 → UMA에서 걷어내기 (역-리팩토링)**

| 옵션 | NVIDIA(분리형) | Apple Silicon(UMA) | 이유 |
|---|---|---|---|
| `pin_memory` | `True`(page-lock([peɪdʒ lɒk], 페이지 락, 페이지 록)+DMA([diː ɛm eɪ], 디엠에이, 디엠에이)) | **생략(False)** | 통합 메모리라 page-lock 무의미(건널 PCIe([piː siː aɪ iː], 피시아이이, 피시아이이)가 없음) |
| `.to(device)` | PCIe 데이터 이주(비쌈) | zero-copy([ˈzɪəroʊ ˈkɒpi], 제로 카피, 제로 카피) 포인터 바인딩 | 전송 병목 구조적 0 |
| `num_workers` | 데이터 준비 병렬화 | **데이터 준비 병렬화(동일)** | ⚠️ 전송과 무관 — 아래 보강 |

> [!tip] 🪂 정밀 보강 — `num_workers`는 UMA로 사라지지 않는다   `#사이드/심화`
> 흔한 오해: *"맥은 zero-copy라 `num_workers=0`이면 그만."* → **틀림.** `num_workers`는 *전송* 병목이 아니라 **디스크 I/O([aɪ oʊ], 아이오, 아이오) + CPU([siː piː juː], 시피유, 시피유) 전처리**(이미지 디코딩·증강 등) 병렬화다. UMA가 없애는 건 *전송 복사*뿐 *데이터 준비 비용*은 그대로 → 무거운 `__getitem__`이면 **맥에서도 `num_workers>0`가 유효.** 이 예제가 `0`인 진짜 이유 = "UMA라서"가 아니라 "데이터가 이미 메모리에 올라온 초소형 토이라서"(워커 오버헤드 > 이득).

→ 결론: 프레임워크(파이토치)가 이미 MPS 백엔드로 *하부 가속 로직을* 리팩토링해 뒀으니, 우리 코드는 **x86 PCIe 때문에 끼워 넣던 군더더기(`pin_memory`·과한 `num_workers`)를 걷고 `device` 변수로 투명하게** 만들면 된다. 단, `num_workers`는 *디바이스*가 아니라 *데이터 준비 비용* 기준으로 판단. (전체 비교 원리 → 🔬 "Apple Silicon UMA" 콜아웃)

> [!abstract]- 📖 책 설명 기반 — 코드 라인별 1:1 매핑
> 《밑바닥부터 만들면서 배우는 LLM》 부록 A의 설명 문장을 각 코드 라인에 그대로 매핑(어느 코드의 설명인지 1:1로).
>
> **`torch.manual_seed(123)`** — 난수 생성기(RNG([ɑːr ɛn dʒiː], 알엔지, 알엔지))의 시드를 특정 값으로 고정하여, 코드를 다시 실행할 때마다 동일한 무작위 가중치가 생성되도록 **재현성(Reproducibility([ˌriːprəˌdjuːsəˈbɪlɪti], 리프로듀서빌리티, 리프로듀서빌리티))** 을 보장한다.
> **`model = NeuralNetwork(num_inputs=2, num_outputs=2)`** — 정의된 신경망 클래스(NeuralNetwork) 구조에 따라 객체를 인스턴스화하며, 입력 특징 수(`num_inputs`)와 최종 분류 출력 클래스 수(`num_outputs`)를 각각 2로 설정해 레이어를 생성한다.
> **`optimizer = torch.optim.SGD(model.parameters(), lr=0.5)`** — 모델의 학습 가능한 매개변수(`model.parameters()`)를 최적화 알고리즘 SGD에 전달하고, 가중치 업데이트 보폭인 학습률(`lr`)을 0.5로 지정한다.
> **`for epoch in range(num_epochs):`** — 전체 훈련 데이터셋을 처음부터 끝까지 온전히 한 번 순회하는 단위를 **에포크(Epoch([ˈiːpɒk], 이폭, 에포크))** 라 하며, 지정 횟수(`num_epochs=3`)만큼 반복해 전체 학습을 관리한다.
> **`model.train()`** — 모델을 **훈련 모드(Training Mode([ˈtreɪnɪŋ moʊd], 트레이닝 모드, 트레이닝 모드))** 로 설정한다. 드롭아웃·배치 정규화처럼 훈련과 평가 시 다르게 동작하는 레이어들이 훈련 프로토콜에 맞춰 작동하도록 플래그를 전환한다.
> **`for batch_idx, (features, labels) in enumerate(train_loader):`** — 데이터 로더(`train_loader`)를 순회하며 대규모 데이터를 **미니배치(Minibatch([ˈmɪnibætʃ], 미니배치, 미니배치))** 단위로 가져온다. 현재 배치 순번(`batch_idx`)·입력 벡터(`features`)·정답 레이블(`labels`)을 반복 추출한다.
> **`logits = model(features)`** — [순방향 전파 / Forward([ˈfɔːrwərd], 포워드, 포워드) Pass([pæs], 패스, 패스)] 입력 데이터(`features`)를 신경망 가중치 행렬에 통과시켜, 최종 활성화 함수(softmax 등)를 거치기 전 원시 예측값인 **로짓(logits([ˈloʊdʒɪts], 로짓츠, 로지트))** 배열을 연산한다.
> **`loss = F.cross_entropy(logits, labels)`** — [손실 계산 / Loss([lɒs], 로스, 로스) Computation([ˌkɒmpjuˈteɪʃən], 컴퓨테이션, 컴퓨테이션)] 로짓 예측값과 데이터셋의 실제 정답 타깃(`labels`)을 교차 엔트로피(Cross Entropy([krɒs ˈɛntrəpi], 크로스 엔트로피, 크로스 엔트로피)) 손실 함수에 대입해, 예측이 정답과 얼마나 다른지 수치화한 **단일 스칼라 손실(loss)** 을 도출한다.
> **`optimizer.zero_grad()`** — [그레디언트 초기화] PyTorch의 Autograd([ˈɔːtoʊɡræd], 오토그래드, 오토그래드) 엔진은 그레디언트가 **누적**되는 성질이 있으므로, 이전 미니배치에서 가중치 버퍼에 남은 손실 그레디언트 기록을 `0`으로 비워 초기화한다.
> **`loss.backward()`** — [역전파 / Backpropagation([ˌbækprɒpəˈɡeɪʃən], 백프로파게이션, 백프로퍼게이션)] 도출된 손실 값에서부터 역방향으로 미분의 **연쇄 법칙(Chain Rule([tʃeɪn ruːl], 체인 룰, 체인 룰))** 을 적용해, 에러를 줄이려 각 가중치 매개변수를 어느 방향·크기로 수정해야 하는지 나타내는 **손실 그레디언트(편미분값)** 를 계산해 기록한다.
> **`optimizer.step()`** — [가중치 업데이트 / Weight([weɪt], 웨이트, 웨이트) Update([ˈʌpdeɪt], 업데이트, 업데이트)] 옵티마이저가 `backward` 단계에서 기록한 손실 그레디언트와 학습률(`lr=0.5`)을 사용해, 모델 내부 매개변수의 값을 실제로 **갱신**한다(경사하강법 적용).
> **`print(f"에포크 …")`** — [학습 지표 출력] 매 미니배치 스텝마다 현재 에포크·배치 인덱스·실시간 손실(`loss`)을 콘솔에 출력해, 가중치가 올바른 방향으로 수렴하는지 확인한다.
> **`model.eval()`** — 하나의 에포크 루프가 끝나면 모델을 **평가 모드(Evaluation Mode([ɪˌvæljuˈeɪʃən moʊd], 이밸류에이션 모드, 이밸류에이션 모드))** 로 전환해, 추론·테스트 단계에서 일관되고 결정론적인 예측이 나오도록 신경망 상태를 고정한다.
>
> → **손실 그레디언트(Loss Gradient([ˈɡreɪdiənt], 그레이디언트, 그레이디언트))** = 손실 함수 값을 기준으로 각 가중치를 편미분해 얻은 *"손실 최소화를 위해 가중치를 어느 방향으로 움직여야 하는지 가리키는 벡터(지침서)"*. `loss.backward()`가 이 지침서를 작성하고, `optimizer.step()`이 그대로 값을 수정한다.
> 📌 이 '정적(컴파일타임)' 훈련 루프와 대비되는 '동적(런타임)' 강화학습 → [[정적 vs 동적 — 컴파일타임과 런타임이 가르는 것들|정적 vs 동적]]

> [!example]- 🧭 사이드 · 개발자 비유 — 손실·그레디언트 용어 번역기   `#사이드/비유`
> *책 설명이 아니라* 전통 개발자 직관으로 번역(이해 보조용). 위 📖 책 설명과 분리.
> - **손실(Loss)** = '에러율(Error Deficit([ˈɛrər ˈdɛfɪsɪt], 에러 데피싯, 에러 데피싯))'. 유닛 테스트에서 기댓값 ≠ 실제 반환값일 때 발생하는 **Assertion Failure([əˈsɜːrʃən ˈfeɪljər], 어서션 페일류어, 어서션 페일러)의 오차 크기**.
> - **그레디언트(Gradient)** = '메모리 가중치 수정 지침서' 또는 **Git([ɡɪt], 깃, 깃) Diff([dɪf], 디프, 디프)의 변경 방향 패치 파일**.
> - **손실 그레디언트(Loss Gradient)** = 에러(`Loss`)를 `0`으로 만들기 위해 각 가중치를 **+로 보정할지 −로 보정할지** 가리키는 **방향·크기의 배열(Vector([ˈvɛktər], 벡터, 벡터))**.

> [!quote]- 🔬 하드웨어·메모리 관점 — 이 루프가 GPU([dʒiː piː juː], 지피유, 지피유)에서 도는 법   `#연결/CS`
> *코드 설명이 아니라*, 이 루프가 시스템 RAM([ræm], 램, 램)·GPU VRAM([viː ræm], 브이램, 브이램)을 어떻게 제어하는지(직접 요청한 심화). 책 설명(📖)과 분리.
>
> **무대 — 두 메모리 영토**
> - **시스템 RAM (Host([hoʊst], 호스트, 호스트)):** `DataLoader`가 디스크에서 퍼 올린 배치가 페이징으로 임시 적재되는 곳.
> - **GPU VRAM (Device([dɪˈvaɪs], 디바이스, 디바이스)):** 가속 연산의 핵심 영토. `model = NeuralNetwork()` 순간 **752개 가중치 원본**이 VRAM 주소 공간(Address Space([ˈædrɛs speɪs], 어드레스 스페이스, 어드레스 스페이스))을 점유.
>
> **스텝별 하드웨어 시퀀스** (미니배치 루프 1회)
> 1. **`model(features)` (forward):** CPU가 PCIe 버스로 RAM의 `features`를 VRAM 캐시(cache([kæʃ], 캐시, 캐시))로 고속 전송 → SM([ɛs ɛm], 에스엠, 에스엠)의 **텐서 코어(Tensor([ˈtɛnsər], 텐서, 텐서) Core([kɔːr], 코어, 코어))**가 병렬 행렬곱 `XW+b` → `logits`를 VRAM 새 주소에 기록 + **동적 계산 그래프**를 링크드 리스트(Linked List([lɪŋkt lɪst], 링크드 리스트, 링크드 리스트))로 연결.
> 2. **`cross_entropy` (loss):** GPU가 VRAM의 `logits`·`labels` 주소를 참조 → 두 분포 차이를 **단일 `float32` 스칼라**로 도출 → 계산 그래프의 **루트(Root([ruːt], 루트, 루트)) 노드**로 바인딩.
> 3. **`zero_grad` (버퍼 플러시):** 가중치와 1:1 크기의 **그레이디언트 버퍼**를 `memset(grad, 0)` 으로 청소(이전 누적 차단).
> 4. **`backward` (역전파 write):** Autograd가 그래프를 루트(`loss`)부터 역추적, 연쇄법칙으로 grad 계산 → 방금 비운 grad 버퍼 주소에 실시간 write.
> 5. **`step` (원자적 overwrite):** SGD 커널이 [가중치 주소]·[grad 버퍼] 값을 로드 → `W = W − lr·grad` → **원본 가중치 메모리를 destructive([dɪˈstrʌktɪv], 디스트럭티브, 디스트럭티브) overwrite([ˌoʊvərˈraɪt], 오버라이트, 오버라이트)**(= 물리적 '학습').
>
> | 단계 | 코드 | 하드웨어 핵심 | VRAM 변동 |
> |---|---|---|---|
> | Forward | `model(features)` | 병렬 행렬곱 | `logits` 신규 할당 + 그래프 생성 |
> | Loss | `cross_entropy` | 오차 스칼라 연산 | 그래프 루트 값 확정 |
> | Zero | `zero_grad` | grad 영역 초기화 | `memset(grad, 0)` |
> | Backward | `backward` | 그래프 역추적·연쇄법칙 | grad 버퍼에 델타 write |
> | Step | `step` | 갱신식·인플레이스 | 원본 가중치 overwrite |
>
> **반복 구조 (중첩 루프):** 안쪽 = 미니배치(데이터 1000·배치 10이면 100회), 바깥 = 에포크(×3) → 가중치 원본 메모리가 **총 300번 진화**. 매 사이클마다 **가중치**(누적 갱신)·**grad 버퍼**(생성→청소 반복)·**loss**(하향 곡선)가 변한다.
> → 한 줄: **"정적 파일을 페이징 스트리밍하며, 하나의 가중치 원본 메모리를 수백 번 destructive하게 덮어써 정답 회로로 굳히는 수치해석 반복 사이클."** 원리 → [[메모리·캐시·멀티프로세스 (시스템 원리)|CS 시스템]] · 정적/동적 → [[정적 vs 동적 — 컴파일타임과 런타임이 가르는 것들|정적 vs 동적]]

> [!quote]- 🔬 하드웨어 심화 — 이 루프의 진짜 병목은 어디? (Host→Device 전송)   `#연결/CS`
> *내 추론 검증:* "매 루프마다 RAM→VRAM 메모리 복사가 병목 아닐까?" → **정확하다.** 딥러닝 엔지니어링에서 이를 **호스트–디바이스 전송 병목(Host-to-Device Transfer([ˈtrænsfər], 트랜스퍼, 트랜스퍼) Bottleneck([ˈbɒtlnɛk], 보틀넥, 보틀넥))** 이라 부르고, 그 결과로 비싼 GPU 연산 코어가 데이터를 기다리며 노는 현상을 **GPU 굶주림(GPU Starvation([stɑːrˈveɪʃən], 스타베이션, 스타베이션))** 이라 한다.
>
> **왜 여기가 병목인가 — 대역폭 비대칭(PCIe의 한계)**
> - GPU **내부** VRAM↔텐서 코어 대역폭 = 초당 수 **TB/s([tiː biː pər ˈsɛkənd], 테라바이트 퍼 세컨드, 테라바이트 퍼 세컨드)** (압도적). 반면 시스템 RAM→VRAM **통로인 PCIe 버스**는 물리적으로 훨씬 좁다. → 같은 데이터라도 *건너오는 데*가 *연산하는 것*보다 느리다.
> - 순진한 동기(synchronous([ˈsɪŋkrənəs], 싱크러너스, 싱크로너스)) 루프면: ① `train_loader`가 RAM에서 배치 꺼냄 → ② PCIe로 VRAM 복사 **(이 동안 GPU 코어는 I/O 블로킹(blocking([ˈblɒkɪŋ], 블로킹, 블로킹))으로 대기)** → ③ forward/backward 고속 연산 → ④ 다음 배치 기다리며 또 굶음. 위 5단계 시퀀스의 **`model(features)` 1번 항목(PCIe 전송)** 이 바로 그 지점.
>
> **융합 포인트 — A.6에서 배운 옵션이 사실 이 병목의 해법이었다**
> [[07 데이터 로더 (Dataset · DataLoader)|A.6 데이터 로더]]에서 "그냥 성능 옵션"으로 넘긴 `num_workers`·`pin_memory`는, 알고 보면 **정확히 이 호스트–디바이스 병목을 숨기려고 존재**한다. (A.6에서 *무엇*을, 여기 A.7 하드웨어 관점에서 *왜* 가 합류)
>
> | 병목 요소 | 증상 | 해법 (A.6 옵션) | 시스템 원리 |
> |---|---|---|---|
> | 전송 대기(직렬) | GPU가 다음 배치 기다리며 유휴 | **`num_workers>0`** | CPU 백그라운드 프로세스가 다음 배치를 *미리* RAM 큐에 적재 → **이중 버퍼링(Double Buffering([ˈdʌbəl ˈbʌfərɪŋ], 더블 버퍼링, 더블 버퍼링))**, 전송을 연산 뒤에 숨김 |
> | 페이징 가능 RAM | OS가 스왑·주소이동 → DMA 불가 | **`pin_memory=True`** | RAM을 **페이지 잠금(Page-locked([peɪdʒ lɒkt], 페이지 락트, 페이지 록트)/Pinned([pɪnd], 핀드, 핀드))** → CPU 개입 없이 **DMA**로 VRAM에 최고 대역폭 비동기 복사 |
>
> → 즉 **`num_workers`는 "전송 시점을 당겨 가리고"(double buffering), `pin_memory`는 "전송 자체를 빠르게"(DMA)** 한다. 둘을 합쳐 GPU 코어가 100% 가동되도록 병목을 *비동기로 은닉*하는 게 딥러닝 파이프라인 최적화의 핵심. (NVLink([ɛn viː lɪŋk], 엔브이링크, 엔브이링크)·`non_blocking=True`·`prefetch_factor` 등 고난도 기법도 전부 같은 목표 → [[T03 PyTorch 데이터 로딩 성능 최적화|성능 탐구]])
> 📎 생산자–소비자·페이지 캐시·DMA 원리 → [[메모리·캐시·멀티프로세스 (시스템 원리)|CS 시스템]] · 전역 다리 → [[_지식 연결 허브]]

> [!quote]- 🔬 하드웨어 심화 — RAM vs VRAM 분업 & 모델 '체급'   `#연결/CS`
> *내 추론 확장 검증* — "SWAP([swɒp], 스왑, 스왑)은 HDD([eɪtʃ diː diː], 에이치디디, 에이치디디)/SSD([ɛs ɛs diː], 에스에스디, 에스에스디)(보조기억)?" → **맞다.** 그 사실이 `pin_memory` 병목의 *실제 정체*를 완성하고, "왜 RAM·VRAM 크기가 AI 학습의 큰 요소인가"로 이어진다.
>
> **① 병목의 진짜 정체 — `pageable → pinned` 중간 복사**
> - 메모리 계층: **레지스터 → 캐시 → RAM(주기억) → SSD/HDD(보조기억)**. OS 가상 메모리는 RAM이 부족하면 안 쓰는 페이지를 **SSD/HDD 스왑 영역으로 swap-out**. 그래서 일반 RAM은 **페이지 가능(Pageable([ˈpeɪdʒəbəl], 페이저블, 페이저블))** = 주소가 언제든 바뀌거나 디스크로 쫓겨날 수 있다.
> - **DMA는 OS를 모른다** — 가상주소·스왑을 모르고 *물리 주소*만 보고 복사. pageable 영역은 복사 도중 OS가 옮기거나 swap할 수 있어 위험 → CUDA([ˈkuːdə], 쿠다, 쿠다)가 어쩔 수 없이 **물리 RAM(pageable) → 안전한 임시 pinned 버퍼 → VRAM** 로 **중간 복사를 한 번 더** 낀다(= 병목의 정체).
> - `pin_memory=True` = OS에 *"이 배치는 절대 스왑·이동 마라"*(page-lock) → 중간 복사 생략, **DMA가 RAM→VRAM 1:1 직통**(중간 오버헤드 0).
>
> **② RAM과 VRAM은 역할이 다르다 — 왜 둘 다 크기가 중요한가**
>
> | | 시스템 RAM (Host) | GPU VRAM (Device) |
> |---|---|---|
> | 역할 | **I/O 완충 큐**(SSD 느림 가리는 캐시) | **학습 회로의 상주 영토** |
> | 담는 것 | 워커가 prefetch한 미니배치 큐 + 데이터 인덱스 | 가중치·grad 버퍼·옵티마이저 상태(Adam([ˈædəm], 애덤, 아담)의 m·v)·활성화값 |
> | 쪼갤 수? | **O** — 데이터는 미니배치로 분할·교체(rotation) | (단일 GPU) **✕** — 회로가 통째 올라가야 |
> | 넘치면 | RAM **OOM([oʊ oʊ ɛm], 오오엠, 오오엠)** → 프로세스 Kill([kɪl], 킬, 킬) | VRAM **OOM** → 모델 로드 자체 불가 |
>
> → 데이터는 `SSD → RAM 큐 → VRAM`로 **슬라이딩 윈도우처럼 흘려보내**면 되니(그 펌프가 [[07 데이터 로더 (Dataset · DataLoader)|`num_workers`]]) RAM은 *완충*만 하면 족하다. 하지만 **모델 회로(가중치+옵티마이저 상태)는 쪼갤 수 없어** 통째 VRAM에 상주 → **VRAM 크기 = "다룰 수 있는 모델 최대 체급"**(가중치가 1바이트만 더 커도 OOM).
>
> **③ 정밀 보강 — '쪼갤 수 없다'는 단일 GPU 한정**
> ②의 "회로는 못 쪼갠다"는 *단일 GPU 1차 근사*. 실제 초거대 모델은 **모델 병렬화·샤딩(sharding([ˈʃɑːrdɪŋ], 샤딩, 샤딩))(Tensor Parallel([ˈpærəlɛl], 패러렐, 패러렐)·FSDP([ɛf ɛs diː piː], 에프에스디피, 에프에스디피)·ZeRO([ˈzɪəroʊ], 제로, 제로))** 으로 가중치·옵티마이저 상태를 여러 GPU에 쪼개 올린다. 또 VRAM은 가중치뿐 아니라 **활성화값**도 먹어 *배치가 클수록* 더 쓴다(그래서 `batch_size`도 VRAM 제약) → 3단계에서 다시.
>
> **④ 초거대 LLM의 종착 — 스트리밍 데이터셋**
> 수십 TB 텍스트는 128GB RAM에도 인덱스조차 다 못 올린다 → **`IterableDataset`** 로 *SSD 파일·네트워크 소켓에서 한 줄씩 실시간으로 짜내* RAM 큐에 공급(데이터 스트리밍). 임의접근 map-style([mæp staɪl], 맵 스타일, 맵 스타일) `Dataset[idx]`와 달리 **순차 스트림**. → [[T03 PyTorch 데이터 로딩 성능 최적화|성능 탐구]]
>
> → 한 줄: **"데이터는 쪼개 흘리면 되니 RAM은 완충이면 족하고, 회로는 (단일 GPU 기준) 못 쪼개니 VRAM은 거거익선."** 메모리 계층 원리 → [[메모리·캐시·멀티프로세스 (시스템 원리)|CS 시스템]] · 전역 다리 → [[_지식 연결 허브]]

> [!quote]- 🔬 하드웨어 심화 — 내 맥(Apple Silicon UMA)에선 이 병목이 사라진다   `#연결/CS`
> *내 환경 적용* — [[01 PyTorch 설치 트러블슈팅 (Apple Silicon)|note 01에서 잡은 MPS(Metal) 백엔드]] = Apple Silicon. 위 호스트–디바이스 병목은 **x86 + NVIDIA 외장 GPU 가정**이고, 내 M칩은 토폴로지가 달라 병목 양상이 바뀐다.
>
> **① 토폴로지 — 분리 vs 통합(UMA)**
> - **NVIDIA 외장 GPU:** CPU용 RAM ↔ GPU용 VRAM이 물리적으로 분리 → **PCIe 버스**를 반드시 건너야 함 → 그 대역폭이 절대 병목.
> - **Apple Silicon (UMA · Unified Memory Architecture([ˈjuːnɪfaɪd ˈmɛməri ˈɑːrkɪtɛktʃər], 유니파이드 메모리 아키텍처, 유니파이드 메모리 아키텍처)):** CPU·GPU·뉴럴엔진(ANE([eɪ ɛn iː], 에이엔이, 에이엔이))이 한 SoC([ɛs oʊ siː], 에스오시, 에스오시) 다이(die([daɪ], 다이, 다이))에 묶이고 고대역폭 RAM을 **통합 공유**. *"시스템 RAM이 곧 VRAM."*
>
> **② 코드 단 대격변 — `.to(device)`의 운명**
> - NVIDIA: `batch.to("cuda")` = PCIe로 **물리적 데이터 이주(Data Migration([daɪˈɡreɪʃən], 마이그레이션, 마이그레이션))** ← 병목.
> - MPS: 같은 통합 메모리라 **데이터 복사 자체가 없다** → CPU가 GPU에 *포인터(주소)만* 넘기는 **제로 카피(Zero-Copy)**. I/O 블로킹 지연이 구조적으로 0. → 그래서 `pin_memory`(page-lock+DMA)·이중 버퍼링 같은 우회 최적화는 **MPS에선 의미가 거의 없다**(애초에 건너갈 버스가 없으니).
>
> **③ 체급 — 용량 한계 돌파**
> 외장 GPU는 VRAM이 하드 상한(예: RTX([ɑːr tiː ɛks], 알티엑스, 알티엑스) 4090 = 24GB, 넘으면 OOM). UMA는 **시스템 메모리 전체를 GPU가 VRAM처럼** 사용 → 통합 메모리 64·128·192GB 구성이면 그만큼 큰 모델 회로를 단일 칩에 통째 적재. (위 ②의 "VRAM=모델 체급"을 자본이 아니라 통합 용량으로 푸는 길)
>
> **④ 정밀 보강 — UMA가 '무조건 우위'는 아니다 (트레이드오프)**
> UMA가 이기는 건 **전송(zero-copy)·용량**이지 *전부가 아니다.*
> - **순수 대역폭·연산 처리량**은 전용 VRAM(GDDR6X([dʒiː diː diː ɑːr sɪks ɛks], 지디디알식스엑스, 지디디알식스엑스)·HBM([eɪtʃ biː ɛm], 에이치비엠, 에이치비엠))을 단 고급 외장 GPU가 여전히 앞선다(H100([eɪtʃ wʌn ˈhʌndrəd], 에이치원헌드레드, 에이치백)의 HBM ≈ 수 TB/s vs 통합 LPDDR5([ɛl piː diː diː ɑːr faɪv], 엘피디디알파이브, 엘피디디알파이브)는 그보다 낮음). 큰 배치를 *쉴 새 없이* 갈아 넣는 대규모 훈련 throughput([ˈθruːpʊt], 스루풋, 스루풋)은 외장 클러스터(cluster([ˈklʌstər], 클러스터, 클러스터)) 우위.
> - 즉 **UMA = "전송 병목 없는 큰 단일 작업대"**(추론·중대형 모델 로컬 구동·메모리 큰 작업에 강함) vs **외장 GPU = "전송은 비싸도 연산이 미친 듯 빠른 전용 엔진"**(대규모 사전훈련). 우열이 아니라 *다른 트레이드오프*.
>
> → 한 줄: **"느린 PCIe 고속도로를 아예 없애고 CPU·GPU가 한 운동장(통합 메모리)을 공유해 포인터만 주고받는 zero-copy 구조 — 단, 전용 VRAM의 폭발적 대역폭·연산력과 맞바꾼 선택."** 토폴로지 다리 → [[_지식 연결 허브]] · 내 환경 → [[01 PyTorch 설치 트러블슈팅 (Apple Silicon)|MPS 설치]]

## 🆕 새 용어 정리
| 용어 | 한 줄 정의 |
|---|---|
| **훈련 루프** | `forward→loss→zero_grad→backward→step`을 배치·에포크로 반복 |
| **`F.cross_entropy`** | 로짓을 받아 softmax+음의로그가능도로 손실 계산(분류 표준) |
| **`optimizer.step()`** | grad로 파라미터 갱신(SGD: `param -= lr·grad`) |
| **`optimizer.zero_grad()`** | grad 0으로 초기화 — 안 하면 누적됨 |
| **`model.train()` / `eval()`** | 드롭아웃·배치정규화의 훈련/추론 모드 전환 |
| **`argmax(dim=1)`** | 행마다 최대값 인덱스 → 클래스 레이블 예측 |
| **정확도** (compute_accuracy) | 맞은 예측 / 전체. `(correct/total).item()` |
| **하이퍼파라미터** | `lr`·`num_epochs`처럼 사람이 정하는 설정값 |
| **검증셋** (validation) | 하이퍼파라미터 튜닝용 세 번째 데이터(테스트는 1회만) |
| **GPU 굶주림** (Starvation) | 호스트–디바이스 전송 병목으로 GPU 코어가 데이터 기다리며 노는 것 |
| **이중 버퍼링** (Double Buffering) | 워커가 다음 배치를 미리 준비해 전송 대기를 연산 뒤에 숨김 |
| **DMA** (직접 메모리 접근) | CPU 개입 없이 page-locked RAM↔VRAM을 하드웨어가 비동기 복사 |
| **페이지 가능/잠금** (Pageable/Pinned) | 일반 RAM은 OS가 스왑·이동 가능; pin하면 물리주소 고정 → DMA 직통 |
| **OOM** (Out of Memory) | VRAM 초과=모델 로드 불가 / RAM 큐 초과=프로세스 Kill |
| **`IterableDataset`** | SSD·네트워크에서 한 줄씩 순차 스트리밍(초거대 데이터용) |
| **모델 병렬·샤딩** (FSDP/ZeRO) | 회로를 여러 GPU에 쪼개 올려 단일 VRAM 한계 돌파 |
| **UMA** (통합 메모리) | CPU·GPU가 한 SoC의 RAM 공유(Apple Silicon) — "RAM=VRAM" |
| **제로 카피** (Zero-Copy) | 통합 메모리라 복사 없이 포인터만 넘김 → PCIe 전송 병목 0 |
| **디바이스-애그노스틱** | `device` 변수로 mps·cuda·cpu 어디서나 도는 코드 패턴 |
| **MPS** (Metal) | Apple Silicon GPU 가속 백엔드 — `.to("mps")` |

## 막힌 점 · 다시 볼 것
- [x] 05 모델 + 07 데이터로더 + 04 autograd를 합쳐 실제 학습 (A.7) → `00_setup/a7_training_loop.py`
- [ ] `cross_entropy`가 softmax+NLL을 어떻게 합치는지 수학적으로 — 2단계 [필수수학]
- [ ] 검증셋으로 과적합 모니터링(train vs val 손실) — 실전 데이터에서

## 📌 암기 카드
> 앞면(cue)을 보고 뒷면을 떠올린 뒤 확인. (→ [[_핵심 암기장]])

| 앞면 (cue) | 뒷면 (외울 내용) |
|---|---|
| 훈련 루프 5단계 (순서) | `forward(logits)` → `loss(F.cross_entropy)` → `zero_grad` → `backward` → `step` |
| 배치 안 코드 4줄 순서 | `loss = F.cross_entropy(model(x), y)` → `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` |
| `optimizer.zero_grad()`를 왜 매번? | PyTorch는 `backward` 시 grad를 누적 → 안 비우면 이전 배치 grad가 쌓여 잘못 갱신 |
| `F.cross_entropy`의 특징 | 로짓을 그대로 받아 내부에서 softmax+음의로그가능도 처리(수치 안정). 마지막 층에 softmax 안 붙임 |
| `optimizer.step()`이 하는 일 | grad로 파라미터 갱신. SGD: `param -= lr·grad` |
| `model.train()` / `model.eval()` | 드롭아웃·배치정규화처럼 훈련/추론 동작이 다른 층을 위한 모드 전환 |
| 예측 절차 | `eval()` + `with torch.no_grad():` → `argmax(logits, dim=1)` → 정답과 비교 |
| 정확도 공식 | 맞은 예측 / 전체 = `(correct / total).item()` |
| 하이퍼파라미터 예 | `lr`(학습률) · `num_epochs` — 사람이 손실 보고 정하는 설정값 |
| 검증셋 vs 테스트셋 | 검증셋=하이퍼파라미터 튜닝(여러 번) / 테스트셋=평가 편향 방지 위해 딱 한 번 |

## 연결
- 이전: [[07 데이터 로더 (Dataset · DataLoader)|데이터 로더]] ｜ 다음: [[09 모델 저장과 로드 (state_dict)|모델 저장과 로드]]
- 개념·한계(파생): [[T01 경사 하강법과 훈련 루프|경사 하강법과 훈련 루프]]
- 합류한 부품: [[05 퍼셉트론·편향·은닉층|모델]]·[[04 자동 미분 (autograd)|autograd]]·[[06 가중치 — 상태·초기화·학습 실패|no_grad]]
- 단계: [[_0단계 - 닻 올리기]]
