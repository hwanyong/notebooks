---
유형: 개념노트
코스: AI 암반 정복
단계: 0
도서: "[밑바닥LLM] 부록 A.9 (GPU로 훈련 성능 최적화)"
상태: 학습중
순서: 10
이해도: 🟡
태그:
  - ai-ml
  - gpu
  - 분산훈련
  - ddp
---

# GPU 가속과 분산 훈련 (DDP)

> 훈련을 빠르게 — ① 텐서를 **GPU([dʒiː piː juː], 지피유, 지피유) 장치(device([dɪˈvaɪs], 디바이스, 디바이스))** 로 보내 단일 GPU 가속, ② 여러 GPU로 **분산 훈련(DDP([diː diː piː], 디디피, 디디피))**. 0단계의 종착점이자, [[08 훈련 루프 (Training Loop)|08의 🔬 하드웨어 콜아웃]]에서 미리 판 PCIe([piː siː aɪ iː], 피시아이이, 피시아이이)·전송·병렬이 *책 본문으로 확인*되는 장.

> 🗺️ [[_0단계 - 닻 올리기|0단계 허브]] ｜ ⬅️ 이전: [[09 모델 저장과 로드 (state_dict)|모델 저장과 로드]] ｜ ➡️ 다음: (1단계 — 선형대수 예정)

> 🎯 **핵심 (TL;DR)**
> - **장치(device):** 텐서가 *놓인 곳*에서 연산이 일어난다. **모든 텐서가 같은 장치**에 있어야(아니면 RuntimeError([ˈrʌntaɪm ˈɛrər], 런타임 에러, 런타임 에러)).
> - **단일 GPU = 코드 3줄:** ① `device = torch.device(...)` ② `model.to(device)` ③ `features, labels = features.to(device), labels.to(device)`.
> - **디바이스-애그노스틱:** `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — 내 맥은 `"mps"` 분기. 코드 공유 시 표준.
> - **작은 데이터는 가속 無**(전송 비용 > 이득), LLM 같은 심층망에서 큰 향상.
> - **다중 GPU(DDP):** 모델을 GPU마다 **복제** + 데이터를 **분할**(DistributedSampler([dɪˈstrɪbjutɪd ˈsæmplər], 디스트리뷰티드 샘플러, 디스트리뷰티드 샘플러)) → 각자 forward([ˈfɔːrwərd], 포워드, 포워드)/backward([ˈbækwərd], 백워드, 백워드) → grad **동기화(all-reduce([ɔːl rɪˈdjuːs], 올 리듀스, 올 리듀스))**. = **데이터 병렬**.

> [!note] 🗺️ 이 노트 읽는 법
> 책 A.9를 *척추*로 한다. A.9.1(장치 개념)→A.9.2(단일 GPU)→A.9.3(다중 GPU/DDP) 순. **DDP는 NVIDIA([ɛnˈvɪdiə], 엔비디아, 엔비디아) 다중 GPU 전용**이라 내 맥(단일 MPS([ɛm piː ɛs], 엠피에스, 엠피에스))에선 안 돌지만, 책도 *"여러 GPU 필요 없다, 궁금한 독자용"* 이라 명시 — 개념·구조 위주로 본다.

> [!warning] 🔖 복습 지점 — DDP(A.9.3)부터는 훑고 넘어감 (RESUME([rɪˈzuːm], 리줌, 리줌) HERE([hɪər], 히어, 히어))
> **학습 완료:** A.9.1 장치 · A.9.2 단일 GPU(내 맥 MPS) — 여기까지는 이해함.
> **보류(대충 확인):** **A.9.3 다중 GPU(DDP)부터는 정독 안 하고 다음 단계로 진행.** 분산 훈련이 실제 필요해질 때(3~4단계 대규모 학습) 다시 정독한다. 책도 "여러 GPU 불필요, 궁금한 독자용"이라 했으니 지금 보류 OK.
> ↳ **재학습 범위:** 아래 "③ A.9.3 DDP" · 🔑 데이터/모델 병렬 · 🌉 all-reduce(집합통신) · `a9_ddp_reference.py`.

---

> [!question] 🎯 왜 이걸 배우나 — 필요성 사슬 (결핍 → 필요 → 발명)
> *출발 문제(구체적 목표):* 훈련 루프(08)는 돌아가지만 CPU로는 너무 느리다. 심층망·대규모 데이터는 한 장비로 며칠이 걸리거나 아예 메모리에 안 들어간다.
> 1. CPU 직렬 연산이 느림 → 행렬곱을 대규모 병렬로 처리할 가속기 필요 → **GPU 장치(`device`)로 텐서 이동**
> 2. 가중치는 GPU·입력은 CPU면 연산 불가(RuntimeError) → 둘을 같은 장치에 모을 규칙 필요 → **`model.to(device)` + `features.to(device)` 쌍**
> 3. 코드를 cuda/mps/cpu 어디서나 돌리고 싶음 → 환경에 맞춰 분기할 패턴 필요 → **디바이스-애그노스틱(`is_available()`)**
> 4. GPU 한 장으로도 느리거나 모델이 안 들어감 → 여러 GPU에 데이터를 나눠 동시 처리 필요 → **DDP(모델 복제 + 데이터 분할)**
> 5. 복제본마다 가중치가 따로 갱신되면 발산 → grad를 모아 맞출 동기화 필요 → **all-reduce**
> **한 줄:** CPU가 느림 → GPU로 이동(`.to(device)`) → 같은 장치 규칙 지키고 → 어디서나 돌게 분기 → 여러 GPU에 데이터 나눠(DDP) grad 동기화(all-reduce).

> [!abstract] 📘 책 핵심 — 부록 A.9 GPU로 훈련 성능 최적화
> **① A.9.1 GPU 장치 계산**
> - 파이토치에서 **장치(device)** = 계산이 수행되고 데이터가 놓이는 곳. 텐서가 한 장치에 있으면 그 텐서의 연산도 *같은 장치*에서 실행.
> - `torch.cuda.is_available()` → `True`면 GPU 사용 가능.
> - `tensor.to("cuda")` 로 GPU 이동 → 결과에 `device='cuda:0'` 표시(`cuda:0`=첫 번째 GPU, `cuda:1`=두 번째…).
> - ⚠️ **모든 텐서가 동일 장치**여야: 한 텐서는 CPU·다른 텐서는 GPU면 `RuntimeError: Expected all tensors to be on the same device`.
>
> **② A.9.2 단일 GPU 훈련 (코드 A-11)** — A.7 루프에서 **3줄만** 변경
> ```python
> device = torch.device("cuda")   # 장치 변수 정의
> model.to(device)                # 모델을 GPU로
> ...
> for batch_idx, (features, labels) in enumerate(train_loader):
>     features, labels = features.to(device), labels.to(device)  # 데이터를 GPU로
> ```
> - 공유용 베스트: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
> - **참고(macOS):** Apple Silicon은 `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`. ← *책이 명시한 내 환경 코드*
> - 이 작은 예제는 전송 비용 때문에 **속도 향상 없음**. LLM 같은 심층망에서 큰 향상.
> > [!quote]- 📐 연습문제 A.4 — CPU vs GPU 행렬 곱 시간 비교
> > 주피터 `%timeit a @ b` 로 CPU·GPU 행렬 곱 시간을 재, **GPU가 빨라지기 시작하는 행렬 크기**를 찾아본다(작으면 CPU가 빠를 수 있음 = 전송 오버헤드).
>
> **③ A.9.3 다중 GPU 훈련 — DDP (코드 A-12·A-13)**
> - **분산 훈련** = 훈련을 여러 GPU·머신에 분할 → 이론상 GPU 8개면 에포크 처리 8배(통신 오버헤드 제외).
> - **DDP(DistributedDataParallel([dɪˈstrɪbjutɪd ˈdeɪtə ˈpærəlɛl], 디스트리뷰티드 데이터 패러렐, 디스트리뷰티드 데이터 패러렐)):** 입력 데이터를 여러 장치에 *분할*해 동시 처리(그림 A-12·A-13).
>   - 각 GPU가 **모델 복사본**을 가짐 → 훈련 반복마다 **다른 미니배치**(`DistributedSampler`로 겹치지 않게 분배) → 복사본마다 다른 로짓·grad → **grad를 평균·동기화**해 모든 복사본을 동일 가중치로 갱신(발산 방지).
>
> ![[그림 A-12 DDP 모델·데이터 전송.png|620]]
> ▲ 그림 A-12 — ① 각 GPU에 모델 복사본을 만들고 ② 입력을 고유 미니배치로 분할해 각 복사본에 전달.
>
> ![[그림 A-13 DDP 정방향·역전파 동기화.png|620]]
> ▲ 그림 A-13 — ③ 각 GPU가 독립적으로 forward(로짓) ④ 역전파 후 **모든 GPU의 grad를 동기화**(평균) → 모든 복사본이 동일 가중치로 갱신.
> - **핵심 함수(코드 A-12·A-13):** `init_process_group`/`destroy_process_group`(분산 시작/종료), `DistributedSampler`(배치 분배), `mp.spawn`(GPU당 프로세스 1개 생성), `DDP(model, device_ids=[rank])`.
>   - `ddp_setup`: `MASTER_ADDR`·`MASTER_PORT`(프로세스 간 통신), `backend="nccl"`(**NCCL([ˈnɪkəl], 니클, 니클) = NVIDIA 집합통신 라이브러리**, GPU↔GPU), `torch.cuda.set_device(rank)`.
>   - **`rank` = 프로세스 ID = GPU ID**, **`world_size` = 총 프로세스(GPU) 수**. `rank`는 `spawn`이 자동 주입.
> - ⚠️ DDP는 **주피터 등 인터랙티브 환경 불가** → `.py` 스크립트로. GPU 선택은 `CUDA_VISIBLE_DEVICES=0,2 python script.py`.
> - 중복 출력 방지: 각 프로세스가 독립 실행이라 출력이 GPU 수만큼 중복 → `if rank == 0:` 로 한 프로세스만 출력.
> > [!quote]- 📐 NOTE — 이 책은 다중 GPU 불필요
> > A.9.3은 *작동 방식이 궁금한 독자용*. 단일 GPU(또는 내 맥 MPS)로 책 전체를 따라갈 수 있다. 더 쉬운 다중 GPU는 Fabric 같은 애드온 API.

> 💭 **여기서 내 사고가 분기됨 ↓**

## 🧠 근본 이해 · 융합 · 통찰

### 🎯 08에서 미리 판 것이 '책 본문'으로 확인됨
[[08 훈련 루프 (Training Loop)|훈련 루프 노트]]의 🔬 하드웨어 콜아웃에서 *호스트–디바이스 전송·zero-copy·UMA* 를 미리 팠는데, A.9가 그걸 **책 코드로 공식 확인**해 준다.

| 08에서 내가 판 통찰 | A.9 책 본문의 확인 |
|---|---|
| `.to(device)` = 호스트→디바이스 전송 | "텐서를 GPU로 이동"·`device='cuda:0'` |
| 작은 데이터는 전송 비용이 이득보다 큼 | "이 예제는 전송 비용 때문에 속도 향상 없음" |
| 모든 데이터가 GPU 영토에 모여야 연산 | "모든 텐서가 동일 장치에 있어야"(RuntimeError) |
| 내 맥 = MPS(UMA) | 책 'macOS와 파이토치' 참고 박스의 `mps` 분기 |

→ 즉 0단계 마지막 장은 *새 내용*이라기보다 **앞서 세운 하드웨어 직관의 공식 코드화**. 내가 깊게 판 보람이 여기서 회수된다.

### 🧩 `model.to(device)`가 실제로 옮기는 것 — 구조가 아니라 '상태(가중치)'
> 의문: *"모델은 로직(코드) 아닌가? 왜 GPU로 '전송'하지? 학습된 데이터만 모델인가?"*

답: 모델 = **구조(로직) + 상태(가중치)의 결합체**([[09 모델 저장과 로드 (state_dict)|09의 그 '구조 vs 상태' 분리]]). 그래서 `.to(device)`가 물리적으로 옮기는 건 **구조(파이썬 클래스 코드)가 아니라, 등록된 파라미터·버퍼 텐서(=가중치)** — 곧 [[09 모델 저장과 로드 (state_dict)|09에서 저장하던 바로 그 `state_dict` 텐서들]].

| 모델의 두 조각 | `model.to(device)` 시 |
|---|---|
| **구조**(로직, `NeuralNetwork` 클래스) | 파이썬 객체 — **CPU RAM에 그대로** 남음 |
| **상태**(가중치 텐서 = `state_dict`) | **VRAM([viː ræm], 브이램, 브이램)으로 물리 이동·상주(resident([ˈrɛzɪdənt], 레지던트, 레지던트))** |

→ 연산 자체(커널)는 `forward` 때 *그 텐서가 사는 장치*에서 디스패치된다. "모델을 GPU로 보낸다" = 정확히는 **"가중치 텐서를 VRAM에 상주시킨다"**.

**같은 동전의 양면 — 09와 10:**
- **09 `state_dict` 저장** = 상태(가중치)를 *디스크에 직렬화*.
- **10 `model.to(device)`** = 같은 상태(가중치)를 *GPU 메모리에 배치*.
- 둘 다 "구조는 코드에 두고 **상태만** 다룬다"는 동일 원리. (그래서 로드 시 `NeuralNetwork(2,2)` 구조를 코드로 재생성한 뒤 상태를 주입했던 것)

**왜 `model.to`만으론 부족하고 데이터도 `.to` 해야 하나(코드 3줄의 ②+③):** 모델 가중치가 VRAM에 있으면, 입력 `features`도 **같은 장치**에 있어야 연산된다(위 '같은 장치 규칙'). 가중치만 GPU·입력은 CPU면 `RuntimeError`. → ② `model.to(device)`(상태를 GPU로) + ③ `features.to(device)`(입력도 GPU로)가 *쌍*인 이유.

> [!example] 🧭 사이드 · 비유 — `.to(device)` = 연산 회로를 가속기 위에 '조립'   `#사이드/비유`
> 추상 클래스 정의(구조)는 코드에 있고, `new`로 인스턴스를 만들 때 필드(가중치)가 메모리에 할당된다. `model.to(device)` = 그 **필드 메모리를 VRAM에 잡는 것** = 비어 있던 연산 회로에 가중치를 채워 *가속기 위에 물리적으로 조립(assembly([əˈsɛmbli], 어셈블리, 어셈블리))* 하는 행위. 그래서 `.to()`는 "데이터 넘기기"가 아니라 **"상태를 가진 객체(Stateful Object([ˈsteɪtfʊl ˈɒbdʒɪkt], 스테이트풀 오브젝트, 스테이트풀 오브젝트))를 가속기 위에 세우기"**.

### 🔑 데이터 병렬(DDP) vs 모델 병렬(샤딩) — 분할의 두 축
[[09 모델 저장과 로드 (state_dict)|모델 저장 노트]]에서 *"단일 GPU면 모델 회로를 못 쪼갠다 → 샤딩(FSDP/ZeRO)"* 이라 했다. A.9의 **DDP는 그것과 다른 축**이다. 헷갈리면 안 되는 핵심 구분:

| | **데이터 병렬 (DDP)** | **모델 병렬 / 샤딩(sharding([ˈʃɑːrdɪŋ], 샤딩, 샤딩)) (FSDP([ɛf ɛs diː piː], 에프에스디피, 에프에스디피)·ZeRO([ˈzɪəroʊ], 제로, 제로))** |
|---|---|---|
| 무엇을 나누나 | **데이터**(미니배치)를 GPU마다 분배 | **모델**(가중치·옵티마이저 상태)을 GPU마다 분할 |
| 모델은? | GPU마다 **통째 복제**(같은 모델 N개) | 한 모델을 **쪼개** 여러 GPU에 분산 |
| 언제 | 모델은 1장에 들어가는데 **데이터/속도**가 문제 | 모델이 **1장 VRAM에 안 들어갈** 때 |
| 동기화 | grad를 **all-reduce로 평균** | 분할된 파라미터/grad를 주고받음 |

→ A.9가 가르치는 건 **데이터 병렬(DDP)**. "모델이 너무 커서 한 장에 안 들어가는" 09의 문제는 *모델 병렬*이 푼다. 둘은 보완적(대규모 LLM([ɛl ɛl ɛm], 엘엘엠, 엘엘엠)은 둘을 **혼합**: 데이터+텐서+파이프라인 병렬 = 3D([θriː diː], 쓰리디, 쓰리디) 병렬).

> [!quote] 🌉 타 영역 연결 — DDP의 grad 동기화 = 분산 시스템의 '집합 통신'   `#연결/CS`
> DDP가 매 backward마다 GPU들의 grad를 평균하는 연산 = **all-reduce**(모든 노드 값을 모아 합/평균 후 전원에 재배포). 이건 AI 고유가 아니라 **분산 시스템·HPC([eɪtʃ piː siː], 에이치피시, 에이치피시)의 집합 통신(collective communication([kəˈlɛktɪv kəˌmjuːnɪˈkeɪʃən], 컬렉티브 커뮤니케이션, 컬렉티브 커뮤니케이션))** 그 자체 — MPI([ɛm piː aɪ], 엠피아이, 엠피아이)의 `MPI_Allreduce`와 같은 계보. NCCL = 그 GPU판 구현. `init_process_group`·`MASTER_ADDR`/`PORT` = 분산 노드들이 서로를 찾는 **부트스트랩(bootstrap([ˈbuːtstræp], 부트스트랩, 부트스트랩))/합의** 절차. 같은 머신이면 NVLink([ɛn viː lɪŋk], 엔브이링크, 엔브이링크), 멀티 노드면 네트워크 — [[IPC — 프로세스 간 통신 (공유 메모리·소켓·OS별)|IPC·프로세스 통신]]의 확장. 전역 다리 → [[_지식 연결 허브]]

> [!tip] 🪂 사이드 · 심화 — `DistributedSampler` = 07 셔플의 분산판   `#사이드/심화`
> [[07 데이터 로더 (Dataset · DataLoader)|데이터 로더]]의 `shuffle`은 *한 프로세스* 안에서 순서를 섞었다. DDP에선 **GPU마다 겹치지 않는 부분집합**을 줘야 해서 `shuffle`을 끄고 `sampler=DistributedSampler(...)`가 그 역할을 한다. `set_epoch(epoch)`를 매 에포크 호출해야 분산 셔플이 갱신된다(안 하면 매 에포크 같은 분배). → 07의 "반복 주기 갇힘 방지"가 분산에서도 그대로 적용.

> [!example] 🧭 사이드 · 비유 — DDP = '같은 시험을 N명이 나눠 채점'   `#사이드/비유`
> 모델 복사본 = 똑같이 훈련된 채점자 N명. 답안지(데이터)를 N등분해 각자 채점(forward/backward)하고, 끝에 **점수 기준을 한데 모아 평균**(grad 동기화)해 *모두 같은 기준으로 업데이트*. 그래서 N명이 따로 채점해도 한 명이 전부 채점한 것과 같은 결과 + N배 빠름.

## 🧩 코드 실습 (A.9)
> 🎯 목표: ① 장치 기초(`.to(device)`·같은 장치 규칙)와 단일 GPU(=내 맥 MPS) 훈련을 디바이스-애그노스틱으로 확인, ② DDP 구조를 코드로 이해(실행은 멀티GPU 환경).

**① 장치 기초 + 단일 GPU** — `~/LOCAL/03-00_STUDIES/AI/00_setup/a9_gpu_basics.py` (검증: 예측 `[0,0,0,1,1]`)
```python
def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")  # 내 맥
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()
model = NeuralNetwork(2, 2).to(device)                 # ② 모델을 장치로
for features, labels in loader:
    features, labels = features.to(device), labels.to(device)   # ③ 데이터를 장치로
    ...
# 같은 장치 규칙: cpu 텐서 + gpu 텐서 → RuntimeError
```

**② DDP (참고용)** — `~/LOCAL/03-00_STUDIES/AI/00_setup/a9_ddp_reference.py`
> ⚠️ **NVIDIA GPU 2장+ & NCCL 전용 — 내 맥(MPS)에선 미실행**(문법만 검증). 책 코드 A-12·A-13 + 공식 저장소 패턴.
```python
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"; os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)  # NCCL
    torch.cuda.set_device(rank)

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    model = NeuralNetwork(2, 2).to(rank)
    model = DDP(model, device_ids=[rank])              # grad 동기화 자동화
    # train_loader = DataLoader(..., sampler=DistributedSampler(train_ds))
    ...
    if rank == 0: print(...)                            # 중복 출력 방지
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, 3), nprocs=world_size)   # GPU당 프로세스 1개
```

관찰 포인트:
- **단일 GPU = 3줄 변경** — `device`·`model.to`·`배치.to`. 나머지는 A.7과 동일.
- **DDP는 구조가 핵심** — `init_process_group`(시작) → `DistributedSampler`(분배) → `DDP(model)`(동기화) → `mp.spawn`(프로세스 생성) → `destroy_process_group`(종료).
- **내 장비**는 `a9_gpu_basics.py`의 `mps` 분기까지가 실전, DDP는 개념 이해.

## 🆕 새 용어 정리
| 용어 | 한 줄 정의 |
|---|---|
| **장치 (device)** | 텐서가 놓이고 연산이 일어나는 곳(`cpu`·`cuda:0`·`mps`) |
| **같은 장치 규칙** | 연산에 참여하는 모든 텐서가 동일 device여야(아니면 RuntimeError) |
| **분산 훈련** | 훈련을 여러 GPU·머신에 분할(이론상 GPU N개=N배) |
| **DDP** (DistributedDataParallel) | 모델 복제+데이터 분할로 동시 훈련하는 **데이터 병렬** |
| **데이터 병렬 vs 모델 병렬** | 데이터를 나눔(DDP) vs 모델을 나눔(샤딩/FSDP) — 다른 축 |
| **`DistributedSampler`** | 각 프로세스(GPU)에 겹치지 않는 배치 부분집합 분배 |
| **NCCL** | NVIDIA 집합통신 라이브러리(GPU↔GPU all-reduce 등) |
| **all-reduce** | 모든 노드 값을 모아 합/평균 후 전원에 재배포(grad 동기화) |
| **`rank` / `world_size`** | 프로세스 ID(=GPU ID) / 총 프로세스(GPU) 수 |
| **`init_process_group`** | 분산 프로세스 그룹 초기화(부트스트랩) |
| **`mp.spawn`** | GPU당 프로세스 1개를 생성해 함수 병렬 실행 |
| **`CUDA_VISIBLE_DEVICES`** | 스크립트 수정 없이 쓸 GPU를 환경변수로 제한 |

## 막힌 점 · 다시 볼 것
- [x] 장치 기초 + 단일 GPU(MPS) 디바이스-애그노스틱 훈련 (A.9.1~.2) → `00_setup/a9_gpu_basics.py`
- [ ] 🔴 **DDP(A.9.3) 재학습** — 지금은 구조만 대충 훑음. 분산이 실제 필요할 때 정독 → `00_setup/a9_ddp_reference.py` (위 🔖 복습 지점 참조)
- [ ] 연습문제 A.4 — `%timeit`로 CPU vs MPS 행렬 곱 크로스오버 크기 측정 (내 맥에서)
- [ ] 데이터+모델+파이프라인 **3D 병렬**(대규모 LLM 실전) — 3~4단계

## 📌 암기 카드
> 앞면(cue)을 보고 뒷면을 떠올린 뒤 확인. (→ [[_핵심 암기장]])

| 앞면 (cue) | 뒷면 (외울 내용) |
|---|---|
| 장치(device)란? | 텐서가 놓이고 연산이 일어나는 곳(`cpu`·`cuda:0`·`mps`) |
| 같은 장치 규칙 | 연산에 참여하는 모든 텐서가 동일 device여야(아니면 RuntimeError) |
| 단일 GPU 훈련 = 코드 3줄 | ① `device = torch.device(...)` ② `model.to(device)` ③ `features, labels = features.to(device), labels.to(device)` |
| 디바이스-애그노스틱 패턴 | `torch.device("cuda" if torch.cuda.is_available() else "cpu")`(맥은 `"mps"` 분기) |
| `model.to(device)`가 옮기는 것 | 구조(클래스 코드)가 아니라 상태(가중치 텐서=state_dict)를 VRAM에 상주 |
| DDP란? | 모델을 GPU마다 복제 + 데이터를 분할(DistributedSampler) → 각자 forward/backward → grad all-reduce 동기화 = 데이터 병렬 |
| 데이터 병렬 vs 모델 병렬 | DDP=데이터(미니배치)를 나눔 / 샤딩(FSDP·ZeRO)=모델(가중치·옵티마이저 상태)을 나눔 |
| all-reduce | 모든 노드 grad를 모아 합/평균 후 전원에 재배포(grad 동기화) |
| `rank` / `world_size` | rank=프로세스 ID(=GPU ID) / world_size=총 프로세스(GPU) 수 |
| DDP 핵심 함수 | `init_process_group`(시작) → `DistributedSampler`(분배) → `DDP(model)`(동기화) → `mp.spawn`(프로세스 생성) → `destroy_process_group`(종료) |

## 연결
- 이전: [[09 모델 저장과 로드 (state_dict)|모델 저장과 로드]] ｜ 다음: (1단계 — 선형대수 예정)
- GPU 하드웨어 원리(왜 빠른가·SIMT·텐서코어·ARM 관점): [[GPU 하드웨어 구조 — SIMT·SM·텐서코어 (ARM 관점)|GPU 하드웨어 구조]] (CS)
- 하드웨어 토대(미리 판 통찰의 회수): [[08 훈련 루프 (Training Loop)|훈련 루프 🔬 하드웨어 콜아웃]]
- 데이터 분배 토대: [[07 데이터 로더 (Dataset · DataLoader)|DataLoader]] (→ DistributedSampler)
- 모델 병렬 대비: [[09 모델 저장과 로드 (state_dict)|샤딩(FSDP/ZeRO)]] · GPU간 통신 원리: [[IPC — 프로세스 간 통신 (공유 메모리·소켓·OS별)|IPC]]
- 단계: [[_0단계 - 닻 올리기]]
