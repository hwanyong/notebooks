---
유형: 탐구
코스: AI 암반 정복
유래: "A.6 DataLoader(07)의 num_workers·pin_memory에서 성능으로 파생 — PyTorch에 적용하는 실전 지식. 이후 텍스트·이미지·3D 등 모든 데이터 로딩에 재사용"
이해도: 🟡
태그:
  - 탐구
  - pytorch
  - 데이터로더
  - 성능최적화
---
# PyTorch([ˈpaɪtɔːrtʃ], 파이토치, 파이토치) 데이터 로딩 성능 최적화

> CS([siː ɛs], 시에스, 시에스)의 메모리·캐시·멀티프로세스 **원리**를 PyTorch `DataLoader`·`Dataset`에서 *어떻게 실현*하나 — prefetch([ˈpriːfetʃ], 프리페치, 프리페치)·persistent_workers·캐시 친화 Dataset([ˈdeɪtəset], 데이터셋, 데이터세트)·pin_memory. (책 A.6의 `num_workers`를 넘어선 실전 파생)

> 🌱 **탐구 노트(PyTorch 실전)** — 책 절이 아니라 성능을 파고든 조사. 특정 챕터에 안 묶이고, 어떤 모델 학습의 데이터 로딩에도 재사용.
> 🔗 관련 지점: [[07 데이터 로더 (Dataset · DataLoader)|A.6 데이터 로더]] (유래·적용 대상) · ⚙️ 원리: [[메모리·캐시·멀티프로세스 (시스템 원리)|CS/메모리·캐시·멀티프로세스]] ｜ 🗂️ [[_탐구 MOC]] · 🌐 [[_지식 연결 허브]]

> 🎯 **핵심 (TL;DR([tiː ɛl diː ɑːr], 티엘디알, 티엘디알))**
> - **로딩 병목 숨기기:** `num_workers>0` + `prefetch_factor`(미리 큐에 채움) + `persistent_workers`(워커·캐시 생존).
> - **캐시 친화 `Dataset`:** 데이터를 연속 텐서로·작은 dtype으로 보관, `__getitem__`은 벡터화·read-only([ˌriːd ˈoʊnli], 리드 오운리, 리드온리).
> - **GPU([dʒiː piː juː], 지피유, 지피유) 전송 가속:** `pin_memory=True` → DMA([diː ɛm eɪ], 디엠에이, 디엠에이)로 RAM([ræm], 램, 램)→VRAM([ˈviːræm], 브이램, 브이램) 비동기 고속 복사.
> - *왜* 그런지는 전부 → [[메모리·캐시·멀티프로세스 (시스템 원리)|CS 원리 노트]].

> [!note] 🗺️ 이 노트 읽는 법
> 책 A.6은 `num_workers`까지만 다룬다. 그 너머 *"데이터 로딩을 어떻게 더 빠르게?"* 를 파고든 PyTorch 실전 파생. **원리(WHY)는 CS 노트, 적용(HOW)은 여기.** → 출처 가지: [[07 데이터 로더 (Dataset · DataLoader)]].

---

## 1. 로딩 병목을 숨기는 DataLoader([ˈdeɪtəˌloʊdər], 데이터로더, 데이터로더) 옵션
**`num_workers`** (기본) — 데이터 로드를 워커 프로세스로 병렬화. (원리: [[메모리·캐시·멀티프로세스 (시스템 원리)|멀티프로세스·GIL]])

**`prefetch_factor`** — 각 워커가 *현재 배치 외에* 앞으로 쓸 배치를 미리 만들어 큐에 쌓아 둠(기본 2). GPU가 한 배치를 끝내자마자 큐에서 즉시 낚아챔 → 기다림 0. (원리: 룩어헤드 prefetch)

**`persistent_workers=True`** — 기본은 매 에포크 끝에 워커가 죽는다(다음 에포크에 재생성 오버헤드 + 내부 캐시 소멸). 켜면 **워커가 살아남아** 재생성 비용·캐시 손실을 없앤다.

```python
loader = DataLoader(
    ds, batch_size=64, shuffle=True,
    num_workers=4,            # 병렬 로드 프로세스
    prefetch_factor=2,        # 워커당 미리 준비할 배치 수
    persistent_workers=True,  # 워커를 에포크 간 유지
    pin_memory=True,          # ↓ 3번
)
```

## 2. 명시적 RAM 캐시 — `Dataset` 내부에 직접
데이터셋 전체가 RAM보다 작으면, 처음 읽은 가공 결과를 멤버에 저장해 디스크 재접근을 없앤다. (`persistent_workers=True`와 짝이어야 캐시가 에포크 간 유지됨)

```python
class RAMCachedDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.cache = [None] * len(file_paths)
    def __getitem__(self, index):
        if self.cache[index] is not None:        # 캐시 히트 → 디스크 스킵
            return self.cache[index]
        data = self.load_and_preprocess(self.file_paths[index])  # 최초 1회만
        self.cache[index] = data
        return data
    def __len__(self):
        return len(self.file_paths)
```
> 참고: OS가 자동으로 해 주는 **페이지 캐시**(2에포크부터 디스크 대신 RAM)도 별도로 깔려 있다 → [[메모리·캐시·멀티프로세스 (시스템 원리)|페이지 캐시]].

## 3. 캐시 친화적인 `Dataset` 작성 4원칙
> CS 캐시 원리를 PyTorch `Dataset`에 적용한 코딩 습관. (각 *왜*는 → [[메모리·캐시·멀티프로세스 (시스템 원리)|CS 원리]])

| 습관 | 나쁨 ❌ | 좋음 ✅ | 깔린 원리 |
|---|---|---|---|
| **연속 메모리** | `self.data`를 파이썬 리스트/딕셔너리로 | 하나의 **텐서/`np.array`** 로 묶어 보관 | C-contiguous([siː kənˈtɪɡjuəs], 시 컨티규어스, 시 콘티규어스)(Pointer Chasing([ˈpɔɪntər ˈtʃeɪsɪŋ], 포인터 체이싱, 포인터 체이싱) 회피) |
| **dtype 다이어트** | 레이블을 `int64`, 실수를 `float64` | `int8`/`int16`, `float32`/`float16` | 캐시 라인 밀도↑ |
| **`__getitem__` 경량화** | 안에서 파이썬 `for`·`if`로 파싱 | 벡터화 연산 / 무거운 전처리는 `__init__`에서 미리(pre-compute([ˌpriːkəmˈpjuːt], 프리컴퓨트, 프리컴퓨트)) | 공간 지역성 |
| **read-only 반환** | `self.some_list[i] = 새값` (공용 멤버 수정) | 새 **지역 변수**를 만들어 반환 | False Sharing([fɔːls ˈʃerɪŋ], 폴스 셰어링, 폴스 셰어링) 회피 |

- 특히 마지막: `num_workers>0`이면 워커는 **독립 프로세스**라 공용 데이터를 수정하면 IPC([aɪ piː siː], 아이피시, 아이피시)·캐시 충돌이 난다. `Dataset` 데이터는 **읽기 전용**, 가공물은 지역 변수로. (원리: [[메모리·캐시·멀티프로세스 (시스템 원리)|False Sharing·프로세스 격리]])

## 4. GPU 전송 가속 — `pin_memory`
**`pin_memory=True`** — 워커가 만든 배치를 **page-lock([peɪdʒ lɒk], 페이지 록, 페이지 록)된(고정) RAM**에 올린다. 그러면 GPU가 CPU([siː piː juː], 시피유, 시피유) 개입 없이 **DMA**로 VRAM에 비동기 고속 복사 → CPU는 다음 배치 준비에 집중. (원리: [[메모리·캐시·멀티프로세스 (시스템 원리)|DMA]])
- 보통 `.to(device, non_blocking=True)`와 함께 써서 전송·연산을 겹친다.

## 5. OS([oʊ ɛs], 오에스, 오에스)별 멀티프로세싱 주의 — `num_workers`는 OS마다 다르게 돈다
`torch.multiprocessing`이 OS를 감지해 추상화하지만 내부 동작이 달라, 디버깅·배포 때 알아야 한다. (원리 → [[IPC — 프로세스 간 통신 (공유 메모리·소켓·OS별)|IPC·OS별 구현]])
- **Linux([ˈlɪnəks], 리눅스, 리눅스):** `fork` + `/dev/shm` — 빠르고 기본값으로 잘 돈다.
- **macOS([ˈmækoʊɛs], 맥오에스, 맥오에스):** `spawn` 강제 + 커널 공유 메모리 용량이 작아 대용량에서 `RuntimeError: received 0 items of ancdata`. 해결: `torch.multiprocessing.set_sharing_strategy('file_system')`.
- **Windows([ˈwɪndoʊz], 윈도우즈, 윈도):** `spawn` — 워커 진입점 보호를 위해 **`if __name__ == '__main__':` 가드 필수**(없으면 무한 프로세스 생성). 워커 생성 오버헤드가 커 `persistent_workers=True` 권장.

## 🆕 새 용어 정리 (PyTorch 옵션)
| 토큰 | 한 줄 정의 |
|---|---|
| `prefetch_factor` | 워커당 미리 큐에 채워 둘 배치 수(룩어헤드, 기본 2) |
| `persistent_workers` | 워커 프로세스를 에포크 간 살려 둬 재생성·캐시 손실 방지 |
| `pin_memory` | 배치를 page-lock RAM에 올려 DMA로 GPU에 비동기 고속 복사 |
| `non_blocking=True` | `.to(device)`를 비동기로 — 전송·연산 겹치기 |
| RAM 캐시 (`Dataset`) | 처음 읽은 가공 결과를 멤버에 저장해 디스크 재접근 제거 |

## 막힌 점 · 다시 볼 것
- [ ] `num_workers`·`prefetch_factor`를 바꿔가며 1에포크 시간 측정(최적값 탐색)
- [ ] `pin_memory` on/off로 GPU 전송 시간 비교
- [ ] 이미지 폴더 Dataset에서 4원칙(연속·dtype·벡터화·read-only) 적용 전후 비교

## 📌 암기 카드
> 앞면(cue)을 보고 뒷면을 떠올린 뒤 확인. (→ [[_핵심 암기장]])

| 앞면 (cue) | 뒷면 (외울 내용) |
|---|---|
| 로딩 병목 숨기는 3옵션? | num_workers>0 + prefetch_factor + persistent_workers |
| prefetch_factor 역할? | 워커당 미리 큐에 채워 둘 배치 수(룩어헤드, 기본 2) |
| persistent_workers=True 효과? | 워커를 에포크 간 살려 둬 재생성·캐시 손실 방지 |
| pin_memory=True 효과? | 배치를 page-lock RAM에 올려 DMA로 GPU에 비동기 고속 복사 |
| 캐시 친화 Dataset 4원칙? | 연속 메모리·dtype 다이어트·__getitem__ 경량화·read-only 반환 |
| 무거운 전처리는 어디서? | __init__에서 미리(pre-compute), __getitem__이 아님 |
| non_blocking=True 용도? | .to(device)를 비동기로 — 전송·연산 겹치기 |
| macOS num_workers 에러 해결? | set_sharing_strategy('file_system') |
| Windows num_workers 필수 가드? | if __name__ == '__main__': |

## 🔗 연결
- 유래·적용 대상: [[07 데이터 로더 (Dataset · DataLoader)|A.6 데이터 로더]]
- ⚙️ 원리(WHY): [[메모리·캐시·멀티프로세스 (시스템 원리)|CS/메모리·캐시·멀티프로세스]]
- 단계: [[_0단계 - 닻 올리기]] · 🗂️ [[_탐구 MOC]] · 🌐 [[_지식 연결 허브]]
