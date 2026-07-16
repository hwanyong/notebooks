"""
부록 A.7 — 훈련 루프: Apple Silicon (MPS / UMA) 최적화 버전
[밑바닥LLM] 코드 A-9 ~ A-10 을 내 장비(Apple Silicon, 통합 메모리)에 맞춰 정리.

a7_training_loop.py(기본/디바이스 비지정)와의 차이 = '하드웨어 토폴로지'에 대한 대응:
  - x86 + NVIDIA 외장 GPU : RAM ↔ VRAM 이 PCIe 로 분리 → 전송이 병목.
                            그래서 pin_memory=True(page-lock+DMA) · 무거운 num_workers 로
                            전송 병목을 '숨기는' 코드가 필요했다.
  - Apple Silicon (UMA)   : CPU·GPU 가 한 SoC 의 RAM 을 통합 공유 → .to(device) 가
                            PCIe 데이터 이주가 아니라 사실상 포인터 바인딩(zero-copy).
                            전송 병목이 구조적으로 없으니 pin_memory 가 무의미.

⚠️ 정확성 보강 (흔한 오해):
  num_workers 는 '전송' 병목이 아니라 '데이터 준비(디스크 I/O + CPU 전처리)' 병렬화다.
  UMA(zero-copy)가 없애는 건 '전송 복사'뿐, '데이터 준비 비용'은 그대로다.
  → 무거운 __getitem__(이미지 디코딩·증강 등)이면 *맥에서도* num_workers>0 가 여전히 유효.
  이 예제가 num_workers=0 인 진짜 이유 = "UMA라서"가 아니라
  "데이터가 이미 메모리에 올라온 초소형 토이라서"(워커 오버헤드 > 이득).

실행: python a7_training_loop_mps.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── 05: 신경망 ────────────────────────────────────────────────
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 30), nn.ReLU(),
            nn.Linear(30, 20), nn.ReLU(),
            nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


# ── 07: 데이터셋 / 로더 ───────────────────────────────────────
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, i):
        return self.features[i], self.labels[i]

    def __len__(self):
        return self.labels.shape[0]


# ── 디바이스 선택 (핵심) ──────────────────────────────────────
# 디바이스-애그노스틱 패턴: 코드 한 벌로 mps(맥 GPU)·cuda·cpu 어디서나 돈다.
# 우선순위: Apple Silicon GPU(mps) → 없으면 cpu. (NVIDIA면 "cuda"로 분기 추가)
def pick_device():
    if torch.backends.mps.is_available():     # Apple Silicon GPU(Metal)
        return torch.device("mps")
    if torch.cuda.is_available():             # NVIDIA(참고용 분기)
        return torch.device("cuda")
    return torch.device("cpu")


# 코드 A-5: 예시 데이터 (float32 기본 — MPS는 float64 미지원이라 float32가 안전)
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# UMA 최적화 포인트:
#   pin_memory 생략(=False) — 통합 메모리라 GPU가 같은 RAM을 보므로 page-lock 무의미.
#   num_workers=0 — 단, 이건 데이터가 초소형 메모리 텐서라서지 UMA라서가 아님(위 ⚠️).
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)


# ── 코드 A-10: 정확도 함수 (디바이스 인자 추가) ──────────────────
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0.0, 0
    for features, labels in dataloader:
        # UMA: 이 .to(device)는 PCIe 이주가 아니라 통합 메모리 내 포인터 바인딩(zero-copy)
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total += len(compare)
    return (correct / total).item()


def main():
    device = pick_device()
    print(f"사용 디바이스: {device}")

    # ── 코드 A-9: 훈련 루프 ──────────────────────────────────
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)  # 회로(가중치)를 GPU 영토에 상주
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            # NVIDIA였다면 여기서 매번 PCIe 복사 오버헤드 / UMA는 zero-copy 포인터 전환
            features, labels = features.to(device), labels.to(device)
            logits = model(features)                 # ① forward
            loss = F.cross_entropy(logits, labels)   # ② 손실(softmax 내장)
            optimizer.zero_grad()                    # ③ grad 초기화
            loss.backward()                          # ④ 역전파
            optimizer.step()                         # ⑤ 가중치 갱신
            print(f"에포크 {epoch+1:03d}/{num_epochs:03d}"
                  f" | 배치 {batch_idx:03d}/{len(train_loader):03d}"
                  f" | 훈련 손실 {loss:.2f}")
        model.eval()

    # ── 예측 & 확률 ─────────────────────────────────────────
    torch.set_printoptions(sci_mode=False)
    model.eval()
    with torch.no_grad():
        outputs = model(X_train.to(device))          # 입력도 같은 디바이스로
    probas = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probas, dim=1)
    print("\n예측:", predictions.tolist(), "| 정답:", y_train.tolist())

    # ── 정확도 ─────────────────────────────────────────────
    print("train 정확도:", compute_accuracy(model, train_loader, device))
    print("test  정확도:", compute_accuracy(model, test_loader, device))


if __name__ == "__main__":
    main()
