"""
부록 A.9.1~A.9.2 — GPU 장치 계산 & 단일 GPU 훈련 (디바이스-애그노스틱)
[밑바닥LLM] 코드 A-11 주변.

A.9의 핵심: "텐서가 놓인 '장치(device)'에서 연산이 일어난다. 모든 텐서가 같은 장치에
있어야 한다." → 훈련 루프를 GPU로 보내는 건 코드 3줄(device 정의 · model.to · 배치.to).

내 장비(Apple Silicon)는 cuda 대신 mps. 그래서 device-agnostic으로 짜면 코드 한 벌로
cuda·mps·cpu 어디서나 돈다(책 'macOS와 파이토치' 참고 박스와 동일).
실행: python a9_gpu_basics.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def pick_device():
    if torch.backends.mps.is_available():     # Apple Silicon GPU(Metal)
        return torch.device("mps")
    if torch.cuda.is_available():             # NVIDIA
        return torch.device("cuda")
    return torch.device("cpu")


# ── A.9.1: 장치(device) 기초 ──────────────────────────────────
def device_basics():
    device = pick_device()
    print(f"[A.9.1] is_available → mps:{torch.backends.mps.is_available()} "
          f"cuda:{torch.cuda.is_available()} → 사용: {device}")

    # 텐서를 장치로 이동(.to) — 연산은 그 장치에서 일어난다
    t1 = torch.tensor([1., 2., 3.]).to(device)
    t2 = torch.tensor([4., 5., 6.]).to(device)
    print("[A.9.1] 같은 장치 덧셈:", (t1 + t2))   # device=가 붙어 출력(cuda/mps면)

    # ⚠️ '같은 장치' 규칙 위반 시 RuntimeError 재현(설명용)
    if device.type != "cpu":
        t_cpu = torch.tensor([7., 8., 9.])         # cpu에 남김
        try:
            _ = t1 + t_cpu                          # 서로 다른 장치 → 에러
        except RuntimeError as e:
            print("[A.9.1] 다른 장치 연산 차단:", str(e).splitlines()[0])


# ── A.9.2: 단일 GPU(=내 장비 mps) 훈련 ────────────────────────
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


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features, self.labels = X, y

    def __getitem__(self, i):
        return self.features[i], self.labels[i]

    def __len__(self):
        return self.labels.shape[0]


X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])


def single_device_training():
    device = pick_device()                                  # ① 장치 변수
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2).to(device)                  # ② 모델을 장치로
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(ToyDataset(X_train, y_train), batch_size=2, shuffle=True)

    for epoch in range(3):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)  # ③ 데이터를 장치로
            loss = F.cross_entropy(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[A.9.2] epoch {epoch+1} | loss {loss:.2f}")

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_train.to(device)), dim=1)
    print("[A.9.2] 예측:", preds.tolist(), "| 정답:", y_train.tolist())


if __name__ == "__main__":
    device_basics()
    print("-" * 40)
    single_device_training()
