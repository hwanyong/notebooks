"""
부록 A.8 — 모델 저장과 로드 (state_dict)
[밑바닥LLM] 부록 A.8.

훈련한 모델을 디스크에 저장했다가 다시 불러와 같은 예측이 나오는지 확인한다.
책 추천 방식 = '모델 전체'가 아니라 학습된 상태(가중치·편향)만 담은 state_dict 저장.

a7_training_loop_mps.py에서 학습한 뒤 → 저장 → 새 모델에 로드 → 예측 일치 검증.
실행: python a8_save_load.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])

PATH = "model.pth"


def train(device):
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(ToyDataset(X_train, y_train), batch_size=2, shuffle=True)
    for _ in range(3):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            loss = F.cross_entropy(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict(model, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_train.to(device))
    return torch.argmax(logits, dim=1).tolist()


def main():
    device = pick_device()
    print(f"사용 디바이스: {device}")

    # 1) 학습
    model = train(device)
    before = predict(model, device)
    print("저장 전 예측:", before)

    # 2) 저장 — state_dict(층↔파라미터 매핑 딕셔너리)만 디스크로
    torch.save(model.state_dict(), PATH)
    print(f"state_dict 키: {list(model.state_dict().keys())}")
    print(f"파라미터 총 개수: {sum(p.numel() for p in model.parameters())}")  # 752

    # 3) 로드 — 먼저 '동일 구조'의 빈 모델을 만들고(구조=코드), 거기에 상태(데이터)를 주입
    loaded = NeuralNetwork(2, 2)                       # 구조가 원본과 정확히 일치해야 함
    # map_location: GPU에서 저장한 텐서를 다른 디바이스(cpu/mps)로 안전하게 매핑
    # weights_only=True: 임의 객체 역직렬화(pickle) 위험 차단 — 순수 텐서만 로드(권장)
    state = torch.load(PATH, map_location=device, weights_only=True)
    loaded.load_state_dict(state)
    loaded.to(device)

    after = predict(loaded, device)
    print("로드 후 예측:", after)
    print("일치 여부:", before == after)


if __name__ == "__main__":
    main()
