"""[밑바닥LLM] 부록 A.5 — 2개의 은닉층을 가진 다층 퍼셉트론(MLP)

핵심 구성 요소:
  - nn.Module    : 모든 신경망의 부모 클래스. __init__(층 정의) + forward(순전파) 두 메서드로 구성.
  - nn.Sequential: 여러 층을 순서대로 쌓아 한 묶음으로 실행.
  - nn.Linear(in, out) : 완전연결층 = (가중치 W·입력 + 편향 b). 내부에 W·b 파라미터를 자동 보유.
  - nn.ReLU()    : 은닉층 사이의 비선형 함수. 이게 있어야 공간을 '꺾어' XOR 같은 비선형 문제를 푼다.
  - logits       : 마지막 층의 (softmax 전) 날 출력값.

입력 특성(num_inputs)과 클래스(num_outputs) 개수를 변수로 받아 데이터셋이 달라도 재사용.

실행:
    python 00_setup/a5_neural_network.py
"""
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            # 1번째 은닉층: num_inputs -> 30
            nn.Linear(num_inputs, 30),
            nn.ReLU(),
            # 2번째 은닉층: 30 -> 20  (한 층의 출력 수 = 다음 층의 입력 수)
            nn.Linear(30, 20),
            nn.ReLU(),
            # 출력층: 20 -> num_outputs
            nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)   # 마지막 층의 출력 = 로짓(logits)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    # 디바이스 자동 선택 (Apple Silicon이면 mps)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # 예: 입력 특성 50개, 분류 클래스 3개
    model = NeuralNetwork(num_inputs=50, num_outputs=3).to(device)
    print(model)

    # 학습 가능한 파라미터(가중치+편향) 총 개수
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n학습 가능 파라미터 수: {n_params:,}")
    # 검산: 50*30+30 + 30*20+20 + 20*3+3 = 1530 + 620 + 63 = 2213

    # 더미 입력 1개 배치(샘플 4개)로 순전파
    x = torch.rand(4, 50, device=device)   # (batch=4, features=50)
    logits = model(x)
    print("\n입력 shape :", tuple(x.shape))
    print("출력 shape :", tuple(logits.shape))   # (4, 3) = 샘플 4개 × 클래스 3
    print("첫 샘플 로짓:", logits[0].tolist())
