"""[밑바닥LLM] 부록 A.5 — 모델 파라미터 확인 & 정방향 계산(forward)

a5_neural_network.py 의 NeuralNetwork(2개 은닉층 MLP)를 가져와:
  1) 파라미터(가중치·편향) 모양 확인
  2) forward 실행 → grad_fn(AddmmBackward0) 관찰
  3) torch.no_grad() 로 추론(역전파 추적 OFF)
  4) softmax 로 로짓 → 클래스 확률 변환

실행 (AI 루트에서, venv 활성화 상태):
    python 00_setup/a5_model_usage.py
"""
import torch
from a5_neural_network import NeuralNetwork   # 같은 00_setup 폴더의 모델 재사용

torch.manual_seed(123)                         # 재현성: 매번 같은 초기 가중치
model = NeuralNetwork(num_inputs=50, num_outputs=3)

# 1) 파라미터 모양 확인 ---------------------------------------------------
print(model)                                                   # 층 구조
print("weight.shape :", model.layers[0].weight.shape)          # torch.Size([30, 50]) = (출력 30 × 입력 50) 2D 행렬
print("bias.shape   :", model.layers[0].bias.shape)            # torch.Size([30])     = 1D 편향 벡터
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("학습 가능 파라미터 수:", f"{n_params:,}")

# 2) 정방향 계산(forward) -------------------------------------------------
X = torch.rand((1, 50))          # 50차원 특성 샘플 1개
out = model(X)                   # model(X) == model.forward(X) 자동 호출
print("\nout      :", out)       # grad_fn=<AddmmBackward0>  (Addmm = 행렬곱 mm 후 덧셈 add = Wx+b)
print("out.shape:", tuple(out.shape))   # (1, 3) = 샘플 1 × 클래스 3 (로짓)

# 3) 추론 전용: 역전파 추적 끄기 -----------------------------------------
with torch.no_grad():            # grad 계산 안 함 → 메모리·속도 절약
    out_infer = model(X)
print("\nno_grad out:", out_infer)       # grad_fn 없음 (그래프 미생성)

# 4) 로짓 → 확률: softmax -------------------------------------------------
with torch.no_grad():
    probs = torch.softmax(model(X), dim=1)
print("\nprobs:", probs)
print("확률 합:", round(probs.sum().item(), 4))   # ≈ 1.0 (랜덤 초기화라 세 클래스가 비슷)
