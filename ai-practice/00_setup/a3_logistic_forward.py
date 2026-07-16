"""[밑바닥LLM] 부록 A.3 — 모델을 계산 그래프로 보기

코드 A-2: 로지스틱 회귀 분류기의 정방향 계산(forward pass).
한 개의 층을 가진 신경망으로, 0~1 사이 점수를 반환하고 정답 레이블(0/1)과
비교해 손실(loss)을 계산한다. autograd는 이 forward 과정을 동적 계산 그래프로
기록해 두었다가, 역전파(backprop)에서 그레이디언트를 자동 계산한다.

실행:
    python 00_setup/a3_logistic_forward.py
"""
import torch
import torch.nn.functional as F  # 파이토치에서 코드를 짧게 하려고 즐겨 쓰는 임포트

y  = torch.tensor([1.0])   # 정답 레이블
x1 = torch.tensor([1.1])   # 입력 특성
w1 = torch.tensor([2.2])   # 가중치 파라미터
b  = torch.tensor([0.0])   # 편향 유닛

z = x1 * w1 + b            # 순 입력(net input)
a = torch.sigmoid(z)       # 활성화 함수와 출력

loss = F.binary_cross_entropy(a, y)

print("z (net input):", z)
print("a (output)   :", a)
print("loss         :", loss)
