"""[밑바닥LLM] 부록 A.4 — 자동 미분(autograd)으로 그레이디언트 구하기

a3_visualize_graph.py 에서 만든 forward 그래프를 그대로 이어받아,
역전파(연쇄 법칙)로 가중치별 그레이디언트를 두 가지 방법으로 구한다.

  방법 1) torch.autograd.grad(...)  : 원하는 텐서의 그레이디언트만 콕 집어 계산 (수동/디버깅용)
  방법 2) loss.backward()           : 그래프의 모든 leaf 노드 그레이디언트를 .grad 에 자동 저장 (실전용)

두 방법의 결과가 같음을 확인한다.  (∂L/∂w1 ≈ -0.0898, ∂L/∂b ≈ -0.0817)

실행:
    python 00_setup/a4_autograd.py
"""
import torch
import torch.nn.functional as F
from torch.autograd import grad

y  = torch.tensor([1.0])                       # 정답 레이블
x1 = torch.tensor([1.1])                       # 입력 특성
w1 = torch.tensor([2.2], requires_grad=True)   # 가중치 → 그레이디언트 계산 대상
b  = torch.tensor([0.0], requires_grad=True)   # 편향   → 그레이디언트 계산 대상

# --- forward pass (계산 그래프 기록) ---
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

# === 방법 1: grad() 로 원하는 파라미터만 계산 ===
# retain_graph=True : grad()는 기본적으로 계산 후 그래프를 지운다.
#                     아래에서 그래프를 한 번 더 쓰므로 유지시킨다.
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b  = grad(loss, b,  retain_graph=True)
print("[grad()]      dL/dw1 =", grad_L_w1)   # (tensor([-0.0898]),)
print("[grad()]      dL/db  =", grad_L_b)    # (tensor([-0.0817]),)

# === 방법 2: backward() 로 한 번에 모든 leaf 의 .grad 채우기 ===
loss.backward()
print("[backward()]  w1.grad =", w1.grad)    # tensor([-0.0898])
print("[backward()]  b.grad  =", b.grad)     # tensor([-0.0817])

# 두 방법 결과 일치 확인
assert torch.allclose(grad_L_w1[0], w1.grad)
assert torch.allclose(grad_L_b[0],  b.grad)
print("\nOK: grad() 와 backward() 결과가 일치한다.")
