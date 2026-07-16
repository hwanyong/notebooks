"""[밑바닥LLM] 부록 A.3~A.4 — 계산 그래프를 "눈으로" 확인하기

a3_logistic_forward.py 의 forward pass에, 계산 그래프의 흔적을 두 가지 방법으로 확인.
  (1) grad_fn  : 각 텐서가 "어떤 연산으로 태어났는지" 기록한 텍스트 흔적
  (2) torchviz : 계산 그래프를 실제 박스·화살표 그림(.png)으로 시각화

⚠️ 핵심: 계산 그래프는 **누군가 requires_grad=True 일 때만** 만들어진다.
   (학습 대상인 w1, b 를 requires_grad=True 로 둬야 그래프가 기록됨)

설치(시각화용, 선택):
    pip install torchviz
    brew install graphviz      # torchviz가 그림을 그릴 때 쓰는 'dot' 실행기

실행:
    python 00_setup/a3_visualize_graph.py
"""
import torch
import torch.nn.functional as F

y  = torch.tensor([1.0])                       # 정답 레이블
x1 = torch.tensor([1.1])                       # 입력 특성
w1 = torch.tensor([2.2], requires_grad=True)   # 가중치 → 학습 대상이라 그래프에 포함
b  = torch.tensor([0.0], requires_grad=True)   # 편향   → 학습 대상이라 그래프에 포함

z = x1 * w1 + b        # 순 입력(net input)
a = torch.sigmoid(z)   # 활성화 함수와 출력
loss = F.binary_cross_entropy(a, y)

# (1) 텍스트로 그래프 흔적 확인 — grad_fn 을 타고 거꾸로 가면 역전파 경로
print("loss.grad_fn:", loss.grad_fn)            # <BinaryCrossEntropyBackward0 ...>
print("a.grad_fn   :", a.grad_fn)               # <SigmoidBackward0 ...>
print("z.grad_fn   :", z.grad_fn)               # <AddBackward0 ...>

# (2) torchviz 로 그래프 그림 저장 (설치돼 있을 때만)
try:
    from torchviz import make_dot
    dot = make_dot(loss, params={"w1": w1, "b": b, "x1": x1, "y": y})
    dot.render("00_setup/logistic_graph", format="png", cleanup=True)
    print("그래프 이미지 저장 완료 → 00_setup/logistic_graph.png")
except ImportError:
    print("torchviz 미설치 — `pip install torchviz` (+ `brew install graphviz`) 후 다시 실행")
