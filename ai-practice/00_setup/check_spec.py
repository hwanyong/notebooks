"""[밑바닥LLM] 부록 A.1 — 가속(GPU) 디바이스 확인

책 본문은 NVIDIA GPU 기준으로 `torch.cuda.is_available()` 를 사용한다.
하지만 Apple Silicon(M1·M2·M3…) 맥에는 CUDA가 없고, 대신 **MPS(Metal
Performance Shaders)** 백엔드로 GPU 가속을 확인한다 → `torch.backends.mps.is_available()`.
(주의: `mps`이지 `map`이 아니다. 오타 시 AttributeError.)

실행:
    python 00_setup/check_spec.py
"""
import torch

print("torch          :", torch.__version__)
print("CUDA  사용 가능 :", torch.cuda.is_available())          # NVIDIA GPU → 맥에선 False
print("MPS   사용 가능 :", torch.backends.mps.is_available())  # Apple Silicon GPU → True

# 환경에 맞춰 실제 사용할 디바이스 자동 선택 (앞으로 .to(device) 로 재사용)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("선택된 디바이스 :", device)

# 간단 동작 확인: 텐서를 선택된 디바이스에 올려본다
x = torch.rand(3, device=device)
print("샘플 텐서       :", x)
