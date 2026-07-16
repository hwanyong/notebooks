"""[밑바닥LLM] 부록 A.2 — 텐서 이해하기

목표: 텐서를 만들고(rank), dtype을 확인·변환하고,
      모양 연산(shape / reshape / view / T / matmul)을 직접 돌려본다.

실행:
    python 00_setup/a2_tensors.py
"""
import torch

# 1) 차수(rank)별 텐서 만들기 --------------------------------------------
t0 = torch.tensor(1)                       # 0D 스칼라
t1 = torch.tensor([1, 2, 3])               # 1D 벡터
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2D 행렬
t3 = torch.tensor([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])      # 3D 텐서
print("rank:", t0.dim(), t1.dim(), t2.dim(), t3.dim())   # 0 1 2 3

# 2) 데이터 타입(dtype) ---------------------------------------------------
print("int  dtype:", torch.tensor([1, 2, 3]).dtype)       # torch.int64
print("float dtype:", torch.tensor([1.0, 2.0]).dtype)     # torch.float32
print("to(float32):", t1.to(torch.float32).dtype)         # torch.float32

# 3) 모양(shape) 다루기 ---------------------------------------------------
print("\nshape   :", tuple(t2.shape))          # (2, 3)
print("reshape :", tuple(t2.reshape(3, 2).shape))  # (3, 2)
print("view    :", tuple(t2.view(3, 2).shape))     # (3, 2)
print("T(전치)  :", tuple(t2.T.shape))         # (3, 2)

# 4) 행렬 곱 (matmul == @) ------------------------------------------------
print("\nmatmul:\n", t2.matmul(t2.T))   # [2,3]·[3,2] -> [2,2]
print("@ (동일):\n", t2 @ t2.T)
assert torch.equal(t2.matmul(t2.T), t2 @ t2.T)
print("\nOK: matmul 과 @ 결과 동일.")
