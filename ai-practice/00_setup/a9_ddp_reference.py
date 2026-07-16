"""
부록 A.9.3 — 다중 GPU 분산 훈련 (DistributedDataParallel, DDP) [참고용]
[밑바닥LLM] 코드 A-12 ~ A-13.

⚠️ 실행 환경: NVIDIA GPU 2장 이상 + NCCL 백엔드 전용. **내 맥(Apple Silicon/MPS)에선
   실행되지 않는다**(NCCL은 NVIDIA 전용). 개념·구조 학습용 참고 코드.
   또한 DDP는 주피터 같은 인터랙티브 환경 X → 반드시 .py 스크립트로 실행.
   실제 실행은 책 공식 저장소: https://github.com/rickiepark/llm-from-scratch

DDP = '데이터 병렬'. 모델을 GPU마다 '복제'하고, 데이터를 GPU마다 '다른 미니배치'로 나눠
동시에 forward/backward → 각 GPU의 그레이디언트를 all-reduce로 '평균·동기화' → 모든
복사본이 같은 가중치로 갱신. (cf. '모델 병렬/샤딩'은 모델 자체를 쪼갬 → 다른 축)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 코드 A-13: 분산 환경 셋업
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"   # 메인 노드 주소
    os.environ["MASTER_PORT"] = "12345"       # 사용 가능한 임의의 포트
    init_process_group(
        backend="nccl",                       # NCCL = NVIDIA 집합통신 라이브러리(GPU↔GPU)
        rank=rank,                            # rank = 이 프로세스가 쓸 GPU 인덱스
        world_size=world_size,                # world_size = 총 프로세스(GPU) 수
    )
    torch.cuda.set_device(rank)               # 이 프로세스가 쓸 현재 GPU 지정


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


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0.0, 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        correct += torch.sum(torch.argmax(logits, dim=1) == labels)
        total += len(labels)
    return (correct / total).item()


def prepare_dataset():
    X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
    y_test = torch.tensor([0, 1])
    train_ds, test_ds = ToyDataset(X_train, y_train), ToyDataset(X_test, y_test)
    train_loader = DataLoader(
        dataset=train_ds, batch_size=2,
        shuffle=False,                              # 셔플은 DistributedSampler가 담당
        pin_memory=True, drop_last=True,            # GPU 전송 가속(NVIDIA 분리형이라 유효)
        sampler=DistributedSampler(train_ds),       # 각 프로세스(GPU)에 겹치지 않는 부분집합 분배
    )
    test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False)
    return train_loader, test_loader


# 코드 A-13: DDP 모델 훈련
def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)                                  # rank = GPU ID
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])           # DDP로 감싸 grad 동기화 자동화

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)       # 에포크마다 분산 셔플 갱신
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            loss = F.cross_entropy(model(features), labels)
            optimizer.zero_grad()
            loss.backward()                         # backward 시 all-reduce로 grad 평균·동기화
            optimizer.step()
            print(f"[GPU{rank}] 에포크 {epoch+1:03d}/{num_epochs:03d}"
                  f" | 배치 크기 {labels.shape[0]:03d} | 훈련 손실 {loss:.2f}")

    model.eval()
    # 중복 출력 방지: rank 0 프로세스만 평가 출력
    train_acc = compute_accuracy(model, train_loader, device=rank)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    if rank == 0:                                   # 첫 프로세스만 출력(중복 제거)
        print(f"[GPU{rank}] 훈련 정확도 {train_acc} | 테스트 정확도 {test_acc}")
    destroy_process_group()                         # 분산 자원 해제


if __name__ == "__main__":
    # 단일 GPU 머신과 다중 GPU 머신 모두 동작. (CPU/MPS에선 NCCL 미지원으로 실패)
    print("사용 가능한 GPU 개수:", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    # GPU당 프로세스 1개를 spawn. main의 첫 인자 rank는 spawn이 자동 주입.
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # GPU 선택은 스크립트 수정 없이: CUDA_VISIBLE_DEVICES=0,2 python a9_ddp_reference.py
