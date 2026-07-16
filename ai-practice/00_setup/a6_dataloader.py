"""
부록 A.6 — 효율적인 데이터 로더 (Dataset · DataLoader)
[밑바닥LLM] 코드 A-5 ~ A-8 종합 실습.

목표: Dataset을 직접 구현해 DataLoader로 감싸고,
      배치 분할 · shuffle · drop_last · num_workers 의 효과를 눈으로 확인.

실행: python a6_dataloader.py
"""
import torch
from torch.utils.data import Dataset, DataLoader


# ── 코드 A-5: 예시 데이터셋 (train 5건 / test 2건, 특성 2개) ──────────────
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5],
])
y_train = torch.tensor([0, 0, 0, 1, 1])   # 클래스 레이블은 0부터 시작
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])


# ── 코드 A-6: 사용자 정의 Dataset — 3개 메서드만 구현 ────────────────────
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        # DataLoader가 넘겨준 index로 (특성 1건, 레이블 1건) 반환
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]   # 데이터셋의 총 길이


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)


def main():
    print("train_ds 길이:", len(train_ds))      # 5

    # ── 코드 A-7: DataLoader 초기화 ──────────────────────────────────
    torch.manual_seed(123)
    train_loader = DataLoader(dataset=train_ds, batch_size=2,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_ds, batch_size=2,
                             shuffle=False, num_workers=0)

    print("\n[train_loader] batch_size=2, shuffle=True  → 5 = 2+2+1")
    for idx, (x, y) in enumerate(train_loader):
        print(f"  배치 {idx + 1}:", x.tolist(), y.tolist())

    print("\n[test_loader] shuffle=False (테스트는 섞지 않음)")
    for idx, (x, y) in enumerate(test_loader):
        print(f"  배치 {idx + 1}:", x.tolist(), y.tolist())

    # ── 코드 A-8: 마지막 반쪽 배치 버리기 (drop_last=True) ────────────
    torch.manual_seed(123)
    train_loader_drop = DataLoader(dataset=train_ds, batch_size=2,
                                   shuffle=True, num_workers=0,
                                   drop_last=True)
    print("\n[train_loader] drop_last=True  → 마지막 1샘플 배치 제외 (배치 2개만)")
    for idx, (x, y) in enumerate(train_loader_drop):
        print(f"  배치 {idx + 1}:", x.tolist(), y.tolist())

    # 참고: num_workers
    #  - 작은 데이터셋·주피터 노트북 → 0 (워커 생성 오버헤드가 이득보다 큼)
    #  - 실전 대형 데이터 → 4 등으로 올려 GPU가 다음 배치를 기다리는 병목 해소


if __name__ == "__main__":
    main()
