"""
부록 A.7 — 일반적인 훈련 루프 (Training Loop)
[밑바닥LLM] 코드 A-9 ~ A-10.

05 신경망(NeuralNetwork) + 06 가중치 + 07 데이터 로더(ToyDataset/DataLoader)를
한데 합쳐 실제로 학습시키고, 손실 수렴과 정확도를 확인한다.
(책 본문은 주피터 셀처럼 위→아래 누적 실행 전제라, 여기선 한 파일로 합쳐 둠.)

실행: python a7_training_loop.py
"""
import torch                              # 파이토치 본체(텐서·autograd·optim)
import torch.nn as nn                     # 신경망 레이어(Linear·ReLU)·Module 모음
import torch.nn.functional as F           # 손실·활성화 등 함수형 연산(cross_entropy 등)
from torch.utils.data import Dataset, DataLoader  # 데이터셋 규약·배치 공급기


# ── 05: 신경망 ────────────────────────────────────────────────
class NeuralNetwork(nn.Module):           # nn.Module 상속 → 파라미터 자동 추적
    def __init__(self, num_inputs, num_outputs):  # 입력 특징 수·출력 클래스 수를 변수로 받아 재사용
        super().__init__()                # 부모(nn.Module) 초기화: 파라미터 추적 장치 켜기(필수)
        self.layers = nn.Sequential(      # 층을 순서대로 실행하는 묶음
            nn.Linear(num_inputs, 30), nn.ReLU(),  # 1번째 은닉층(완전연결) + 비선형
            nn.Linear(30, 20), nn.ReLU(),          # 2번째 은닉층 + 비선형
            nn.Linear(20, num_outputs),            # 출력층 → 마지막 출력 = 로짓(logits)
        )

    def forward(self, x):                 # 정방향 전파: 입력을 층에 통과시켜 로짓 반환
        return self.layers(x)             # (활성화 함수 전 원시 점수)


# ── 07: 데이터셋 / 로더 ───────────────────────────────────────
class ToyDataset(Dataset):                # Dataset 상속 → "index로 한 건 꺼내는" 규약 구현
    def __init__(self, X, y):             # 속성 준비: 특성 X·레이블 y 보관
        self.features = X
        self.labels = y

    def __getitem__(self, i):             # DataLoader가 넘긴 index로 (특성 1건, 레이블 1건) 반환
        return self.features[i], self.labels[i]

    def __len__(self):                    # 데이터셋 길이(샘플 수) 반환
        return self.labels.shape[0]


# 코드 A-5: 예시 데이터(훈련 5건 / 테스트 2건, 특성 2개). 레이블은 0부터 시작
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])   # 앞 3개=클래스0, 뒤 2개=클래스1
X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)   # 원시 텐서를 Dataset 규격 상자에 포장
test_ds = ToyDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)   # 훈련: 배치2·매 에포크 셔플
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)    # 테스트: 셔플 안 함


# ── 코드 A-10: 정확도 함수 ────────────────────────────────────
def compute_accuracy(model, dataloader):  # 데이터로더를 돌며 정확도(맞은 비율) 계산
    model.eval()                          # 평가 모드(드롭아웃 등 정지, 결정론적 예측)
    correct, total = 0.0, 0               # 맞은 개수·전체 개수 누적용
    for features, labels in dataloader:   # 배치 단위 순회
        with torch.no_grad():             # 추론이므로 그래프 미생성(메모리·속도 절약)
            logits = model(features)      # 정방향 전파 → 로짓
        predictions = torch.argmax(logits, dim=1)  # 행마다 최댓값 인덱스 = 예측 클래스
        compare = labels == predictions          # 정답과 비교 → True/False 텐서
        correct += torch.sum(compare)            # True(맞은 것) 개수 합산
        total += len(compare)                    # 전체 개수 합산
    return (correct / total).item()       # 0~1 비율을 파이썬 실수로 반환


def main():
    # ── 코드 A-9: 훈련 루프 ──────────────────────────────────
    torch.manual_seed(123)                # 난수 시드 고정 → 가중치 초기화 재현성 보장
    model = NeuralNetwork(num_inputs=2, num_outputs=2)  # 신경망 인스턴스화(이 모델 파라미터 = 752개)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 학습 가능 파라미터를 SGD에 전달, 학습률 0.5

    num_epochs = 3                        # 전체 데이터를 몇 바퀴 돌지(에포크 수)
    for epoch in range(num_epochs):       # 에포크 반복(전체 데이터 1회 순회 = 1에포크)
        model.train()                     # 훈련 모드(드롭아웃·배치정규화가 훈련 동작)
        for batch_idx, (features, labels) in enumerate(train_loader):  # 미니배치 순회(+배치 번호)
            logits = model(features)                 # ① 정방향 전파: 예측 로짓 계산
            loss = F.cross_entropy(logits, labels)   # ② 손실 계산: 로짓↔정답 교차 엔트로피(softmax 내장)
            optimizer.zero_grad()                    # ③ 그레이디언트 초기화(누적 방지)
            loss.backward()                          # ④ 역전파: 연쇄법칙으로 손실 그레이디언트 계산
            optimizer.step()                         # ⑤ 가중치 갱신: W ← W − lr·grad
            print(f"에포크 {epoch+1:03d}/{num_epochs:03d}"   # 진행 상황 로깅
                  f" | 배치 {batch_idx:03d}/{len(train_loader):03d}"
                  f" | 훈련 손실 {loss:.2f}")          # 손실이 줄며 수렴하는지 확인
        model.eval()                      # 에포크 종료 후 평가 모드로 전환

    # ── 예측 & 확률 ─────────────────────────────────────────
    torch.set_printoptions(sci_mode=False)  # 지수표기 끄기(읽기 쉬운 출력)
    model.eval()                          # 평가 모드
    with torch.no_grad():                 # 추론: 그래프 미생성
        outputs = model(X_train)          # 훈련 데이터에 대한 로짓
    probas = torch.softmax(outputs, dim=1)      # 로짓 → 클래스 소속 확률(합=1)
    predictions = torch.argmax(probas, dim=1)   # 최대 확률 클래스 = 예측 레이블
    print("\n예측:", predictions.tolist(), "| 정답:", y_train.tolist())  # 예측 vs 정답 비교 출력

    # ── 정확도 ─────────────────────────────────────────────
    print("train 정확도:", compute_accuracy(model, train_loader))  # 훈련셋 정확도(작은 데이터라 1.0)
    print("test  정확도:", compute_accuracy(model, test_loader))   # 테스트셋 정확도


if __name__ == "__main__":               # 스크립트로 직접 실행할 때만 main() 호출
    main()
