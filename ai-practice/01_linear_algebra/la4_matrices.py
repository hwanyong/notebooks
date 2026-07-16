"""
[실전선대] 4장 — 행렬 파트1 (§4.1 NumPy 행렬 생성과 시각화 · §4.1.1 인덱싱·슬라이싱)

핵심: 행렬 = 벡터를 한 차원 위로. 크기 = (행, 열). 수학 1-기반 ↔ 파이썬 0-기반.
슬라이싱 start:stop:step 은 stop 미포함(half-open) → 길이 = stop - start.
실행: python la4_matrices.py
노트: 수학 원자 [2.0 행렬 — 정의·크기·인덱싱·슬라이싱] / 렌즈 [09]
"""
import numpy as np
import matplotlib.pyplot as plt


# ── §4.1.1 크기 규칙 (행, 열) ──────────────────────────────
A = np.arange(60).reshape(6, 10)      # 6행 10열
print("shape:", A.shape)              # (6, 10) — 행 먼저, 열 나중
assert A.shape == (6, 10)

# ── 인덱싱: 수학 a_{3,4} = 파이썬 A[2,3] (각 인덱스 -1) ─────
print("a_{3,4} = A[2,3] =", A[2, 3])  # 23
assert A[2, 3] == 23

# 책 3×5 예제 행렬로 a_{3,4}=8 검증
M = np.array([[1, 3, 5, 7, 9],
              [0, 2, 4, 6, 8],
              [1, 4, 7, 8, 9]])
assert M.shape == (3, 5)              # 행 3, 열 5 → 3×5
assert M[2, 3] == 8                   # 수학 a_{3,4} = 8 (책 예제)

# ── §4.1.1 슬라이싱 (책 코드): 행 2~4 × 열 1~5 ──────────────
sub = A[1:4:1, 0:5:1]                 # stop 미포함 → 행 {1,2,3}, 열 {0..4}
print("원본 행렬:\n", A)
print("부분 행렬:\n", sub)            # [[10..14],[20..24],[30..34]]
assert sub.shape == (3, 5)            # 길이 = stop - start = 3, 5
assert sub[0, 0] == 10 and sub[-1, -1] == 34

# half-open 이득: 겹침·누락 없이 이어진다
assert np.array_equal(np.vstack([A[:3], A[3:]]), A)

# ── 열/행 추출 시 2D 유지 (규칙: 열벡터는 명시적 (N,1)) ──────
col_1d = A[:, 1]                      # (6,)  ← 방향 잃은 1D
col_2d = A[:, [1]]                    # (6,1) ← 열벡터 유지
row_2d = A[[1], :]                    # (1,10) ← 행벡터 유지
print("A[:,1]:", col_1d.shape, "| A[:,[1]]:", col_2d.shape, "| A[[1],:]:", row_2d.shape)
assert col_1d.shape == (6,) and col_2d.shape == (6, 1) and row_2d.shape == (1, 10)


# ── §4.1.2 특수 행렬: 범주별 전용 생성 함수 ─────────────────
R = np.random.randn(4, 6)             # 난수(가우스): Mrows=4(shape 0), Ncols=6(shape 1)
assert R.shape == (4, 6)              # 비정방: 열>행 → '넓다(wide)' (행>열이면 '높다(tall)')

D = np.diag([2, 5, 1, 2])             # 벡터 입력 → 대각 '행렬' 생성
d = np.diag(D)                        # 행렬 입력 → 대각 '벡터' 추출 (이중 동작, 대각화 아님!)
assert D.shape == (4, 4) and np.array_equal(d, [2, 5, 1, 2])

U, L = np.triu(R), np.tril(R)         # 상/하 삼각 추출
assert np.allclose(np.tril(U, -1), 0) # 상삼각: 대각 아래 전부 0
assert np.allclose(np.triu(L, 1), 0)  # 하삼각: 대각 위 전부 0

I5 = np.eye(5)                        # 단위(=항등) 행렬 I₅ — 대각 1 정방 대각
v = np.arange(5.0).reshape(-1, 1)     # (5,1) 열벡터
assert np.array_equal(I5 @ v, v)      # I 곱해도 동일 (숫자 1과 동등)

Z = np.zeros((3, 4))                  # 영 행렬 (굵은 0 — 영벡터와 같은 기호)
assert not Z.any()


# ── §4.2.1 행렬 덧셈/뺄셈: 대응 원소끼리, 같은 크기만 ────────
M1 = np.array([[2, 3, 4], [1, 2, 4]])
M2 = np.array([[0, 3, 1], [-1, -4, 2]])
assert np.array_equal(M1 + M2, [[2, 6, 5], [0, -2, 6]])   # 책 예제
try:
    M1 + np.zeros((3, 3)); assert False
except ValueError:
    pass                                  # 크기 다르면 불가 ✓

# ── §4.2.2 행렬 '이동': A + λI (A+s는 이동이 아님!) ─────────
B = np.array([[4, 5, 1], [0, 1, 11], [4, 9, 7]])
s = 6
shifted = B + s * np.eye(len(B))          # 이동: 대각만 +6
assert np.array_equal(shifted, [[10, 5, 1], [0, 7, 11], [4, 9, 13]])
broadcast = B + s                         # 브로드캐스팅: 전 원소 +6 (선대 연산 아님)
assert broadcast[0, 1] == 11 and shifted[0, 1] == 5       # 비대각 차이로 구분
off_diag_unchanged = shifted - np.diag(np.diag(shifted))
assert np.array_equal(off_diag_unchanged, B - np.diag(np.diag(B)))  # 비대각 보존

# ── §4.2.3 스칼라곱·아다마르곱 ⭐ * vs @ ──────────────────
G = np.array([[2, 3], [4, 5]])
assert np.array_equal(3 * G, [[6, 9], [12, 15]])          # 스칼라-행렬 곱(원소별)
H = np.array([[10, 20], [30, 40]])
assert np.array_equal(G * H, np.multiply(G, H))            # * == multiply == 아다마르
assert np.array_equal(G * H, [[20, 60], [120, 200]])       # 원소별 곱
assert not np.array_equal(G * H, G @ H)                    # ⭐ 아다마르 ≠ 행렬곱!
P = np.random.randn(3, 4)
try:
    P @ P; assert False
except ValueError:
    pass                                   # (3,4)@(3,4) 무효 — 내부 차원 불일치
assert (P * P).shape == (3, 4)             # ⊙는 같은 크기면 항상 OK

# ── §4.3 표준 행렬곱: 유효성(내부/외부)·식 4-1·행렬-벡터 곱 ──
assert (np.random.randn(3, 4) @ np.random.randn(4, 2)).shape == (3, 2)  # (M,N)(N,K)→(M,K)
E = np.array([[2, 3], [4, 5]])             # 식 4-1: 각 칸 = 행·열 내적
F = np.array([[1, 10], [100, 1000]])
assert np.array_equal(E @ F, [[2*1+3*100, 2*10+3*1000], [4*1+5*100, 4*10+5*1000]])
# 행렬-벡터 곱: Av(열벡터 뒤)만 유효, 결과 방향 = 곱한 벡터 방향
Mv_mat = np.array([[2, 3], [2, 1]])
col = np.array([[1], [1.5]])
assert (Mv_mat @ col).shape == (2, 1)      # 열벡터 → 열벡터
row = np.array([[1, 1.5]])
assert (row @ Mv_mat).shape == (1, 2)      # 행벡터 앞곱 → 행벡터
# 고유벡터 맛보기(그림 4-5): v=[1.5,1]ᵀ는 M의 고유벡터(고윳값 4)
v_eig = np.array([[1.5], [1]])
assert np.allclose(Mv_mat @ v_eig, 4 * v_eig)              # Mv = 4v (방향 유지)

# ── §4.4~4.5 전치 · LIVE EVIL ──────────────────────────────
T = np.array([[3, 0, 4], [9, 8, 3]])
assert np.array_equal(T.T, [[3, 9], [0, 8], [4, 3]])       # 식 4-2 예제
assert np.array_equal(T.T.T, T)                            # Cᵀᵀ = C
v1d = np.array([3, 4, 5])
assert v1d.T.shape == (3,)                                 # 각주 3: 1D는 .T 먹통 (T04)
# 내적/외적 표기 = 유효성 규칙의 필연 (§4.4.1)
a = np.array([[1], [2], [3]]); b = np.array([[4], [5], [6]])
assert (a.T @ b).shape == (1, 1)                           # aᵀb: (1,3)(3,1)→(1,1) 스칼라
assert (a @ b.T).shape == (3, 3)                           # abᵀ: (3,1)(1,3)→(3,3) 외적
# LIVE EVIL: (AB)ᵀ = BᵀAᵀ (AᵀBᵀ는 무효)
Lm = np.random.randn(2, 3); Vm = np.random.randn(3, 5)
assert np.allclose((Lm @ Vm).T, Vm.T @ Lm.T)               # 순서 반전 ✓
try:
    Lm.T @ Vm.T; assert False
except ValueError:
    pass                                                   # (3,2)(5,3) 내부 불일치

# ── §4.6 대칭 행렬: Aᵀ=A · AᵀA/AAᵀ 곱셈 기법 ──────────────
S = np.array([[1, 7], [7, 2]])
assert np.array_equal(S.T, S)                              # 대칭 정의: Aᵀ = A
R46 = np.random.randn(4, 6)                                # 비정방·비대칭이라도
AtA, AAt = R46.T @ R46, R46 @ R46.T
assert AtA.shape == (6, 6) and AAt.shape == (4, 4)         # 둘 다 정방 (크기는 다름!)
assert np.allclose(AtA.T, AtA) and np.allclose(AAt.T, AAt) # 둘 다 대칭 (LIVE EVIL 증명)
assert AtA.shape != AAt.shape                              # 📌 AᵀA ≠ AAᵀ


# ── §4.1.1 큰 행렬은 이미지로 시각화 (그림 4-1 재현) ─────────
def plot_matrices_as_images(save_path="la4_matrices_as_images.png"):
    rng = np.random.default_rng(4)
    blocks = rng.random((4, 6))                       # 무작위 블록
    noise = rng.standard_normal((100, 100))           # 가우스 노이즈
    n = 12
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    grad = -np.abs(I - J).astype(float)               # 대각 그레이디언트

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for ax, (mat, cmap, title) in zip(axes, [
        (blocks, "Blues", "Random blocks (4x6)"),
        (noise, "gray", "Random noise (100x100)"),
        (grad, "Blues_r", "Diagonal gradient (12x12)"),
    ]):
        ax.imshow(mat, cmap=cmap, interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print("saved:", save_path)


if __name__ == "__main__":
    plot_matrices_as_images()
    print("✅ §4.1~4.5 전 검증 통과 — 인덱싱·슬라이싱 · 특수 행렬 · 덧셈·이동 · ⭐아다마르(*)≠행렬곱(@) · 유효성 · 고유벡터 Mv=4v · 전치 Cᵀᵀ=C · LIVE EVIL (AB)ᵀ=BᵀAᵀ")
