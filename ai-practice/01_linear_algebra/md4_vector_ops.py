"""
[필수수학] §4.1 — 벡터 연산: 생성·덧셈(교환법칙)·스케일링(음수 반전)
※ 열벡터를 명시적 2D (N,1)로 표기 (→ 탐구 T04 규칙). 책의 1D는 대조용 병기.
   [수학] v=[3;2] → [책 1D] np.array([3,2]) (2,) → [내 2D] np.array([[3],[2]]) (2,1)
(도판은 md4_figs.py 로 분리.)  실행: python md4_vector_ops.py
"""
import numpy as np

# ── 예제 4-1~4-4: 벡터 생성 (열벡터는 명시적 (N,1)) ───────────
v  = np.array([[3], [2]])            # 2D 열벡터 (2,1)
v3 = np.array([[4], [1], [2]])       # 3차원 열벡터 (3,1)
v5 = np.array([[6], [1], [5], [8], [3]])   # 5차원 열벡터 (5,1)
print("v.shape", v.shape, "| v3.shape", v3.shape, "| v5.shape", v5.shape)

# ── 예제 4-5: 덧셈 (교환법칙) ─────────────────────────────────
w = np.array([[2], [-1]])
print("v+w =", (v + w).ravel(), "| 교환법칙 v+w==w+v:", np.array_equal(v + w, w + v))  # [5 1]

# ── 예제 4-6: 스케일링 (음수=방향 반전, 같은 선상) ────────────
a = np.array([[3], [1]])
print("2a =", (2.0 * a).ravel(), "| 0.5a =", (0.5 * a).ravel(), "| -a =", (-1 * a).ravel())

# ── 스칼라 2 vs 1×1 행렬 [[2]]: 결과 같고 정체 다름 ───────────
res_scalar = a * 2                          # 스칼라 곱(브로드캐스팅)
res_matrix = a @ np.array([[2]])            # 행렬 곱 (2,1)·(1,1)=(2,1)
print("스칼라곱 =", res_scalar.ravel(), "| 행렬곱 =", res_matrix.ravel(),
      "| 동일:", np.array_equal(res_scalar, res_matrix))
print("정체: ndim(2)=", np.ndim(2), "| [[2]].shape=", np.array([[2]]).shape)  # 0 vs (1,1)

# ── 대조: 책 1D shortcut은 전치가 '먹통' (→ T04) ──────────────
v1d = np.array([3, 2])              # (2,) 축 없음
print("\n[책 1D] shape", v1d.shape, "→ .T", v1d.T.shape, "(전치 먹통)")
print("[내 2D] shape", v.shape, "→ .T", v.T.shape, "(→ (1,2) 행벡터로 누움)")
