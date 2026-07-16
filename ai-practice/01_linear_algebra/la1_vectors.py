"""
[실전선대] 1장 — 벡터 파트1: 벡터와 기본 연산 (§1.1 NumPy 벡터 생성·시각화)

핵심: 벡터 = 순서대로 나열된 수. 두 특징 = 차원(원소 수)·방향(열/행).
파이썬에선 NumPy 배열로 만들고, '방향 없는 1차원 배열' vs '방향 있는 2차원 배열(행/열)'을 구분한다.
실행: python la1_vectors.py
"""
import numpy as np
import matplotlib.pyplot as plt


# ── §1.1 벡터 생성 4가지 방법 (책 코드) ───────────────────────
asList = [1, 2, 3]                    # 파이썬 리스트(선대 연산엔 부적합)
asArray = np.array([1, 2, 3])         # 1차원 배열 — '방향 없음'
rowVec = np.array([[1, 2, 3]])        # 행벡터(2차원, 1×3)
colVec = np.array([[1], [2], [3]])    # 열벡터(2차원, 3×1)

# 방향은 'shape(차수=행,열)'로 드러난다
print("asList :", np.shape(asList))   # (3,)
print("asArray:", asArray.shape)      # (3,)   ← 방향 없는 1차원
print("rowVec :", rowVec.shape)       # (1, 3) ← 행 방향
print("colVec :", colVec.shape)       # (3, 1) ← 열 방향
# 차원(수학적, 원소 수) = 3 = len, shape의 곱
print("수학적 차원(원소 수):", asArray.size, "| 파이썬 ndim:", asArray.ndim)


# ── §1.1.1 기하학적 해석: 같은 벡터, 여러 위치 (그림 1-1 개념 재현) ──
def plot_vector_geometry(save_path="la1_vector_geometry.png"):
    v = np.array([-1, 2])             # 벡터 v = 크기+방향 (꼬리→머리)
    # 같은 벡터를 서로 다른 '꼬리 위치'에 그려도 모두 동일한 벡터
    tails = [(0, 0), (-4, -1), (3, -2)]   # (0,0)=기준 위치(standard position)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (tx, ty) in enumerate(tails):
        ax.annotate("", xy=(tx + v[0], ty + v[1]), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4"))
    ax.annotate("standard position", xy=(v[0], v[1]), xytext=(0.3, 2.4))
    ax.axhline(0, ls="--", c="gray", lw=0.8); ax.axvline(0, ls="--", c="gray", lw=0.8)
    ax.set_xlim(-6, 6); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_xlabel("v0"); ax.set_ylabel("v1")
    ax.set_title("Same vector v=[-1,2] at different positions")
    fig.tight_layout(); fig.savefig(save_path, dpi=130)
    print(f"그림 저장: {save_path}")


if __name__ == "__main__":
    plot_vector_geometry()
