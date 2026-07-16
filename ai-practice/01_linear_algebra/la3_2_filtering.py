"""
[실전선대] §3.2 시계열 필터링과 특징 탐지 — 커널 슬라이딩 내적(합성곱/상관관계) 재현.
목표: 커널과 신호 조각 사이의 내적을 한 칸씩 밀며 반복 계산 → 필터링된 신호 생성.
검증: 출력 길이 = n - klen + 1, 커널 합 = 1(평활화 커널이라 신호 평균 보존).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── 색코딩(house convention): 원본=파랑 · 커널=주황 · 필터링 결과=초록 ──
C_SIGNAL = "#1c7ed6"
C_KERNEL = "#e8590c"
C_RESULT = "#2f9e44"
C_NEG = "#d6336c"

rng = np.random.default_rng(3)
n = 40
t = np.arange(n)
clean = np.sin(t / 5) + 0.4 * np.sin(t / 2.3)
signal = clean + rng.normal(0, 0.35, n)

# 대칭 평활화 커널(삼각형, 합=1) — 대칭이라 "합성곱"과 "상관관계"가 값이 같아지는 특수 케이스.
kernel = np.array([1, 2, 3, 2, 1], dtype=float)
kernel = kernel / kernel.sum()
klen = len(kernel)

# 슬라이딩 내적(= 이 책이 "합성곱"이라 부르는 절차; 커널을 뒤집지 않았으므로 엄밀히는 상관관계 — 노트 ⚠️정확성 참조)
filtered = np.array([np.dot(signal[i:i + klen], kernel) for i in range(n - klen + 1)])
assert len(filtered) == n - klen + 1
assert abs(kernel.sum() - 1.0) < 1e-9

i0 = 14  # 강조할 윈도우 시작 위치

fig, axes = plt.subplots(
    3, 1, figsize=(9, 8),
    gridspec_kw={"height_ratios": [2.4, 1.3, 2.0], "hspace": 0.55},
)

# ── 상단: 원본 신호 + 강조 윈도우 ──
ax0 = axes[0]
ax0.plot(t, signal, color=C_SIGNAL, lw=1.6, marker="o", ms=3)
ax0.add_patch(Rectangle((i0 - 0.5, min(signal) - 0.3), klen, max(signal) - min(signal) + 0.6,
                         facecolor=C_KERNEL, alpha=0.15, edgecolor=C_KERNEL, lw=1.5, ls="--"))
ax0.set_title("original signal", loc="left", fontsize=11, color=C_SIGNAL, fontweight="bold")
ax0.set_xlim(-1, n)
ax0.set_ylabel("value")

# ── 중단: 커널(강조 윈도우와 x축 정렬) + Σ ──
ax1 = axes[1]
ax1.bar(np.arange(i0, i0 + klen), kernel, width=0.6, color=C_KERNEL, alpha=0.85, zorder=2)
ax1.set_xlim(-1, n)
ax1.set_ylim(0, kernel.max() * 2.3)
ax1.set_title("kernel (aligned with highlighted window)", loc="left", fontsize=11, color=C_KERNEL, fontweight="bold")
ax1.set_ylabel("weight")
sigma_y = kernel.max() * 1.9
for k in range(klen):
    ax1.plot([i0 + k, i0 + (klen - 1) / 2], [kernel[k] + 0.01, sigma_y - 0.03],
              ls="dashed", color="gray", lw=0.8, zorder=1)
ax1.scatter([i0 + (klen - 1) / 2], [sigma_y], s=380, color="white", edgecolor="black", zorder=3)
ax1.text(i0 + (klen - 1) / 2, sigma_y, "Σ", fontsize=15, ha="center", va="center",
          color="black", fontweight="bold", zorder=4)

# ── 하단: 필터링된 신호 + 이번 윈도우가 만든 점 강조 ──
ax2 = axes[2]
t_out = np.arange(len(filtered)) + (klen - 1) / 2  # 윈도우 중심에 정렬
ax2.plot(t_out, filtered, color=C_RESULT, lw=1.8, marker="o", ms=3)
out_idx = i0  # filtered[i0]는 win=[i0,i0+klen)의 내적
ax2.scatter([t_out[out_idx]], [filtered[out_idx]], color=C_NEG, zorder=5, s=60)
ax2.annotate("this window's dot product\n(one output point)", xy=(t_out[out_idx], filtered[out_idx]),
             xytext=(t_out[out_idx] - 15, filtered[out_idx] - 1.15),
             arrowprops=dict(arrowstyle="->", color=C_NEG), fontsize=9, color=C_NEG)
ax2.annotate("", xy=(t_out[out_idx] + 18, 1.35), xytext=(t_out[out_idx] + 2, 1.35),
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.4))
ax2.text(t_out[out_idx] + 10, 1.42, "kernel keeps sliding right", ha="center", fontsize=9, color="gray")
ax2.set_xlim(-1, n)
ax2.set_ylim(-1.6, 1.6)
ax2.set_title("filtered signal = sliding dot product output", loc="left",
              fontsize=11, color=C_RESULT, fontweight="bold")
ax2.set_xlabel("t")
ax2.set_ylabel("value")

fig.suptitle("[LA] Fig 3-2 reconstruction - kernel sliding dot product -> filtered signal", fontsize=12, y=0.98)

out_path = "la3_2_filtering.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
print("saved:", out_path)
print("filtered length:", len(filtered), "| expected:", n - klen + 1)
print("filtered[0:5] =", np.round(filtered[:5], 3))
