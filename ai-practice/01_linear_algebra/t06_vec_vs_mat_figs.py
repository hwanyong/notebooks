"""T06 figures: (A) vector=state vs matrix=operator (geometric),
                (B) arithmetic intensity: bandwidth-bound vs compute-bound.
English labels only (avoid Hangul tofu in matplotlib). Palette:
  vector=blue #1c7ed6, operator/transformed=orange #e8590c,
  basis=green #2f9e44, accent=pink #d6336c.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BLUE, ORANGE, GREEN, PINK = "#1c7ed6", "#e8590c", "#2f9e44", "#d6336c"
GRID = "#ced4da"
OUT = "/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/탐구/attachments"

def arrow(ax, p0, p1, color, lw=2.4, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, zorder=z))

# ---------- Figure A: state vs operator ----------
figA, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.4))

# left: vector = state on a FIXED grid
for ax in (axL, axR):
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
    ax.set_aspect("equal"); ax.set_xticks(range(-1, 6)); ax.set_yticks(range(-1, 6))
    ax.axhline(0, color="#868e96", lw=1.0, zorder=1)
    ax.axvline(0, color="#868e96", lw=1.0, zorder=1)

# static grid (left)
for k in range(-1, 6):
    axL.plot([-1, 5], [k, k], color=GRID, lw=0.8, zorder=0)
    axL.plot([k, k], [-1, 5], color=GRID, lw=0.8, zorder=0)
v = np.array([3, 2])
arrow(axL, (0, 0), v, BLUE)
axL.plot(*v, "o", color=BLUE, zorder=6)
axL.text(v[0]+0.1, v[1]+0.15, "v = (3, 2)\na STATE / coordinate", color=BLUE, fontsize=11, fontweight="bold")
# unit basis (faint) on left
arrow(axL, (0, 0), (1, 0), "#adb5bd", lw=1.6, z=3)
arrow(axL, (0, 0), (0, 1), "#adb5bd", lw=1.6, z=3)
axL.set_title("Vector = STATE\n(a fixed point in an unchanged space)", fontsize=12, fontweight="bold")

# right: matrix = operator -> transform the WHOLE grid
A = np.array([[1.0, 1.0],
              [0.3, 1.6]])   # shear + stretch
# transformed grid lines: map each gridline endpoint through A
for k in range(-1, 6):
    # horizontal line y=k  -> points (x,k)
    pts = np.array([[x, k] for x in np.linspace(-1, 5, 30)]).T
    tp = A @ pts
    axR.plot(tp[0], tp[1], color="#ffd8a8", lw=0.9, zorder=0)
    # vertical line x=k
    pts = np.array([[k, y] for y in np.linspace(-1, 5, 30)]).T
    tp = A @ pts
    axR.plot(tp[0], tp[1], color="#ffd8a8", lw=0.9, zorder=0)
# basis images = columns of A
c1 = A @ np.array([1, 0]); c2 = A @ np.array([0, 1])
arrow(axR, (0, 0), c1, GREEN, z=5)
arrow(axR, (0, 0), c2, ORANGE, z=5)
axR.text(c1[0]+0.05, c1[1]-0.35, "col 1 = A·î", color=GREEN, fontsize=10, fontweight="bold")
axR.text(c2[0]+0.05, c2[1]+0.1, "col 2 = A·ĵ", color=ORANGE, fontsize=10, fontweight="bold")
# show v mapped to Av
Av = A @ v
arrow(axR, (0, 0), Av, BLUE, lw=2.0, z=6)
axR.plot(*Av, "o", color=BLUE, zorder=7)
axR.text(Av[0]+0.1, Av[1]+0.1, "A·v", color=BLUE, fontsize=11, fontweight="bold")
axR.set_title("Matrix = OPERATOR\n(it reshapes the whole grid; columns = where basis lands)",
              fontsize=12, fontweight="bold")

figA.suptitle("State vs Operator — a vector is acted upon, a matrix does the acting",
              fontsize=13, fontweight="bold")
figA.tight_layout(rect=[0, 0, 1, 0.96])
figA.savefig(f"{OUT}/T06-1 state-vs-operator.png", dpi=150, bbox_inches="tight")
print("saved A")

# ---------- Figure B: arithmetic intensity ----------
figB, ax = plt.subplots(figsize=(9.2, 5.0))
labels = ["Vector ops\n(dot, axpy — BLAS-1)", "Matrix mul\n(GEMM — BLAS-3)"]
# arithmetic intensity ~ FLOP/byte: vector op O(1); GEMM O(n) grows with size
intensity = [0.25, 8.0]
colors = [BLUE, ORANGE]
bars = ax.bar(labels, intensity, color=colors, width=0.55, zorder=3)
balance = 1.2  # machine balance (illustrative)
ax.axhline(balance, color=PINK, ls="--", lw=2, zorder=4)
ax.text(1.46, balance+0.15, "machine balance\n(FLOP/byte the HW can sustain)",
        color=PINK, fontsize=9, ha="right", va="bottom")
ax.text(-0.42, balance/2, "BELOW → Memory-\nBANDWIDTH bound", color=BLUE, fontsize=10, fontweight="bold", va="center")
ax.text(-0.42, 5.0, "ABOVE → \nCOMPUTE bound", color=ORANGE, fontsize=10, fontweight="bold", va="center")
for b, val, note in zip(bars, intensity, ["~O(1): read data, 1-2 FLOPs, toss",
                                          "~O(n): reuse data in cache, n³ FLOPs"]):
    ax.text(b.get_x()+b.get_width()/2, val+0.2, note, ha="center", fontsize=9)
ax.set_ylabel("Arithmetic Intensity  (FLOP / byte)", fontsize=11)
ax.set_ylim(0, 9.5)
ax.set_title("Why matrices are the GPU's favorite: data reuse\nVector ops starve on bandwidth; GEMM saturates the compute units",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
figB.tight_layout()
figB.savefig(f"{OUT}/T06-2 arithmetic-intensity.png", dpi=150, bbox_inches="tight")
print("saved B")
