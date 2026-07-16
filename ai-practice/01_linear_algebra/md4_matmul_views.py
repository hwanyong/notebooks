import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import numpy as np

# palette
RED="#d6336c"; BLUE="#1c7ed6"; GREY="#adb5bd"; FILLR="#ffe3ec"; FILLB="#d0ebff"; DARK="#343a40"
plt.rcParams.update({"font.size":11})

fig, (axL, axR) = plt.subplots(1,2, figsize=(13.5,5.6))

# ---------- helper to draw a grid ----------
def grid(ax, x0, y0, nrow, ncol, cw=0.62, ch=0.62, row_fill=None, col_fill=None, cell_fill=None, label=None):
    # returns center coords dict
    centers={}
    for r in range(nrow):
        for c in range(ncol):
            x=x0+c*cw; y=y0-r*ch
            fc="white"
            if row_fill and r in row_fill: fc=row_fill[r]
            if col_fill and c in col_fill: fc=col_fill[c]
            if cell_fill and (r,c) in cell_fill: fc=cell_fill[(r,c)]
            ax.add_patch(Rectangle((x,y),cw,ch,fill=True,facecolor=fc,edgecolor=GREY,lw=1.2))
            centers[(r,c)]=(x+cw/2,y+ch/2)
    # brackets
    ax.plot([x0-0.05,x0-0.05],[y0+ch,y0-(nrow-1)*ch],color=DARK,lw=1.6)
    ax.plot([x0-0.05,x0+0.04],[y0+ch,y0+ch],color=DARK,lw=1.6)
    ax.plot([x0-0.05,x0+0.04],[y0-(nrow-1)*ch,y0-(nrow-1)*ch],color=DARK,lw=1.6)
    xr=x0+ncol*cw
    ax.plot([xr+0.05,xr+0.05],[y0+ch,y0-(nrow-1)*ch],color=DARK,lw=1.6)
    ax.plot([xr+0.05,xr-0.04],[y0+ch,y0+ch],color=DARK,lw=1.6)
    ax.plot([xr+0.05,xr-0.04],[y0-(nrow-1)*ch,y0-(nrow-1)*ch],color=DARK,lw=1.6)
    if label: ax.text(x0-0.45, y0-((nrow-1)*ch)/2+ch/2, label, fontsize=18, fontweight="bold", va="center", ha="center", color=DARK)
    return centers

# ================= LEFT: view ① cell = dot product =================
axL.set_title("① cell view :  C[i,j] = (row i of A) · (col j of B)", fontsize=12.5, fontweight="bold", pad=12)
# A 3x3 bottom-left, rows 0(red) & 2(blue) highlighted
A = grid(axL, 0.4, 3.0, 3,3, row_fill={0:FILLR,2:FILLB}, label="A")
# B 3x3 top, cols 0(red) & 1(blue) highlighted
B = grid(axL, 4.6, 5.4, 3,3, col_fill={0:FILLR,1:FILLB}, label="B")
# C 3x3 result, cells (0,0)=red (2,1)=blue
C = grid(axL, 4.6, 3.0, 3,3, cell_fill={(0,0):RED,(2,1):BLUE}, label="C")
# arrows: A row0 -> C(0,0)
axL.annotate("", xy=(C[(0,0)][0]-0.05,C[(0,0)][1]), xytext=(A[(0,2)][0]+0.4,A[(0,2)][1]),
             arrowprops=dict(arrowstyle="-|>",color=RED,lw=2.4))
# B col0 -> C(0,0)
axL.annotate("", xy=(C[(0,0)][0],C[(0,0)][1]+0.05), xytext=(B[(2,0)][0],B[(2,0)][1]-0.4),
             arrowprops=dict(arrowstyle="-|>",color=RED,lw=2.4))
# A row2 -> C(2,1)
axL.annotate("", xy=(C[(2,1)][0]-0.05,C[(2,1)][1]), xytext=(A[(2,2)][0]+0.4,A[(2,2)][1]),
             arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=2.4))
# B col1 -> C(2,1)
axL.annotate("", xy=(C[(2,1)][0],C[(2,1)][1]+0.05), xytext=(B[(2,1)][0],B[(2,1)][1]-0.4),
             arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=2.4))
axL.text(4.6+1.5*0.62, 1.55, "each cell = one dot product (weighted sum  Σ aₖbₖ)",
         ha="center", fontsize=10.5, color=DARK)
axL.set_xlim(-0.4,7.2); axL.set_ylim(1.2,6.6); axL.axis("off")

# ================= RIGHT: view ② column = linear combination =================
axR.set_title("② column view :  a column of C = linear combination of A's columns", fontsize=12.5, fontweight="bold", pad=12)
# show: C_col1 = 3*colKor + 3*colMath + 4*colEng  using the grade example
cols = {
 "k":(np.array([80,75,90,99]), "#e8590c"),  # weights drawn as scalars
}
# We'll render an equation of column vectors
def colvec(ax, x, y, vals, color, scale=0.34, w=0.62):
    n=len(vals)
    for r,val in enumerate(vals):
        yy=y-r*scale
        ax.add_patch(Rectangle((x,yy),w,scale,fill=True,facecolor="white",edgecolor=color,lw=1.6))
        ax.text(x+w/2, yy+scale/2, str(val), ha="center", va="center", fontsize=9.5, color=DARK)
    # brackets
    ax.plot([x-0.04,x-0.04],[y+scale, y-(n-1)*scale],color=color,lw=1.5)
    ax.plot([x+w+0.04,x+w+0.04],[y+scale, y-(n-1)*scale],color=color,lw=1.5)
    return x

y0=4.2
axR.text(0.05, y0+0.55, "우수반 column of C", fontsize=10.5, color=DARK)  # will be tofu? korean -> avoid
# replace korean with english
axR.texts[-1].set_text("'class-A' column of C")
colvec(axR, 0.2, y0, [750,825,815,787], "#2f9e44")
axR.text(1.15, y0-0.55, "=", fontsize=16, ha="center")
axR.text(1.5, y0-0.2, "3", fontsize=15, color=RED, ha="center", fontweight="bold")
colvec(axR, 1.75, y0, [80,75,90,99], RED)
axR.text(2.7, y0-0.55, "+", fontsize=16, ha="center")
axR.text(3.05, y0-0.2, "3", fontsize=15, color=BLUE, ha="center", fontweight="bold")
colvec(axR, 3.3, y0, [90,80,95,70], BLUE)
axR.text(4.25, y0-0.55, "+", fontsize=16, ha="center")
axR.text(4.6, y0-0.2, "4", fontsize=15, color="#e8590c", ha="center", fontweight="bold")
colvec(axR, 4.85, y0, [60,90,65,70], "#e8590c")
axR.text(3.0, 1.75, "weights (3,3,4) = B's column   →   scale A's score-columns and add",
         ha="center", fontsize=10, color=DARK)
axR.text(3.0, 1.35, "= linear combination  (note 06,  w = λ₁v₁+λ₂v₂+λ₃v₃)",
         ha="center", fontsize=10, color="#862e9c", fontweight="bold")
axR.set_xlim(-0.3,6.0); axR.set_ylim(1.0,5.2); axR.axis("off")

fig.suptitle("Matrix product AB — two views (same operation, different zoom)", fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0,0,1,0.96])
out="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments/MD4-matmul 두 시점 칸내적·열선형결합 (재구성).png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
