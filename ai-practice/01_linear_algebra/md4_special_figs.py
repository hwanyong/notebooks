import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
DARK="#343a40"; NZ="#1c7ed6"; Z="#f1f3f5"; DI="#2f9e44"; GREEN="#2f9e44"
plt.rcParams.update({"font.size":11})
def matgrid(ax,vals,title,note,fill):
    n=len(vals)
    for r in range(n):
        for c in range(n):
            v=vals[r][c]; fc=fill(r,c,v)
            ax.add_patch(Rectangle((c,n-1-r),1,1,facecolor=fc,edgecolor="white",lw=2))
            ax.text(c+0.5,n-1-r+0.5,str(v),ha="center",va="center",
                    fontsize=11,color="white" if fc!=Z else "#adb5bd",fontweight="bold")
    ax.set_xlim(-0.1,n+0.1); ax.set_ylim(-1.0,n+0.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title,fontsize=11.5,fontweight="bold",pad=4)
    ax.text(n/2,-0.6,note,ha="center",fontsize=8.6,color=DARK)
figB,axs=plt.subplots(2,3,figsize=(13,8))
matgrid(axs[0,0],[[4,2,7],[5,1,9],[4,0,1]],"Square",
        "rows = cols; basis of linear transforms", lambda r,c,v:NZ)
matgrid(axs[0,1],[[1,0,0],[0,1,0],[0,0,1]],"Identity  I",
        "diag=1, else 0; '×1' = undo transform", lambda r,c,v:(DI if r==c else Z))
matgrid(axs[0,2],[[4,0,0],[0,2,0],[0,0,5]],"Diagonal",
        "diag only; per-axis scalar stretch", lambda r,c,v:(DI if r==c else Z))
matgrid(axs[1,0],[[4,2,9],[0,1,6],[0,0,5]],"Triangular",
        "diag + one side; easy to solve (LU)", lambda r,c,v:(NZ if c>=r else Z))
matgrid(axs[1,1],[[0,0,0],[0,0,2],[0,0,0]],"Sparse",
        "mostly 0; store nonzeros only (memory)", lambda r,c,v:(NZ if v!=0 else Z))
ax=axs[1,2]; ax.axis("off")
ax.text(0.5,0.80,"Inverse  A inv",ha="center",fontsize=11.5,fontweight="bold",transform=ax.transAxes)
ax.text(0.5,0.54,r"$A^{-1}A = I$",ha="center",fontsize=20,transform=ax.transAxes,color=GREEN)
ax.text(0.5,0.32,"undoes A's transform",ha="center",fontsize=9.5,transform=ax.transAxes,color=DARK)
ax.text(0.5,0.15,"exists iff  det(A) ≠ 0  (square, invertible)",ha="center",fontsize=9,transform=ax.transAxes,color="#d6336c")
figB.suptitle("Special matrices — patterns at a glance",fontsize=13.5,fontweight="bold",y=0.98)
figB.tight_layout(rect=[0,0,1,0.95])
outB="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments/MD4-special 특수행렬 6종 지도 (재구성).png"
figB.savefig(outB,dpi=150,bbox_inches="tight"); print("saved",outB)
