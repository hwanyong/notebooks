import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
BLUE="#1c7ed6"; ORANGE="#e8590c"; GREEN="#2f9e44"; GREY="#ced4da"; DARK="#343a40"
plt.rcParams.update({"font.size":11})

# ===================== FIGURE A: inverse = undo (A⁻¹A = I) =====================
A=np.array([[1.4,0.9],[0.35,1.1]]); Ainv=np.linalg.inv(A)
def draw_space(ax,M,title,sub,basecolor_grid=GREY):
    # transformed grid lines from -1..3
    for k in range(-1,4):
        p1=M@np.array([k,-1]); p2=M@np.array([k,3]); ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=basecolor_grid,lw=0.8,zorder=1)
        p1=M@np.array([-1,k]); p2=M@np.array([3,k]); ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=basecolor_grid,lw=0.8,zorder=1)
    # unit square
    sq=np.array([[0,0],[1,0],[1,1],[0,1]]).T; sq=M@sq
    ax.add_patch(Polygon(sq.T,closed=True,facecolor=BLUE,alpha=0.13,edgecolor="none",zorder=2))
    i=M@np.array([1,0]); j=M@np.array([0,1])
    ax.annotate("",xy=i,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=3),zorder=5)
    ax.annotate("",xy=j,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=3),zorder=5)
    ax.set_title(title,fontsize=12,fontweight="bold",pad=8)
    ax.text(0.5,-1.75,sub,ha="center",fontsize=10,color=DARK,transform=ax.transData)
    ax.set_xlim(-1.6,3.2); ax.set_ylim(-1.6,3.2); ax.set_aspect("equal"); ax.axis("off")
    ax.axhline(0,color="#868e96",lw=1,zorder=0); ax.axvline(0,color="#868e96",lw=1,zorder=0)

figA,axs=plt.subplots(1,3,figsize=(14,5))
draw_space(axs[0],np.eye(2),"original space","basis î=[1,0], ĵ=[0,1]")
draw_space(axs[1],A,"after A  (deform)","A bends the grid")
draw_space(axs[2],Ainv@A,"after A⁻¹  (restored)","A⁻¹ undoes A  →  A⁻¹A = I")
# green check arrows row hint
figA.suptitle("Inverse matrix = undo a transformation   (A⁻¹A = I, identity)",fontsize=13.5,fontweight="bold",y=0.99)
figA.tight_layout(rect=[0,0,1,0.95])
outA="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments/MD4-inverse 역행렬 되돌리기 (재구성).png"
figA.savefig(outA,dpi=150,bbox_inches="tight"); print("saved",outA)

# ===================== FIGURE B: special matrices map =====================
def matgrid(ax,vals,title,note,fill):
    n=len(vals)
    for r in range(n):
        for c in range(n):
            v=vals[r][c]; fc=fill(r,c,v)
            ax.add_patch(Rectangle((c,n-1-r),1,1,facecolor=fc,edgecolor="white",lw=2))
            ax.text(c+0.5,n-1-r+0.5,("" if v is None else str(v)),ha="center",va="center",
                    fontsize=11,color="white" if fc!="#f1f3f5" else "#adb5bd",fontweight="bold")
    ax.set_xlim(-0.1,n+0.1); ax.set_ylim(-0.9,n+0.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title,fontsize=11.5,fontweight="bold",pad=4)
    ax.text(n/2,-0.55,note,ha="center",fontsize=8.8,color=DARK)

NZ="#1c7ed6"; Z="#f1f3f5"; DI="#2f9e44"
figB,axs=plt.subplots(2,3,figsize=(13,8))
# 정방 square (all filled)
matgrid(axs[0,0],[[4,2,7],[5,1,9],[4,0,1]],"Square (정방)",
        "rows = cols; 선형변환의 기본", lambda r,c,v:NZ)
# 항등 identity
matgrid(axs[0,1],[[1,0,0],[0,1,0],[0,0,1]],"Identity I (항등)",
        "diag=1, else 0; '1을 곱함'=변환 취소", lambda r,c,v:(DI if r==c else Z))
# 대각 diagonal
matgrid(axs[0,2],[[4,0,0],[0,2,0],[0,0,5]],"Diagonal (대각)",
        "diag만 값; 축마다 스칼라 신축", lambda r,c,v:(DI if r==c else Z))
# 삼각 triangular (upper)
matgrid(axs[1,0],[[4,2,9],[0,1,6],[0,0,5]],"Triangular (삼각)",
        "대각+한쪽만; 연립방정식 풀기 쉬움(LU)", lambda r,c,v:(NZ if c>=r else Z))
# 희소 sparse
matgrid(axs[1,1],[[0,0,0],[0,0,2],[0,0,0]],"Sparse (희소)",
        "대부분 0; 0 안 저장→메모리 절약", lambda r,c,v:(NZ if v!=0 else Z))
# 역행렬 개념 inverse
ax=axs[1,2]; ax.axis("off")
ax.text(0.5,0.78,"Inverse  A⁻¹  (역행렬)",ha="center",fontsize=11.5,fontweight="bold",transform=ax.transAxes)
ax.text(0.5,0.52,r"$A^{-1}A = I$",ha="center",fontsize=19,transform=ax.transAxes,color=GREEN)
ax.text(0.5,0.30,"A의 변환을 '취소'하는 행렬",ha="center",fontsize=9.5,transform=ax.transAxes,color=DARK)
ax.text(0.5,0.13,"존재 조건: det(A) ≠ 0 (정방·가역)",ha="center",fontsize=9.5,transform=ax.transAxes,color="#d6336c")
figB.suptitle("Special matrices (특수 행렬) — patterns at a glance",fontsize=13.5,fontweight="bold",y=0.98)
figB.tight_layout(rect=[0,0,1,0.95])
outB="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments/MD4-special 특수행렬 6종 지도 (재구성).png"
figB.savefig(outB,dpi=150,bbox_inches="tight"); print("saved",outB)
