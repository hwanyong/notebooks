"""필수수학 §4.3~4.4 figures: matrix mult = composition, determinant = area ratio.
English labels only. Palette: i=green #2f9e44, j=orange #e8590c, area=blue #1c7ed6, neg=pink #d6336c."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

GREEN, ORANGE, BLUE, PINK, GREY = "#2f9e44", "#e8590c", "#1c7ed6", "#d6336c", "#adb5bd"
OUT = "/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"

def ar(ax,p0,p1,c,lw=2.6,z=5):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=15,color=c,lw=lw,zorder=z))

def basis_panel(ax, i_hat, j_hat, title, area_color=BLUE):
    para=Polygon([[0,0],i_hat,np.add(i_hat,j_hat),j_hat],closed=True,
                 facecolor=area_color,alpha=.30,edgecolor=area_color,lw=1.5,zorder=2)
    ax.add_patch(para)
    ar(ax,(0,0),i_hat,GREEN); ar(ax,(0,0),j_hat,ORANGE)
    det=i_hat[0]*j_hat[1]-i_hat[1]*j_hat[0]
    ax.text(0.5,-0.18,f"det = {det:.0f}",transform=ax.transAxes,ha="center",
            fontsize=12,fontweight="bold",color=(PINK if det<0 else area_color if det!=0 else "#212529"))
    ax.set_title(title,fontsize=10.5,fontweight="bold")
    ax.set_xlim(-3,3.5); ax.set_ylim(-3,3.5); ax.set_aspect("equal")
    ax.axhline(0,color=GREY,lw=.6); ax.axvline(0,color=GREY,lw=.6); ax.grid(alpha=.15)
    ax.set_xticks(range(-3,4)); ax.set_yticks(range(-3,4)); ax.tick_params(labelsize=7)

# === Fig A: determinant = area ratio (3 cases) ===
figA,(a1,a2,a3)=plt.subplots(1,3,figsize=(13,4.6))
basis_panel(a1,np.array([3,0]),np.array([0,2]),"Scale ×3,×2 → area ×6\n(det = 6)")
basis_panel(a2,np.array([-2,1]),np.array([1,2]),"Flip orientation → det < 0\n(space flipped)",area_color=PINK)
basis_panel(a3,np.array([3,-1.5]),np.array([-2,1]),"Collapse to a line → det = 0\n(linear dependence)",area_color=GREY)
figA.suptitle("Determinant = how much a transform scales AREA (sign = orientation, 0 = collapse)",
              fontsize=12.5,fontweight="bold")
figA.tight_layout(rect=[0,0,1,0.93])
figA.savefig(f"{OUT}/MD4-20·22·23 행렬식 면적배율 (재구성).png",dpi=150,bbox_inches="tight"); print("A det")

# === Fig B: matrix mult = composition (order matters) ===
figB,(b1,b2)=plt.subplots(1,2,figsize=(11,4.8))
v=np.array([1,2])
T1=np.array([[0,-1],[1,0]])   # rotate 90
T2=np.array([[1,1],[0,1]])    # shear
def comp_panel(ax,first,second,labels,title):
    r=first@v; s=second@r
    ar(ax,(0,0),v,GREY,lw=2.2); ax.text(*(v*1.05),"v",color="#495057",fontsize=11,fontweight="bold")
    ar(ax,(0,0),r,GREEN,lw=2.2); ax.text(*(r*1.05),labels[0],color=GREEN,fontsize=10,fontweight="bold")
    ar(ax,(0,0),s,BLUE,lw=2.8); ax.text(*(s*1.05),labels[1],color=BLUE,fontsize=11,fontweight="bold")
    ax.set_title(title,fontsize=10.5,fontweight="bold")
    ax.set_xlim(-4,4); ax.set_ylim(-1,4.5); ax.set_aspect("equal")
    ax.axhline(0,color=GREY,lw=.6); ax.axvline(0,color=GREY,lw=.6); ax.grid(alpha=.15)
comp_panel(b1,T1,T2,["rotate(v)","shear(rotate(v))"],"shear ∘ rotate  (T2·T1)")
comp_panel(b2,T2,T1,["shear(v)","rotate(shear(v))"],"rotate ∘ shear  (T1·T2)")
figB.suptitle("Matrix product = composing transforms — ORDER MATTERS (T2·T1 ≠ T1·T2)",
              fontsize=12.5,fontweight="bold")
figB.tight_layout(rect=[0,0,1,0.93])
figB.savefig(f"{OUT}/MD4-compose 행렬곱 합성 비교환 (재구성).png",dpi=150,bbox_inches="tight"); print("B compose")
print("done")
