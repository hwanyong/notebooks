import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE="#1c7ed6"; ORANGE="#e8590c"; GREEN="#2f9e44"; GREY="#868e96"; DARK="#343a40"
plt.rcParams.update({"font.size":11})
fig,(axR,axC)=plt.subplots(1,2,figsize=(13,5.8))

# ===== LEFT: ROW picture — two lines intersect at (3,1) =====
axR.set_title("① row picture :  each equation = a line,  solution = intersection",
              fontsize=12, fontweight="bold", pad=10)
x=np.linspace(-1,5,200)
# x - 2y = 1  -> y=(x-1)/2 ;  3x+2y=11 -> y=(11-3x)/2
axR.plot(x,(x-1)/2,color=BLUE,lw=2.4,label="x - 2y = 1")
axR.plot(x,(11-3*x)/2,color=ORANGE,lw=2.4,label="3x + 2y = 11")
axR.plot(3,1,"o",color=GREEN,ms=12,zorder=5)
axR.annotate("solution (3, 1)",(3,1),textcoords="offset points",xytext=(12,10),
             fontsize=11,fontweight="bold",color=GREEN)
axR.axhline(0,color=GREY,lw=1); axR.axvline(0,color=GREY,lw=1)
axR.text(4.3,(4.3-1)/2-0.05,"x-2y=1",color=BLUE,fontsize=10)
axR.text(0.2,(11-3*0.2)/2+0.1,"3x+2y=11",color=ORANGE,fontsize=10)
axR.set_xlim(-1,5); axR.set_ylim(-1,6); axR.set_aspect("equal")
axR.set_xticks(range(0,6)); axR.set_yticks(range(0,7))
axR.grid(alpha=0.25)
axR.text(2.0,-1.9,"Ax = ( (rowi of A) · x ),   geometry: lines meet",
         ha="center",fontsize=10,color=DARK)

# ===== RIGHT: COLUMN picture — 3*col1 + 1*col2 = b =====
axC.set_title("② column picture :  b = x·(col1) + y·(col2),  solution = the weights",
              fontsize=12, fontweight="bold", pad=10)
col1=np.array([1,3]); col2=np.array([-2,2]); b=np.array([1,11])
def arrow(ax,start,vec,color,lw=2.6,ls="-"):
    ax.annotate("",xy=start+vec,xytext=start,
                arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,linestyle=ls))
# base columns
arrow(axC,np.zeros(2),col1,BLUE)
axC.text(col1[0]+0.1,col1[1],"col1 [1,3]",color=BLUE,fontsize=10)
arrow(axC,np.zeros(2),col2,ORANGE)
axC.text(col2[0]-0.2,col2[1]+0.2,"col2 [-2,2]",color=ORANGE,ha="right",fontsize=10)
# 3*col1 then +col2 -> b  (tip to tail)
arrow(axC,np.zeros(2),3*col1,BLUE,lw=2.0,ls=(0,(4,3)))
axC.text(3*col1[0]+0.15,3*col1[1],"3·col1 [3,9]",color=BLUE,fontsize=9.5)
arrow(axC,3*col1,col2,ORANGE,lw=2.0,ls=(0,(4,3)))
arrow(axC,np.zeros(2),b,GREEN,lw=3.0)
axC.text(b[0]+0.2,b[1],"b [1,11]",color=GREEN,fontweight="bold",fontsize=11)
axC.text(0.3,6.2,"3·col1 + 1·col2 = b",color=GREEN,fontsize=11,fontweight="bold")
axC.axhline(0,color=GREY,lw=1); axC.axvline(0,color=GREY,lw=1)
axC.set_xlim(-3,5); axC.set_ylim(-1,12); axC.set_aspect("equal")
axC.grid(alpha=0.25)
axC.text(1.0,-2.6,"weights (x,y)=(3,1) that build b = linear combination (note 06)",
         ha="center",fontsize=10,color="#862e9c")

fig.suptitle("Ax = b — same equations, two geometric pictures (Strang)",
             fontsize=13.5,fontweight="bold",y=0.995)
fig.tight_layout(rect=[0,0,1,0.95])
out="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments/MD4-rowcol Ax=b 행그림·열그림 (재구성).png"
fig.savefig(out,dpi=150,bbox_inches="tight"); print("saved",out)
