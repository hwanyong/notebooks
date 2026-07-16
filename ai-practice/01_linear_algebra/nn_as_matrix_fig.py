import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import numpy as np
BLUE="#1c7ed6"; ORANGE="#e8590c"; RED="#d6336c"; GREY="#ced4da"; DARK="#343a40"
FR="#ffe3ec"; FB="#d0ebff"
plt.rcParams.update({"font.size":11})
fig,(axL,axR)=plt.subplots(1,2,figsize=(13.5,5.8))

# ---- LEFT: neural net diagram (3 inputs -> 2 outputs, fully connected) ----
axL.set_title("Neural-net layer  (the picture)", fontsize=12.5, fontweight="bold", pad=12)
inX=0.0; outX=3.2
inY=[2.4,1.4,0.4]; outY=[2.1,0.9]
# edges (output i, input j): color (1,1)=red, (2,3)=blue, rest grey
def edge(j,i,color,lw):
    a=FancyArrowPatch((inX+0.28,inY[j]),(outX-0.28,outY[i]),arrowstyle='-',color=color,lw=lw,zorder=1)
    axL.add_patch(a)
for j in range(3):
    for i in range(2):
        c,lw=GREY,1.3
        if (i,j)==(0,0): c,lw=RED,3
        if (i,j)==(1,2): c,lw=BLUE,3
        edge(j,i,c,lw)
for j,y in enumerate(inY):
    axL.add_patch(Circle((inX,y),0.26,fc=FB,ec=BLUE,lw=2,zorder=3)); axL.text(inX,y,f"x{j+1}",ha="center",va="center",fontsize=11,color=DARK,zorder=4)
for i,y in enumerate(outY):
    axL.add_patch(Circle((outX,y),0.26,fc="#e9fbe9",ec="#2f9e44",lw=2,zorder=3)); axL.text(outX,y,f"y{i+1}",ha="center",va="center",fontsize=11,color=DARK,zorder=4)
axL.text(inX,2.95,"inputs",ha="center",fontsize=10,color=BLUE); axL.text(outX,2.65,"outputs",ha="center",fontsize=10,color="#2f9e44")
axL.text(1.6,2.62,"edge (x1→y1) = W₁₁",color=RED,fontsize=9.5,ha="center")
axL.text(1.6,-0.05,"edge (x3→y2) = W₂₃",color=BLUE,fontsize=9.5,ha="center")
axL.text(1.6,-0.6,"every node-pair has an edge = fully connected",ha="center",fontsize=9.5,color=DARK)
axL.set_xlim(-0.7,4.0); axL.set_ylim(-0.9,3.3); axL.axis("off")

# ---- RIGHT: the weight matrix W (2x3) ----
axR.set_title("Weight matrix  W  (the same thing)", fontsize=12.5, fontweight="bold", pad=12)
cw=ch=0.9; x0=1.0; y0=2.2
vals=[["W₁₁","W₁₂","W₁₃"],["W₂₁","W₂₂","W₂₃"]]
for i in range(2):
    for j in range(3):
        fc="white"
        if (i,j)==(0,0): fc=RED
        if (i,j)==(1,2): fc=BLUE
        x=x0+j*cw; y=y0-i*ch
        axR.add_patch(Rectangle((x,y),cw,ch,fc=fc,ec=GREY,lw=1.4))
        axR.text(x+cw/2,y+ch/2,vals[i][j],ha="center",va="center",fontsize=12,
                 color="white" if fc in(RED,BLUE) else DARK,fontweight="bold")
# brackets
xr=x0+3*cw
for xx,s in ((x0-0.06,1),(xr+0.06,-1)):
    axR.plot([xx,xx],[y0+ch,y0-ch],color=DARK,lw=2)
    axR.plot([xx,xx+0.09*s],[y0+ch,y0+ch],color=DARK,lw=2); axR.plot([xx,xx+0.09*s],[y0-ch,y0-ch],color=DARK,lw=2)
axR.text(x0-0.55,y0+ch/2-ch/2+ch/2,"rows = outputs (y)",rotation=90,ha="center",va="center",fontsize=9.5,color="#2f9e44")
axR.text(x0+1.5*cw,y0+ch+0.18,"cols = inputs (x)",ha="center",fontsize=9.5,color=BLUE)
axR.text(x0+1.5*cw,y0-1.45,"Wᵢⱼ = weight of edge (input j → output i)",ha="center",fontsize=10,color=DARK)
axR.text(x0+1.5*cw,y0-1.95,"one layer:  y = W x   (matrix × vector)",ha="center",fontsize=11,color="#862e9c",fontweight="bold")
axR.set_xlim(0.0,xr+1.0); axR.set_ylim(0.0,3.3); axR.set_aspect("equal"); axR.axis("off")

fig.suptitle("A neural network IS a matrix — edges = weights = matrix entries",fontsize=13.5,fontweight="bold",y=0.99)
fig.tight_layout(rect=[0,0,1,0.94])
out="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/통찰/attachments/신경망 그림=가중치 행렬 대응 (재구성).png"
import os; os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches="tight"); print("saved",out)
