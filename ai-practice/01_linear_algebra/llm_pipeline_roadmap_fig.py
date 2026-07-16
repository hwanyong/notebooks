import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
DARK="#343a40"
VEC="#1c7ed6"; MAT="#e8590c"; STAT="#862e9c"; CALC="#d6336c"; RAW="#868e96"
LEARNED="#2f9e44"; S2="#f08c00"; S3="#4263eb"
plt.rcParams.update({"font.size":10})
fig,ax=plt.subplots(figsize=(15,8.4)); ax.set_xlim(0,15); ax.set_ylim(0,9); ax.axis("off")

def box(x,y,w,h,title,sub,fill,status,scolor):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.12",fc="white",ec=fill,lw=2.4))
    ax.add_patch(FancyBboxPatch((x,y+h-0.34),w,0.34,boxstyle="round,pad=0.0,rounding_size=0.02",fc=fill,ec=fill,lw=0))
    ax.text(x+w/2,y+h-0.17,title,ha="center",va="center",fontsize=10,color="white",fontweight="bold")
    ax.text(x+w/2,y+h/2-0.05,sub,ha="center",va="center",fontsize=8.4,color=DARK)
    ax.text(x+w/2,y+0.17,status,ha="center",va="center",fontsize=7.8,color="white",
            bbox=dict(boxstyle="round,pad=0.15",fc=scolor,ec="none"))
    return (x+w/2,y+h/2)

def arrow(p1,p2,color=DARK,ls="-",lw=2.2,rad=0.0):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=16,color=color,lw=lw,linestyle=ls,connectionstyle=f"arc3,rad={rad}"))

ax.text(7.5,8.75,"LLM pipeline — where VECTOR / MATRIX / STATISTICS control the data (learning roadmap)",
        ha="center",fontsize=13,fontweight="bold")

w,h=2.0,1.5; y=6.1
b1=box(0.2,y,w,h,"Token","text -> IDs",RAW,"input",RAW)
b2=box(2.5,y,w,h,"Embedding","token -> VECTOR x",VEC,"learned",LEARNED)
b3=box(4.8,y,w,h,"Layer  Wx+b","MATRIX x vector +nonlin",MAT,"learned",LEARNED)
b4=box(7.1,y,w,h,"x N layers","compose = matmul chain",MAT,"learned",LEARNED)
b5=box(9.4,y,w,h,"Attention","Q Kt  (MATRIX)",MAT,"stage 3",S3)
b6=box(11.7,y,w,h,"Logits","output VECTOR",VEC,"stage 3",S3)
for a,b in [(b1,b2),(b2,b3),(b3,b4),(b4,b5),(b5,b6)]: arrow((a[0]+1.0,a[1]),(b[0]-1.0,b[1]))
ax.text(6.0,7.8,"— forward pass —>",ha="center",fontsize=10,color=DARK,style="italic")

y2=3.5
b7=box(11.7,y2,w,h,"Softmax","logits -> PROBS",STAT,"stage 2",S2)
b8=box(9.4,y2,w,h,"Probabilities","STATISTICS dist.",STAT,"stage 2",S2)
b9=box(7.1,y2,w,h,"Loss","cross-entropy (STATS)",STAT,"stage 2",S2)
b10=box(4.8,y2,w,h,"Backprop","chain rule (CALCULUS)",CALC,"stage 2",S2)
b11=box(2.5,y2,w,h,"Update W","grad descent -> new MATRIX",MAT,"stage 2",S2)
arrow((b6[0],b6[1]-0.75),(b7[0],b7[1]+0.75))
for a,b in [(b7,b8),(b8,b9),(b9,b10),(b10,b11)]: arrow((a[0]-1.0,a[1]),(b[0]+1.0,b[1]))
ax.text(7.5,5.2,"<— learning: adjust the matrices —",ha="center",fontsize=10,color=CALC,style="italic")
arrow((b11[0],b11[1]+0.75),(b3[0]-0.3,b3[1]-0.75),color=CALC,ls=(0,(5,3)),lw=1.8,rad=-0.2)
ax.text(3.0,5.35,"loop: W updated every step",fontsize=8,color=CALC)

lx,ly=0.4,1.7
ax.text(lx,ly+0.5,"What controls each stage:",fontsize=9.5,fontweight="bold",color=DARK)
for i,(c,lab) in enumerate([(VEC,"VECTOR = data point/state"),(MAT,"MATRIX = transform/weights"),
                            (STAT,"STATISTICS = probability/loss"),(CALC,"CALCULUS = backprop/learning")]):
    ax.add_patch(FancyBboxPatch((lx+i*3.6,ly),0.3,0.3,boxstyle="round,pad=0.02",fc=c,ec="none"))
    ax.text(lx+i*3.6+0.42,ly+0.15,lab,fontsize=8.2,va="center",color=DARK)
sy=0.8
ax.text(lx,sy+0.15,"Status:",fontsize=9.5,fontweight="bold",color=DARK)
for i,(c,lab) in enumerate([(LEARNED,"learned (stage 1 + insight)"),(S2,"stage 2 (calculus/stats)"),(S3,"stage 3 (attention/LLM)")]):
    ax.text(lx+1.0+i*4.4,sy+0.15,f"[{lab}]",fontsize=8.2,va="center",color="white",
            bbox=dict(boxstyle="round,pad=0.2",fc=c,ec="none"))

out="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/통찰/attachments/LLM 파이프라인 로드맵 (벡터·행렬·통계 제어).png"
import os; os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches="tight"); print("saved",out)
