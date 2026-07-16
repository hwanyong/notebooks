import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
DST="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"
A='#1c7ed6'; B='#e8590c'
def arr(ax,e,c):
    ax.annotate('',xy=e,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=c,lw=2.6,shrinkA=0,shrinkB=0))
cases=[("Acute  theta<90","cos>0  ->  dot>0",1.0,40),
       ("Obtuse  theta>90","cos<0  ->  dot<0",1.0,140),
       ("Right  theta=90","cos=0  ->  dot=0",1.0,90),
       ("Parallel  theta=0","cos=1  ->  dot=+|a||b|",1.0,0),
       ("Anti-parallel  theta=180","cos=-1  ->  dot=-|a||b|",1.0,180)]
fig,axs=plt.subplots(1,5,figsize=(15,3.4))
for ax,(t,s,bl,ang) in zip(axs,cases):
    a=np.array([1.4,0.0])                      # b vector along x (orange)
    rad=np.radians(ang); v=1.4*np.array([np.cos(rad),np.sin(rad)])  # a vector (blue)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axhline(0,color='#ccc',lw=.6); ax.axvline(0,color='#ccc',lw=.6)
    ax.set_xlim(-1.8,1.8); ax.set_ylim(-1.8,1.8); ax.set_aspect('equal')
    arr(ax,a,B); arr(ax,v,A)
    if ang not in (0,180): 
        import matplotlib.patches as mp
        ax.add_patch(mp.Arc((0,0),1.0,1.0,theta1=0,theta2=ang,color='#555',lw=1))
    ax.text(0,-1.6,s,ha='center',fontsize=9,color='#333')
    ax.set_title(t,fontsize=9.5)
axs[0].text(1.45,0.05,'b',color=B,fontsize=11,weight='bold')
fig.suptitle('Figure 1-5 (reconstructed) — dot-product sign = angle (a:blue, b:orange)',fontsize=11,y=1.02)
plt.tight_layout(); plt.savefig(f"{DST}/LA1-5 내적 부호와 각도 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()
print("DONE")
