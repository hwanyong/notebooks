import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
DST="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"
V='#1c7ed6'; W='#e8590c'; R='#2f9e44'; NEG='#d6336c'
def grid(ax,lim=6):
    ax.set_xticks(range(-lim,lim+1,2)); ax.set_yticks(range(-lim,lim+1,2))
    ax.grid(True,color='#ccc',lw=.7,ls='--'); ax.set_axisbelow(True)
    ax.axhline(0,color='#333',lw=1); ax.axvline(0,color='#333',lw=1)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.tick_params(labelsize=8)
def arr(ax,s,e,c,lw=2.6,ls='-'):
    ax.annotate('',xy=e,xytext=s,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw,ls=ls,shrinkA=0,shrinkB=0))
def lbl(ax,p,t,c,dx=.2,dy=.2,fs=11):
    ax.text(p[0]+dx,p[1]+dy,t,color=c,fontsize=fs,weight='bold')

# ── 그림 1-2 재구성: 덧셈(머리-꼬리) · 뺄셈(머리 잇기) ──
v=np.array([1,2]); w=np.array([4,-6])
fig,(axA,axB)=plt.subplots(1,2,figsize=(11,5.2))
# A: v+w (head-to-tail)
grid(axA); r=v+w
arr(axA,(0,0),v,V); lbl(axA,v,'v=[1,2]',V,dx=-2.0,dy=.1)
arr(axA,v,r,W); lbl(axA,(v+r)/2,'w=[4,-6]',W,dx=.2,dy=.2)            # 머리-꼬리
arr(axA,(0,0),r,R,lw=2.2); lbl(axA,r,'v+w=[5,-4]',R,dx=-1.0,dy=-.6)
axA.scatter(*r,color=R,zorder=5)
axA.set_title('(A) Addition v+w  —  head-to-tail',fontsize=10.5)
# B: v-w (connect heads: w-head -> v-head)
grid(axB)
arr(axB,(0,0),v,V); lbl(axB,v,'v=[1,2]',V,dx=-2.0,dy=.1)
arr(axB,(0,0),w,W); lbl(axB,w/2,'w=[4,-6]',W,dx=.2,dy=-.3)
arr(axB,w,v,R,lw=2.2); lbl(axB,(w+v)/2,'v−w=[-3,8]',R,dx=.25,dy=0)   # w머리→v머리
axB.scatter(*v,color=R,zorder=5)
axB.set_title('(B) Subtraction v−w  —  connect heads (w→v)',fontsize=10.5)
fig.suptitle('Figure 1-2 (reconstructed) — Vector addition & subtraction',fontsize=12,y=1.0)
plt.tight_layout(); plt.savefig(f"{DST}/LA1-2 벡터 덧셈·뺄셈 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 그림 1-3 재구성: 동일 벡터 × 여러 스칼라 σ ──
base=np.array([0.7,1.0]); sigmas=[1.00,2.00,0.33,0.00,-0.67]
fig,axs=plt.subplots(1,5,figsize=(14,3.1))
for ax,s in zip(axs,sigmas):
    ax.set_xticks([-2,-1,0,1,2]); ax.set_yticks([-2,-1,0,1,2])
    ax.grid(True,color='#ddd',lw=.6,ls='--'); ax.set_axisbelow(True)
    ax.axhline(0,color='#333',lw=.8); ax.axvline(0,color='#333',lw=.8)
    ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect('equal'); ax.tick_params(labelsize=7)
    sv=s*base; c = NEG if s<0 else (V if s!=0 else '#888')
    arr(ax,(0,0),base,'#222',lw=2.0)                  # 원본(검정)
    if abs(s)>1e-9: arr(ax,(0,0),sv,c,lw=3.0,ls='-')  # 스케일된(색)
    else: ax.scatter(0,0,color=c,s=30)                # σ=0 → 점
    ax.set_title(f'σ = {s:.2f}',fontsize=10)
fig.suptitle('Figure 1-3 (reconstructed) — same vector × scalar σ (black=v, color=σ·v; σ<0 flips)',fontsize=11,y=1.02)
plt.tight_layout(); plt.savefig(f"{DST}/LA1-3 스칼라-벡터 곱 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()
print("DONE")
