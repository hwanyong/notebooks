import numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mp
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
DST="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"
V='#1c7ed6'; W='#e8590c'; R='#2f9e44'; NEG='#d6336c'
def grid(ax,xl,yl):
    ax.set_xticks(range(xl[0],xl[1]+1)); ax.set_yticks(range(yl[0],yl[1]+1))
    ax.grid(True,color='#ddd',lw=.7,ls='--'); ax.set_axisbelow(True)
    ax.axhline(0,color='#333',lw=1); ax.axvline(0,color='#333',lw=1)
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect('equal'); ax.tick_params(labelsize=8)
def arr(ax,s,e,c,lw=2.6,ls='-'):
    ax.annotate('',xy=e,xytext=s,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw,ls=ls,shrinkA=0,shrinkB=0))
fig,(axA,axB)=plt.subplots(1,2,figsize=(12,5.4))
v=np.array([3,1]); w=np.array([1,2]); r=v+w
grid(axA,(-1,5),(-1,5))
arr(axA,(0,0),v,V); axA.text(*v,'  v=(3,1)',color=V,fontsize=11,weight='bold')
arr(axA,(0,0),w,W); axA.text(*w,'  w=(1,2)',color=W,fontsize=11,weight='bold')
arr(axA,(0,0),r,R,lw=2.4); axA.text(r[0]+.05,r[1]+.15,'v+w=(4,3)',color=R,fontsize=11,weight='bold')
axA.plot([r[0],r[0]],[0,r[1]],color=R,ls=':',lw=1); axA.plot([0,r[0]],[r[1],r[1]],color=R,ls=':',lw=1)
th=np.degrees(np.arctan2(r[1],r[0]))
axA.add_patch(mp.Arc((0,0),2,2,angle=0,theta1=0,theta2=th,color='#555'))
axA.text(1.05,0.32,'theta=atan(3/4)',color='#555',fontsize=9)
axA.set_title('(A) Components add independently;\ndirection (angle) = derived ratio',fontsize=10.5)
u=np.array([3,2])
grid(axB,(-4,4),(-4,4))
arr(axB,(0,0),u,V); axB.text(*u,'  v=(3,2)',color=V,fontsize=11,weight='bold')
arr(axB,(0,0),-u,NEG,ls='--'); axB.text(-u[0]-2.0,-u[1]-.3,'-v=(-3,-2)',color=NEG,fontsize=11,weight='bold')
axB.scatter(0,0,color=R,s=70,zorder=6); axB.text(.15,.2,'v+(-v)=0  (collapse to origin)',color=R,fontsize=9.5,weight='bold')
axB.set_title('(B) Additive inverse: -v returns v to 0 (identity)',fontsize=10.5)
fig.suptitle('Figure T05 — direction is derived; negative = additive inverse',fontsize=12,y=1.0)
plt.tight_layout(); plt.savefig(f"{DST}/T05 방향파생·덧셈역원 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()
print("DONE")
