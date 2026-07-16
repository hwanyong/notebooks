import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
DST="/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"
V='#1c7ed6'; W='#e8590c'; R='#2f9e44'; NEG='#d6336c'

def grid2d(ax,xlim,ylim):
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1)); ax.set_yticks(np.arange(ylim[0],ylim[1]+1))
    ax.grid(True,color='#ccc',lw=0.7); ax.set_axisbelow(True)
    ax.axhline(0,color='#333',lw=1.1); ax.axvline(0,color='#333',lw=1.1)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.tick_params(labelsize=8)
def vec(ax,s,e,c,lw=2.6,ls='-'):
    ax.annotate('',xy=e,xytext=s,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw,ls=ls,shrinkA=0,shrinkB=0))
def lbl(ax,p,t,c,dx=0.15,dy=0.15,fs=11):
    ax.text(p[0]+dx,p[1]+dy,t,color=c,fontsize=fs,weight='bold')

# ── 1a. 그림 4-1 + 4-2 : 기본 벡터 + 여러 예시 ──────────────────
fig,axs=plt.subplots(1,4,figsize=(13,3.4))
data=[([3,2],V,'[3, 2]'),([3,4],V,'[3, 4]'),([-3,2],W,'[-3, 2]'),([-4,-4],NEG,'[-4, -4]')]
for ax,(v,c,t) in zip(axs,data):
    grid2d(ax,(-5,5),(-5,5)); vec(ax,(0,0),v,c)
    lbl(ax,v,f'v={t}',c,dx=0.2 if v[0]>=0 else -1.6,dy=0.25)
    ax.set_title(t,fontsize=10)
fig.suptitle('Figure 4-1 / 4-2  —  A vector = direction + length (positive & negative components)',fontsize=11,y=1.02)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-1·2 벡터 기본·여러예시 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 1b. 그림 4-3 : 3차원 벡터 [4,1,2] + i,j,k ──────────────────
fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111,projection='3d')
v=np.array([4,1,2])
# basis
for b,c,t in [((1,0,0),'#2f9e44','i'),((0,1,0),'#d6336c','j'),((0,0,1),'#7048e8','k')]:
    ax.quiver(0,0,0,*b,color=c,lw=2,arrow_length_ratio=0.25)
    ax.text(*(np.array(b)*1.25),t,color=c,fontsize=12,weight='bold')
ax.quiver(0,0,0,*v,color=V,lw=3,arrow_length_ratio=0.12)
ax.text(*(v*1.05),'v=[4,1,2]',color=V,fontsize=12,weight='bold')
# 끝점 좌표 읽기용: 각 축 절편까지 점선 가이드 (축 정렬)
px,py,pz=v; g=dict(color='#555',lw=1.1,ls='--',alpha=0.9)
ax.plot([px,px],[py,py],[pz,0],**g)     # 끝점→바닥(xy평면)
ax.plot([px,px],[py,0],[0,0],**g)       # 바닥→x축 절편
ax.plot([px,0],[py,py],[0,0],**g)       # 바닥→y축 절편
ax.plot([px,0],[py,0],[pz,pz],**g)      # 끝점→z축 절편
for pt,lab,c in [((px,0,0),f'x={px}','#2f9e44'),((0,py,0),f'y={py}','#d6336c'),((0,0,pz),f'z={pz}','#7048e8')]:
    ax.scatter(*pt,color=c,s=26); ax.text(*pt,'  '+lab,color=c,fontsize=10,weight='bold')
ax.set_xlim(0,4.5);ax.set_ylim(0,4.5);ax.set_zlim(0,4.5)
ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
ax.set_title('Figure 4-3 — 3D vector [4,1,2] with axis-intercept guides',fontsize=10.5)
ax.view_init(elev=20,azim=-58)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-3 3차원 벡터 (재구성).png",dpi=150); plt.close()

# ── 2. 그림 4-4 + 4-5 : 덧셈 head-to-tail ──────────────────────
fig,ax=plt.subplots(figsize=(6.2,5)); grid2d(ax,(-1,6),(-2,4))
v=np.array([3,2]); w=np.array([2,-1]); r=v+w
vec(ax,(0,0),v,V); lbl(ax,v/2,'v=[3,2]',V,dx=-1.4,dy=0.3)
vec(ax,v,r,W); lbl(ax,(v+r)/2,'w=[2,-1]',W,dx=0.1,dy=0.25)
vec(ax,(0,0),r,R,lw=2.2,ls='--'); lbl(ax,r/2,'v+w=[5,1]',R,dx=-0.6,dy=-0.55)
ax.scatter(*r,color=R,zorder=6)
ax.set_title('Figure 4-4 / 4-5 — Addition = head-to-tail',fontsize=11)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-4·5 덧셈 head-to-tail (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 3. 그림 4-6 : 교환법칙 (parallelogram) ──────────────────────
fig,ax=plt.subplots(figsize=(6.2,5)); grid2d(ax,(-1,6),(-2,4))
vec(ax,(0,0),v,V); lbl(ax,v/2,'v',V,dx=-0.5,dy=0.3)
vec(ax,v,r,W,ls='--'); lbl(ax,(v+r)/2,'w',W,dx=0.15,dy=0.2)
vec(ax,(0,0),w,W,ls='--'); lbl(ax,w/2,'w',W,dx=-0.2,dy=-0.6)
vec(ax,w,r,V); lbl(ax,(w+r)/2,'v',V,dx=0.2,dy=-0.1)
vec(ax,(0,0),r,R,lw=2.4); lbl(ax,r/2,'v+w = w+v = [5,1]',R,dx=-1.1,dy=0.35)
ax.scatter(*r,color=R,zorder=6)
ax.set_title('Figure 4-6 — Commutativity v+w = w+v',fontsize=11)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-6 교환법칙 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 4. 그림 4-7·8·9 : 스케일링 (2v, 0.5v, -v) ──────────────────
fig,axs=plt.subplots(1,3,figsize=(13,4.2))
a=np.array([3,1])
# 2v
grid2d(axs[0],(-1,7),(-1,4)); vec(axs[0],(0,0),2*a,W,ls='--'); vec(axs[0],(0,0),a,V)
lbl(axs[0],a,'v=[3,1]',V,dx=-1.3,dy=0.25); lbl(axs[0],2*a,'2v=[6,2]',W,dx=-0.4,dy=0.3); axs[0].set_title('2·v  (stretch)')
# 0.5v
grid2d(axs[1],(-1,5),(-1,3)); vec(axs[1],(0,0),a,V); vec(axs[1],(0,0),0.5*a,NEG,ls='--')
lbl(axs[1],a,'v',V,dx=0.15,dy=0.2); lbl(axs[1],0.5*a,'0.5v=[1.5,0.5]',NEG,dx=-1.0,dy=-0.5); axs[1].set_title('0.5·v  (shrink)')
# -v
b=np.array([3,2]); grid2d(axs[2],(-5,5),(-4,4)); vec(axs[2],(0,0),b,V); vec(axs[2],(0,0),-b,NEG,ls='--')
lbl(axs[2],b,'v=[3,2]',V,dx=-0.3,dy=0.3); lbl(axs[2],-b,'-v (flip, same line)',NEG,dx=-2.0,dy=-0.7); axs[2].set_title('-1·v  (flip direction)')
fig.suptitle('Figure 4-7 / 4-8 / 4-9 — Scaling: size changes, direction stays (negative flips, still collinear)',fontsize=11,y=1.02)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-7·8·9 스케일링 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 5. 그림 4-10 : 스팬 (6 combos of v,w reach anywhere) ────────
v0=np.array([2,1.0]); w0=np.array([-1,1.5])
combos=[(1,1),(2,0.5),(-0.5,2),(1.5,-1),(-1,-1),(0.5,1.5)]
fig,axs=plt.subplots(2,3,figsize=(12,7.2))
for ax,(s,t) in zip(axs.ravel(),combos):
    grid2d(ax,(-5,5),(-5,5))
    sv=s*v0; r=s*v0+t*w0
    vec(ax,(0,0),sv,V); vec(ax,sv,r,W,ls='--'); vec(ax,(0,0),r,R,lw=2.0)
    ax.scatter(*r,color=R,zorder=6)
    ax.set_title(f'{s}·v + {t}·w',fontsize=9)
fig.suptitle('Figure 4-10 — Span: scaling & adding two independent vectors reaches ANY point in the plane',fontsize=11,y=1.0)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-10 스팬 (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 6. 그림 4-11 : 선형종속 2D (collinear → span = a line) ──────
fig,axs=plt.subplots(1,3,figsize=(13,4.2))
base=np.array([2,1.2])
pairs=[(1.0,2.0),(0.7,1.6),(1.3,-1.0)]  # both multiples of base => dependent
for ax,(s,t) in zip(axs,pairs):
    grid2d(ax,(-5,5),(-4,4))
    # draw the spanning line (dashed gray) through base direction
    xs=np.array([-4,4]); ys=base[1]/base[0]*xs
    ax.plot(xs,ys,color='#aaa',lw=1,ls=':')
    vec(ax,(0,0),s*base,V); vec(ax,(0,0),t*base,W,ls='--')
    ax.set_title(f'{s}·u  and  {t}·u  (same line)',fontsize=9)
fig.suptitle('Figure 4-11 — Linear DEPENDENCE: both vectors on one line → span collapses to that line',fontsize=11,y=1.02)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-11 선형종속 2D (재구성).png",dpi=150,bbox_inches='tight'); plt.close()

# ── 7. 그림 4-12 : 선형종속 3D (trapped on a plane) ─────────────
fig=plt.figure(figsize=(6.2,5)); ax=fig.add_subplot(111,projection='3d')
v1=np.array([2,1,0.5]); v2=np.array([1,2,0.5])
s=np.linspace(-1.3,1.3,12); t=np.linspace(-1.3,1.3,12); S,T=np.meshgrid(s,t)
P=S[...,None]*v1+T[...,None]*v2
ax.plot_surface(P[...,0],P[...,1],P[...,2],alpha=0.22,color='#6aa9e0')
def guide3d(P, lc):                       # 끝점→각 축 절편 점선 가이드
    px,py,pz=P; g=dict(color=lc,lw=1.0,ls='--',alpha=0.55)
    ax.plot([px,px],[py,py],[pz,0],**g)   # 바닥으로 드롭
    ax.plot([px,px],[py,0],[0,0],**g)     # →x축 절편
    ax.plot([px,0],[py,py],[0,0],**g)     # →y축 절편
    ax.plot([px,0],[py,0],[pz,pz],**g)    # →z축 절편
    for pt in [(px,0,0),(0,py,0),(0,0,pz)]:
        ax.scatter(*pt,color=lc,s=14,alpha=0.85)
for vv,c,lab in [(v1,NEG,'v'),(v2,V,'w')]:
    ax.quiver(0,0,0,*vv,color=c,lw=3,arrow_length_ratio=0.13)
    ax.text(*(vv*1.13),f'{lab}=({vv[0]:g},{vv[1]:g},{vv[2]:g})',color=c,fontsize=9.5,weight='bold')
    guide3d(vv,c)
for b,c in [((1.6,0,0),'gray'),((0,1.6,0),'gray'),((0,0,1.6),'gray')]:
    ax.quiver(0,0,0,*b,color=c,lw=0.8,arrow_length_ratio=0.06)
ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
ax.set_title('Figure 4-12 — Dependent in 3D → span trapped on a 2D plane',fontsize=10.5)
ax.view_init(elev=22,azim=-60)
plt.tight_layout(); plt.savefig(f"{DST}/MD4-12 선형종속 3D (재구성).png",dpi=150); plt.close()

print("ALL FIGURES DONE")
import os
for f in sorted(os.listdir(DST)):
    if '재구성' in f: print(" -",f)
