"""06 (실전선대 §2) figures: span / independence / basis.
English labels only (avoid Hangul tofu). Palette:
 v=blue #1c7ed6, w=orange #e8590c, span/plane=green #2f9e44, accent=pink #d6336c.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BLUE, ORANGE, GREEN, PINK, GREY = "#1c7ed6", "#e8590c", "#2f9e44", "#d6336c", "#adb5bd"
OUT = "/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"

def ar(ax, p0, p1, c, lw=2.6, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, color=c, lw=lw, zorder=z))

# LA2-1: one vector + its 1D span (line)
fig, ax = plt.subplots(figsize=(5.2,5))
v=np.array([1,3])
ax.plot([-2*v[0],2.2*v[0]],[-2*v[1],2.2*v[1]], color=GREEN, lw=8, alpha=.30, zorder=1, solid_capstyle="round")
ar(ax,(0,0),v,BLUE)
ax.text(v[0]+.1,v[1],"v",color=BLUE,fontsize=15,fontweight="bold")
ax.text(2.0,5.6,"span{v} = a LINE\n(all scalar multiples)",color=GREEN,fontsize=10,fontweight="bold",ha="center")
ax.set_xlim(-4,4); ax.set_ylim(-7,7); ax.set_aspect("equal")
ax.axhline(0,color=GREY,lw=.8); ax.axvline(0,color=GREY,lw=.8); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Fig 2-1  one vector spans a 1-D subspace (line)", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/LA2-1 span-1d-line.png", dpi=150, bbox_inches="tight"); print("2-1")

# LA2-2: two independent vectors in R3 -> 2D plane
fig=plt.figure(figsize=(5.6,5)); ax=fig.add_subplot(111,projection="3d")
a=np.array([1,0,2]); b=np.array([-1,1,2])
# plane patch spanned by a,b
s=np.linspace(-1.6,1.6,2); S,T=np.meshgrid(s,s)
P=np.array([S.ravel(),T.ravel()])
pts=(a.reshape(3,1)*P[0]+b.reshape(3,1)*P[1])
X=pts[0].reshape(2,2); Y=pts[1].reshape(2,2); Z=pts[2].reshape(2,2)
ax.plot_surface(X,Y,Z,color=GREEN,alpha=.25,zorder=1)
for vec,c in [(a,BLUE),(b,ORANGE)]:
    ax.quiver(0,0,0,*vec,color=c,lw=2.5,arrow_length_ratio=.12)
ax.text(*(a*1.1),"a",color=BLUE,fontsize=13,fontweight="bold")
ax.text(*(b*1.1),"b",color=ORANGE,fontsize=13,fontweight="bold")
ax.set_title("Fig 2-2  two independent vectors\nspan a 2-D plane in R³", fontsize=11, fontweight="bold")
ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]); ax.view_init(22,35)
fig.tight_layout(); fig.savefig(f"{OUT}/LA2-2 span-2d-plane.png", dpi=150, bbox_inches="tight"); print("2-2")

# LA2-3: two DEPENDENT vectors (collinear) -> still 1D line
fig, ax = plt.subplots(figsize=(5.2,5))
v=np.array([1,1]); w=np.array([2,2])  # w = 2v (dependent)
ax.plot([-3,4],[-3,4], color=PINK, lw=8, alpha=.25, zorder=1, solid_capstyle="round")
ar(ax,(0,0),w,ORANGE,lw=2.4); ar(ax,(0,0),v,BLUE,lw=2.8)
ax.text(v[0]+.05,v[1]-.3,"v",color=BLUE,fontsize=14,fontweight="bold")
ax.text(w[0]+.1,w[1],"w = 2v",color=ORANGE,fontsize=13,fontweight="bold")
ax.text(-2.8,3.4,"DEPENDENT → span is still\njust a 1-D line",color=PINK,fontsize=10,fontweight="bold")
ax.set_xlim(-3.5,4.5); ax.set_ylim(-3.5,4.5); ax.set_aspect("equal")
ax.axhline(0,color=GREY,lw=.8); ax.axvline(0,color=GREY,lw=.8); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Fig 2-3  dependent vectors collapse the span", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/LA2-3 dependent-1d.png", dpi=150, bbox_inches="tight"); print("2-3")

# LA2-4: same points p,q in two bases (standard S vs skewed T)
fig,ax=plt.subplots(figsize=(6,5))
p=np.array([3,1]); q=np.array([-6,2])
# standard basis
ar(ax,(0,0),(1,0),GREY,lw=2,z=2); ar(ax,(0,0),(0,1),GREY,lw=2,z=2)
# skewed basis T = [3,1],[-3,1]
t1=np.array([3,1]); t2=np.array([-3,1])
ar(ax,(0,0),t1,GREEN,lw=2.2,z=3); ar(ax,(0,0),t2,ORANGE,lw=2.2,z=3)
ax.plot(*p,'o',color="#212529",ms=9,zorder=5); ax.text(p[0]+.2,p[1]-.5,"p",fontsize=13,fontweight="bold")
ax.plot(*q,'s',color=BLUE,ms=9,zorder=5); ax.text(q[0]+.2,q[1]+.3,"q",color=BLUE,fontsize=13,fontweight="bold")
ax.text(3.1,1.2,"t1",color=GREEN,fontweight="bold"); ax.text(-3.8,1.2,"t2",color=ORANGE,fontweight="bold")
ax.set_xlim(-8,5); ax.set_ylim(-2,4); ax.set_aspect("equal")
ax.axhline(0,color=GREY,lw=.7); ax.axvline(0,color=GREY,lw=.7); ax.grid(alpha=.2)
ax.set_title("Fig 2-4  same points p,q, different bases\n(gray=standard S, green/orange=basis T)", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/LA2-4 same-points-two-bases.png", dpi=150, bbox_inches="tight"); print("2-4")

# LA2-5: PCA vs ICA basis on 2D data cloud (two panels)
rng=np.random.default_rng(0)
t=rng.normal(0,1,400); data=np.stack([t*2+rng.normal(0,.4,400), t*1+rng.normal(0,.4,400)])
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4.5))
for axx,title,basis in [(a1,"PCA basis (orthogonal)",[(2,1),(-1,2)]),(a2,"ICA basis (data-aligned)",[(2,1),(1,1.7)])]:
    axx.scatter(data[0],data[1],s=6,color=GREY,alpha=.5)
    for vec,c in zip(basis,[BLUE,ORANGE]):
        vv=np.array(vec)/np.linalg.norm(vec)*3
        axx.plot([-vv[0],vv[0]],[-vv[1],vv[1]],color=c,lw=3)
    axx.set_title(title,fontsize=11,fontweight="bold"); axx.set_aspect("equal")
    axx.set_xticks([]);axx.set_yticks([]); axx.axhline(0,color=GREY,lw=.5); axx.axvline(0,color=GREY,lw=.5)
fig.suptitle("Fig 2-5  different bases describe the SAME data (no single 'best')", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(f"{OUT}/LA2-5 pca-ica-basis.png", dpi=150, bbox_inches="tight"); print("2-5")

# LA2-6: a point r outside the 1D span (can't be measured by that basis)
fig,ax=plt.subplots(figsize=(5.2,5))
v=np.array([1,3])
ax.plot([-2*v[0],2.2*v[0]],[-2*v[1],2.2*v[1]], color=GREEN, lw=8, alpha=.30, zorder=1, solid_capstyle="round")
ar(ax,(0,0),v,BLUE)
ax.plot(2.5,1.0,'o',color=PINK,ms=11,zorder=6); ax.text(2.7,1.0,"r (off the line)",color=PINK,fontsize=11,fontweight="bold")
ax.text(-3.6,5.6,"span{v} can't reach r\n→ v is NOT a basis for r's space",color=PINK,fontsize=9.5,fontweight="bold")
ax.set_xlim(-4,5); ax.set_ylim(-7,7); ax.set_aspect("equal")
ax.axhline(0,color=GREY,lw=.8); ax.axvline(0,color=GREY,lw=.8); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Fig 2-6  you can only measure what the basis spans", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/LA2-6 outside-span.png", dpi=150, bbox_inches="tight"); print("2-6")
print("done")
