"""05-LA dot-product INTUITION figures.
 A: projection (shadow) geometry — a·b = shadow_length × ‖b‖
 B: 'matching score' — taste vectors (±), multiply per axis then sum (+8 vs −8)
English labels only (avoid Hangul tofu). Palette:
 a=blue #1c7ed6, b=orange #e8590c, +match=green #2f9e44, −=pink #d6336c.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, ORANGE, GREEN, PINK = "#1c7ed6", "#e8590c", "#2f9e44", "#d6336c"
GREY = "#adb5bd"
OUT = "/sessions/vigilant-beautiful-ritchie/mnt/notebooks/10_학습/AI-ML/AI 암반 정복/1단계 - 선형대수와 임베딩/attachments"

def arrow(ax, p0, p1, color, lw=2.6, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=lw, zorder=z))

# ================= Figure A: projection (shadow) =================
figA, (axM, axS) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios":[1.25,1]})

# --- main projection panel ---
a = np.array([2.0, 4.0]); b = np.array([5.0, 1.0])
t = (a@b)/(b@b); foot = t*b
axM.set_xlim(-0.5, 5.5); axM.set_ylim(-0.5, 4.6); axM.set_aspect("equal")
axM.axhline(0,color=GREY,lw=.8); axM.axvline(0,color=GREY,lw=.8)
# shadow (projection) segment
axM.plot([0,foot[0]],[0,foot[1]], color=GREEN, lw=7, alpha=.5, zorder=2,
         solid_capstyle="round")
# perpendicular drop
axM.plot([a[0],foot[0]],[a[1],foot[1]], color="#495057", lw=1.4, ls=(0,(4,3)), zorder=3)
# right-angle mark at foot
d1=(np.array([0,0])-foot); d1=d1/np.linalg.norm(d1)
d2=(a-foot); d2=d2/np.linalg.norm(d2)
sq=0.28
p=foot+d1*sq; q=foot+d2*sq; r=foot+(d1+d2)*sq
axM.plot([p[0],r[0],q[0]],[p[1],r[1],q[1]], color="#495057", lw=1.1, zorder=3)
arrow(axM,(0,0),a,BLUE); arrow(axM,(0,0),b,ORANGE)
axM.plot(*a,'o',color=BLUE,zorder=6); axM.plot(*b,'o',color=ORANGE,zorder=6)
axM.text(a[0]-.1,a[1]+.18,"a",color=BLUE,fontsize=15,fontweight="bold")
axM.text(b[0]+.1,b[1]-.05,"b",color=ORANGE,fontsize=15,fontweight="bold")
axM.text(foot[0]+.05,foot[1]-.42,"shadow of a on b\n= ‖a‖·cos θ",color=GREEN,
         fontsize=10,fontweight="bold",ha="center")
# theta arc label
axM.annotate("θ", xy=(0.55,0.32), color="#212529", fontsize=13, fontweight="bold")
axM.set_title("a · b  =  (shadow length)  ×  ‖b‖\n“how much of a points along b”, scaled by b's length",
              fontsize=12, fontweight="bold")
axM.set_xticks([]); axM.set_yticks([])

# --- side strip: 3 angle cases ---
axS.set_xlim(0,3); axS.set_ylim(0,3.2); axS.axis("off")
axS.set_title("longer shadow → bigger dot", fontsize=11, fontweight="bold")
cases=[("θ small  →  a·b > 0 (big)", 20, GREEN),
       ("θ = 90°  →  a·b = 0",       90, "#868e96"),
       ("θ > 90°  →  a·b < 0",      145, PINK)]
for i,(lab,deg,col) in enumerate(cases):
    cy=2.6-i*1.0; cx=0.55; L=0.9
    bb=np.array([L,0])
    aa=np.array([L*np.cos(np.radians(deg)), L*np.sin(np.radians(deg))])
    arrow(axS,(cx,cy),(cx+bb[0],cy+bb[1]),ORANGE,lw=2.2)
    arrow(axS,(cx,cy),(cx+aa[0],cy+aa[1]),BLUE,lw=2.2)
    # shadow on b (x-axis dir)
    sh=aa@bb/(bb@bb)*bb
    axS.plot([cx,cx+sh[0]],[cy,cy],color=col,lw=6,alpha=.55,solid_capstyle="round")
    axS.text(cx+1.25,cy,lab,fontsize=10.5,va="center",color=col,fontweight="bold")
figA.suptitle("Dot product, geometrically  =  a's shadow on b, times b's length",
              fontsize=13.5, fontweight="bold")
figA.tight_layout(rect=[0,0,1,0.95])
figA.savefig(f"{OUT}/LA1-6 dot-projection-intuition.png", dpi=150, bbox_inches="tight")
print("saved A")

# ================= Figure B: matching score (multiply→sum) =================
labels=["Sci-Fi","Action","Romance"]
A=np.array([2,2,-1]); B=np.array([2,1,-2]); C=np.array([-2,-1,2])
def panel(ax, x, y, title, va, vb, who):
    prod=va*vb; tot=prod.sum()
    ax.text(0.5,1.02,title,transform=ax.transAxes,ha="center",fontsize=12,fontweight="bold")
    rows=["axis","you (a)",f"{who} (b)","a×b"]
    # header row of taste axes
    xs=[0.30,0.52,0.74]
    for xi,lab in zip(xs,labels):
        ax.text(xi,0.90,lab,transform=ax.transAxes,ha="center",fontsize=9.5,color="#495057")
    ax.text(0.06,0.90,"axis →",transform=ax.transAxes,fontsize=9,color="#868e96")
    def rowvals(yy,name,vals,color="#212529",prod=False):
        ax.text(0.06,yy,name,transform=ax.transAxes,fontsize=9.5,fontweight="bold")
        for xi,v in zip(xs,vals):
            c = (GREEN if v>0 else PINK if v<0 else "#868e96") if prod else color
            ax.text(xi,yy,f"{v:+d}",transform=ax.transAxes,ha="center",
                    fontsize=11,fontweight="bold",color=c)
    rowvals(0.74,"you (a)",va,BLUE)
    rowvals(0.60,f"{who} (b)",vb,ORANGE)
    ax.plot([0.04,0.84],[0.52,0.52],transform=ax.transAxes,color="#ced4da",lw=1)
    rowvals(0.44,"a×b",prod,prod=True)
    # sum bar
    col = GREEN if tot>0 else PINK
    ax.text(0.06,0.24,"SUM =",transform=ax.transAxes,fontsize=10,fontweight="bold")
    ax.add_patch(Rectangle((0.30,0.18),0.012*abs(tot)*3.0,0.10,
                 transform=ax.transAxes,color=col,alpha=.85))
    ax.text(0.30,0.06,f"a · b = {tot:+d}",transform=ax.transAxes,fontsize=13,
            fontweight="bold",color=col)
    verdict = "→ tastes ALIGN (similar)" if tot>0 else "→ tastes OPPOSE (dissimilar)" if tot<0 else "→ unrelated"
    ax.text(0.30,-0.02,verdict,transform=ax.transAxes,fontsize=10,color=col,fontweight="bold")
    ax.axis("off")

figB, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4.8))
panel(ax1,0,0,"Aligned taste",A,B,"friend")
panel(ax2,0,0,"Opposite taste",A,C,"rival")
figB.suptitle("Dot product, by hand  =  multiply each axis, then add up the agreements\n"
              "(+ when both agree on an axis,  − when they disagree)",
              fontsize=12.5,fontweight="bold")
figB.tight_layout(rect=[0,0,1,0.90])
figB.savefig(f"{OUT}/LA1-7 dot-matching-score.png", dpi=150, bbox_inches="tight")
print("saved B")
