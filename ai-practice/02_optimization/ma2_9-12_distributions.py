"""[필수수학AI] §2.9~2.12 연속·이산 분포·결합 확률 밀도·균등·정규 분포 — 수치 검증 + 그림 재구성.

목표:
  1) 균등분포 pdf 면적=1
  2) 표준정규 pdf 피크=1/sqrt(2pi)=0.3989·면적=1·68-95-99.7 정밀값
  3) 이변량정규 pdf — 책의 원시형 수식 vs Sigma 행렬형 수식이 실제로 같은 값인지, scipy 기준과도 일치하는지 3자 대조
  4) 이변량정규 pdf가 R^2 전체에서 적분=1인지 수치 검증(이중적분)
  5) 그림 재구성 5종 + 책원본 모사(가능한 것) + Plotly 인터랙티브 페이로드(JSON)

노트: 수학/확률통계/1.5(연속·이산·결합pdf)·1.6(균등·정규)·1.7(카탈로그) — 2단계/05·06·07(렌즈)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad, dblquad

np.random.seed(42)
BLUE, ORANGE, GREEN, PINK, GRAY = "#1c7ed6", "#e8590c", "#2f9e44", "#d6336c", "#868e96"
OUT = "/sessions/pensive-epic-curie/mnt/notebooks/10_학습/수학/확률통계/attachments/"

# ① 균등분포 면적
xmin, xmax = 0.0, 1.0
area_u = (xmax - xmin) * (1.0 / (xmax - xmin))
assert np.isclose(area_u, 1.0)
print(f"[균등] 면적 = {area_u}")

# ② 표준정규 피크·면적·68-95-99.7
peak = stats.norm.pdf(0, 0, 1)
assert np.isclose(peak, 1 / np.sqrt(2 * np.pi))
area_n, _ = quad(lambda x: stats.norm.pdf(x, 0, 1), -60, 60)
assert np.isclose(area_n, 1.0)
p1 = stats.norm.cdf(1) - stats.norm.cdf(-1)
p2 = stats.norm.cdf(2) - stats.norm.cdf(-2)
p3 = stats.norm.cdf(3) - stats.norm.cdf(-3)
print(f"[정규] 피크={peak:.4f} 면적={area_n:.6f} | 68/95/99.7 정밀값: {p1*100:.2f}% {p2*100:.2f}% {p3*100:.2f}%")

# ③ 이변량정규 — 원시형 vs Sigma형 vs scipy 3자 대조
mu1, mu2, s1, s2, rho = 0.0, 0.0, 1.3, 1.3, 0.5
Sigma = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])

def g_raw(x, y):
    Sig = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])
    Sig_inv = np.linalg.inv(Sig)
    d = np.array([x-mu1, y-mu2])
    qf = d @ Sig_inv @ d
    denom = np.sqrt((2*np.pi)**2 * np.linalg.det(Sig))
    return np.exp(-0.5*qf) / denom

def g_matrix(x, y):
    u = np.array([x, y]); mu = np.array([mu1, mu2])
    Sig_inv = np.linalg.inv(Sigma)
    qf = (u-mu) @ Sig_inv @ (u-mu)
    denom = np.sqrt((2*np.pi)**2 * np.linalg.det(Sigma))
    return np.exp(-0.5*qf) / denom

rv = stats.multivariate_normal(mean=[mu1, mu2], cov=Sigma)
test_pts = [(0,0), (1,1), (-1,2), (0.5,-0.5), (2,2)]
for (x, y) in test_pts:
    a, b, c = g_raw(x,y), g_matrix(x,y), rv.pdf([x,y])
    assert np.isclose(a,b) and np.isclose(a,c), (x,y,a,b,c)
print("[이변량정규] 원시형=Sigma형=scipy 일치:", [round(g_raw(x,y),6) for x,y in test_pts])

peak_biv = g_raw(0,0)
print(f"[이변량정규] 중심 피크 = {peak_biv:.4f} (책 z축 라벨 ~0.12와 대조용, σ=1.3/ρ=0.5는 근사 선택값)")

vol, err = dblquad(lambda y,x: g_raw(x,y), -12,12, lambda x:-12, lambda x:12)
print(f"[이변량정규] 수치이중적분 = {vol:.6f} (오차 {err:.2e})")
assert abs(vol-1.0) < 1e-4

# ④ 그림 재구성
fig, ax = plt.subplots(figsize=(6,4))
x = np.linspace(-2,3,1000); y = np.where((x>=0)&(x<=1),1.0,0.0)
ax.plot(x,y,color=BLUE,lw=2.5); ax.set_ylim(-0.05,1.15); ax.set_xlim(-2,3)
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("Uniform Distribution U(0,1) pdf")
ax.grid(alpha=0.3); ax.axhline(0,color=GRAY,lw=0.8)
fig.tight_layout(); fig.savefig(OUT+"MA그림2-8 균등분포 pdf 구간[0,1] (재구성).png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5,3.2))
ax.plot(x,y,color=ORANGE,lw=2.2); ax.set_ylim(-0.05,1.1); ax.set_xlim(-2,3)
ax.set_xticks([-2,-1,0,1,2,3]); ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
fig.tight_layout(); fig.savefig(OUT+"MA그림2-8 균등분포 pdf 구간[0,1] (책원본 모사).png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6,4))
xs = np.linspace(-5,5,1000)
ax.plot(xs, stats.norm.pdf(xs,0,1), color=BLUE, lw=2.5)
ax.fill_between(xs, stats.norm.pdf(xs,0,1), 0, where=(xs>=-1)&(xs<=1), color=BLUE, alpha=0.15)
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("Normal Distribution N(0,1) pdf")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT+"MA그림2-9 표준정규분포 pdf (재구성).png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5,3.2))
ax.plot(xs, stats.norm.pdf(xs,0,1), color=ORANGE, lw=2.2)
ax.set_xlim(-5,5); ax.set_ylim(0,0.42)
fig.tight_layout(); fig.savefig(OUT+"MA그림2-9 표준정규분포 pdf (책원본 모사).png", dpi=150); plt.close(fig)

gx = np.linspace(-6,6,60); gy = np.linspace(-6,6,60)
GX, GY = np.meshgrid(gx,gy); GZ = rv.pdf(np.dstack([GX,GY]))
fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(GX,GY,GZ,cmap='Blues',linewidth=0,antialiased=True)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('g(x,y)')
ax.set_title('Bivariate Normal pdf  mu=(0,0)  sigma1=sigma2=1.3  rho=0.5')
fig.tight_layout(); fig.savefig(OUT+"MA그림2-10 이변량정규분포 pdf 3D곡면 (재구성).png", dpi=150); plt.close(fig)

fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(GX,GY,GZ,color='black',linewidth=0.4,rstride=2,cstride=2)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('g(x,y)')
fig.tight_layout(); fig.savefig(OUT+"MA그림2-10 이변량정규분포 pdf 3D곡면 (책원본 모사).png", dpi=150); plt.close(fig)

samples = rv.rvs(size=6000, random_state=42)
fig, ax = plt.subplots(figsize=(6.5,6))
ax.scatter(samples[:,0], samples[:,1], s=3, color=BLUE, alpha=0.35, linewidths=0)
sx = np.linspace(-5,5,150); sy = np.linspace(-5,5,150); SX,SY = np.meshgrid(sx,sy)
SZ = rv.pdf(np.dstack([SX,SY]))
ax.contour(SX,SY,SZ,levels=6,colors=GREEN,linewidths=1.2)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Bivariate Normal — 6,000 random samples + density contours')
ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(OUT+"MA그림2-11 이변량정규분포 산점6000+등고선 (재구성).png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(samples[:,0], samples[:,1], s=2, color='black', alpha=0.3, linewidths=0)
ax.contour(SX,SY,SZ,levels=6,colors='black',linewidths=1.0)
ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(OUT+"MA그림2-11 이변량정규분포 산점6000+등고선 (책원본 모사).png", dpi=150); plt.close(fig)

fig = plt.figure(figsize=(7.5,6)); ax = fig.add_subplot(111, projection='3d')
xs_slice = np.linspace(-5,5,200); omegas = np.linspace(-4,4,9)
colors = plt.cm.Blues(np.linspace(0.35,1.0,len(omegas)))
for om, c in zip(omegas, colors):
    zs = rv.pdf(np.column_stack([xs_slice, np.full_like(xs_slice, om)]))
    ax.plot(xs_slice, np.full_like(xs_slice, om), zs, color=c, lw=1.8)
zs0 = rv.pdf(np.column_stack([xs_slice, np.zeros_like(xs_slice)]))
ax.plot(xs_slice, np.zeros_like(xs_slice), zs0, color=GREEN, lw=3)
ax.set_xlabel('x'); ax.set_ylabel('omega (fixed y)'); ax.set_zlabel('f(x; omega)')
ax.set_title('Joint pdf slice  ~  posterior f(x2 | x1=a)')
fig.tight_layout(); fig.savefig(OUT+"MA그림2-7 결합분포 슬라이스와 사후확률 (재구성).png", dpi=150); plt.close(fig)

print("모든 그림 저장 완료")

# ⑤ Plotly 페이로드
lgx = np.linspace(-5,5,20); lgy = np.linspace(-5,5,20)
LGX, LGY = np.meshgrid(lgx,lgy); LGZ = rv.pdf(np.dstack([LGX,LGY]))
surface_payload = {
    "data":[{"type":"surface",
             "x":[[round(v,2) for v in row] for row in LGX.tolist()],
             "y":[[round(v,2) for v in row] for row in LGY.tolist()],
             "z":[[round(v,4) for v in row] for row in LGZ.tolist()],
             "colorscale":"Blues","showscale":False}],
    "layout":{"margin":{"l":0,"r":0,"t":0,"b":0},
              "scene":{"xaxis":{"title":"x"},"yaxis":{"title":"y"},"zaxis":{"title":"g(x,y)"}}}
}
with open(OUT+"_fig2-10_plotly.json","w") as f:
    json.dump(surface_payload, f)
print("Fig2-10 payload bytes:", len(json.dumps(surface_payload)))

omegas7 = np.linspace(-4,4,7); xs7 = np.linspace(-5,5,25)
palette = ["#a5d8ff","#74c0fc","#4dabf7","#1c7ed6","#1864ab","#0b4d91","#083a75"]
traces = []
for om, col in zip(omegas7, palette):
    zs = rv.pdf(np.column_stack([xs7, np.full_like(xs7, om)]))
    traces.append({"type":"scatter3d","mode":"lines",
                    "x":[round(v,2) for v in xs7.tolist()],
                    "y":[round(float(om),2)]*len(xs7),
                    "z":[round(v,4) for v in zs.tolist()],
                    "line":{"color":col,"width":4},"name":f"ω={om:.1f}"})
fig7_payload = {"data":traces,
                 "layout":{"margin":{"l":0,"r":0,"t":0,"b":0},
                           "scene":{"xaxis":{"title":"x"},"yaxis":{"title":"ω"},"zaxis":{"title":"f(x;ω)"}}}}
with open(OUT+"_fig2-7_plotly.json","w") as f:
    json.dump(fig7_payload, f)
print("Fig2-7 payload bytes:", len(json.dumps(fig7_payload)))
