"""
[실전선대] §1.3 보조 시각화 — 단위벡터의 기하 (위치벡터·단위원·광선 투영·초구면)
이해용 그림 2장 생성. (노트 02-LA-1.01 임베드용)
실행: python la1_3_unit_geometry.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def fig_radial_projection(path):
    """단위원 + 광선 투영: 여러 비단위벡터가 같은 단위벡터 하나로 (여럿→하나)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#1f77b4", lw=2, label="unit circle ||v||=1")

    # 같은 방향(0.6,0.8) 위의 여러 비단위벡터 → 단위원 위 같은 점으로 수렴
    d = np.array([0.6, 0.8])
    for L, c in [(2.6, "#d62728"), (1.8, "#ff7f0e")]:
        p = d * L
        ax.annotate("", xy=p, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color=c, alpha=.8))
    ax.plot(*d, "ko", ms=8)                       # 공통 단위벡터(짝꿍)
    ax.annotate("unit vector (0.6,0.8)\n= shared direction", xy=d, xytext=(0.75, 1.45),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("(3,4) etc. — many non-unit\nvectors on ONE ray", xy=d * 2.6,
                xytext=(0.2, 2.4), fontsize=9, color="#d62728")

    # 다른 방향 두 개(광선)도 각자 단위원 위 한 점
    for ang in (140, 250):
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        ax.annotate("", xy=u * 2.2, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#7f7f7f", alpha=.6))
        ax.plot(*u, "o", color="#1f77b4", ms=7)

    ax.plot(0, 0, "k+", ms=12)                    # 원점(위치벡터 기준)
    ax.annotate("origin\n(all tails here)", xy=(0, 0), xytext=(0.15, -0.55), fontsize=8)
    ax.axhline(0, ls="--", c="gray", lw=.6); ax.axvline(0, ls="--", c="gray", lw=.6)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_title("Radial projection: many non-unit vectors → one unit vector")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); print("saved:", path)


def fig_circle_sphere(path):
    """단위원(2D) → 단위구(3D) → 초구면(ND) 확장."""
    fig = plt.figure(figsize=(10, 5))
    # 2D unit circle
    ax1 = fig.add_subplot(1, 2, 1)
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(th), np.sin(th), color="#1f77b4", lw=2)
    ax1.plot(0, 0, "k+", ms=10); ax1.set_aspect("equal")
    ax1.axhline(0, ls="--", c="gray", lw=.6); ax1.axvline(0, ls="--", c="gray", lw=.6)
    ax1.set_title("2D: Unit Circle\n$x^2+y^2=1$")

    # 3D unit sphere
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_wireframe(x, y, z, color="#1f77b4", lw=0.4, alpha=.6)
    ax2.set_title("3D: Unit Sphere\n$x^2+y^2+z^2=1$\n→ ND: Hypersphere")
    fig.tight_layout(); fig.savefig(path, dpi=130); print("saved:", path)


if __name__ == "__main__":
    fig_radial_projection("la1_3_radial.png")
    fig_circle_sphere("la1_3_sphere.png")
