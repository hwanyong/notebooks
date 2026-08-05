"""
[실전선대] 6장 — §6.2 행렬-벡터 곱셈을 통한 기하학적 변환: 순수 회전·직교성·전단 애니메이션

핵심: 좌표 행렬(2×N, 각 열=한 점)에 변환 행렬 T를 왼쪽 곱하면 전 점이 한 번에 변환된다.
     순수 회전 = 직교 행렬(열 정규직교) → TᵀT=I → 길이 보존, det=1.
     ⚠️ 실전선대 T=[[cos,sin],[-sin,cos]] 는 시계방향(CW) = 표준 CCW 회전행렬의 전치.
        원문 대조 완료([그림 6-2] 번역본 스캔 직접 확인): 원본 꼭대기 (0,1)의 상이
        +x 쪽 (0.59, 0.81)로 넘어간다 = 90°→54° = 각이 줄어드는 방향 = CW.
     흔들리는 원 = 회전이 아니라 수평 전단(shear) [[1,1-φ],[0,1]], det=1(넓이 보존).
        원문 대조 완료([그림 6-3] 번역본 스캔 직접 확인): 프레임의 세로 범위는 ±1 그대로이고
        가로만 ±1.22로 벌어진다 = 축 스케일링 없는 순수 수평 전단(전단계수 s≈0.7).
실행: python la6_geometry.py          # assert 검증만 (headless 안전, 그림 생성 없음)
      python la6_geometry.py --fig    # [그림 6-2]·[그림 6-3] 규칙12 재구성 PNG 2장 + 흔들리는 원 GIF
노트: 수학 원자 [2.10 회전·기하 변환 — 순수 회전·직교성·전단] / 렌즈 [28]
"""
import os
import sys

import numpy as np

TOL = 1e-12
HERE = os.path.dirname(os.path.abspath(__file__))

# 볼트 MOC 정리 규칙 12 팔레트 — v=파랑 · w=주황 · 합=초록 · 음수/강조=분홍
BLUE, ORANGE, GREEN, PINK, GREY = "#1c7ed6", "#e8590c", "#2f9e44", "#d6336c", "#adb5bd"
INK, GRID, PANEL = "#495057", "#dee2e6", "#f8f9fa"


def rot_cw(theta):
    """실전선대 §6.2의 순수 회전 행렬 — 시계방향(CW)."""
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def rot_ccw(theta):
    """수학 표준(반시계, CCW) — 각주 2 '사인들의 빼기 기호를 맞바꾼' 결과."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def shear(phi, flip_y=False):
    """흔들리는 원의 변환 행렬. flip_y=True → 각주 3(오른쪽-아래 원소를 −1)."""
    return np.array([[1.0, 1.0 - phi],
                     [0.0, -1.0 if flip_y else 1.0]])


def polygon_area(P):
    """닫힌 폴리곤(2×N, 마지막 열=첫 열)의 신발끈 넓이. 선형변환하면 |det|배로 정확히 스케일."""
    x, y = P
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


# ── ① 순수 회전 = 직교 행렬 (§6.2 본문) ────────────────────────
# 책 주장: 열들은 직교(내적 cosθsinθ − sinθcosθ = 0)하며 단위벡터(cos²+sin²=1).
for th in np.linspace(0, 2 * np.pi, 17):
    T = rot_cw(th)
    assert abs(T[:, 0] @ T[:, 1]) < TOL                    # 열 내적 = 0 (직교)
    assert abs(np.linalg.norm(T[:, 0]) - 1) < TOL          # 각 열 노름 = 1 (단위)
    assert abs(np.linalg.norm(T[:, 1]) - 1) < TOL
    assert np.allclose(T.T @ T, np.eye(2))                 # ⇒ TᵀT = I (직교 행렬)
    assert abs(np.linalg.det(T) - 1) < TOL                 # det = cos²+sin² = 1

# TᵀT=I ⇒ ‖Tv‖=‖v‖ — '시계 침의 길이가 안 변하는' 이유 (노트 §근본이해)
rng = np.random.default_rng(0)
V = rng.standard_normal((2, 500))
assert np.allclose(np.linalg.norm(rot_cw(0.7) @ V, axis=0), np.linalg.norm(V, axis=0))

# ── ② 컨벤션 검산: 실전선대 T는 CW (노트 ⚠️ 표기 주의) ──────────
assert np.allclose(rot_cw(0.0), np.eye(2))                 # θ=0 → T=I (위치 불변)
assert np.allclose(rot_cw(np.pi / 2) @ [1.0, 0.0], [0.0, -1.0])   # (1,0) → (0,−1) = 시계 90°
assert np.allclose(rot_ccw(np.pi / 2) @ [1.0, 0.0], [0.0, 1.0])   # 각주2 부호 맞바꿈 → 반시계

for th in (0.3, 1.1, 2.7):
    assert np.allclose(rot_cw(th), rot_ccw(th).T)          # CW = CCW의 전치
    assert np.allclose(rot_cw(th), rot_ccw(-th))           # = 각 부호 반전
    assert np.allclose(rot_cw(th) @ rot_ccw(th), np.eye(2))   # 전치=역 (직교라서)
assert np.allclose(rot_cw(0.4) @ rot_cw(0.9), rot_cw(1.3))    # 회전 합성 = 각 덧셈

# ── ③ [그림 6-2] 재현: θ=π/5(36°), 수직선 → 기울어진 직선 ───────
theta36 = np.pi / 5
pts_line = np.vstack((np.zeros(21), np.linspace(-1, 1, 21)))   # 2×N, x=0 수직선
rot_line = rot_cw(theta36) @ pts_line

top = rot_line[:, -1]                                      # 원본 (0,1)의 상
assert np.allclose(top, [np.sin(theta36), np.cos(theta36)])
assert np.allclose(top, [0.5877852523, 0.8090169944])      # 그림 6-2 최상단 ≈ (0.59, 0.81)
# 수직선(+x 기준 90°)이 54°로 = 36° 시계 회전
ang = np.degrees(np.arctan2(top[1], top[0]))
assert abs(ang - 54.0) < 1e-9 and abs(90.0 - ang - 36.0) < 1e-9
# 길이·간격 보존: 점들은 여전히 직선 위 등간격, 전체 길이 불변
assert np.allclose(np.linalg.norm(rot_line, axis=0), np.linalg.norm(pts_line, axis=0))
gaps = np.linalg.norm(np.diff(rot_line, axis=1), axis=0)
assert np.allclose(gaps, gaps[0])

# ── ③-b [그림 6-2] 책 스캔 실측과의 교차검증 (번역본 p.146) ──────
# ⚠️ SCAN_TOL 은 TOL(부동소수 오차)과 성격이 다르다 — 인쇄 도판 픽셀 실측의 불확실성이다.
SCAN_TOL = 2e-3
MEASURED_RATIO = 0.7274        # 스캔 실측: 변형 점열의 x/y 중앙값
MEASURED_DEG = 36.03           # 스캔 실측: 수직선 기준 기울기 각

nz = pts_line[1] != 0          # y=0 인 원점은 비(比)가 정의되지 않으므로 제외
ratios = rot_line[0][nz] / rot_line[1][nz]
assert np.allclose(ratios, ratios[0], atol=TOL), "x/y 비가 점마다 다르다 — 직선 위가 아니다"
ratio = float(np.median(ratios))
assert abs(ratio - np.tan(theta36)) < TOL, f"x/y 중앙값이 tanθ가 아니다: {ratio}"
assert abs(ratio - MEASURED_RATIO) < SCAN_TOL, \
    f"스캔 실측 x/y={MEASURED_RATIO}와 어긋남: 계산값 {ratio:.4f}"
assert abs(np.degrees(np.arctan(MEASURED_RATIO)) - MEASURED_DEG) < 5e-3, \
    "실측 비 0.7274가 실측 각 36.03°와 자기모순"
assert abs(np.degrees(np.arctan(ratio)) - 36.0) < 1e-9, "계산 기울기가 정확히 36°가 아니다"
assert abs(MEASURED_DEG - 36.0) < 5e-2, "스캔 실측각이 θ=π/5에서 0.05° 이상 벗어남"

# 꼭대기가 +x 로 '넘어간다' = CW 의 육안 판별근거 (스캔에서 직접 확인한 사실)
upper, lower = pts_line[1] > 0, pts_line[1] < 0
assert np.all(rot_line[0][upper] > 0), "위쪽 절반이 +x로 넘어가지 않았다 — CW가 아니다"
assert np.all(rot_line[0][lower] < 0), "아래쪽 절반이 −x로 가지 않았다 — CW가 아니다"
assert rot_cw(theta36)[0, 1] > 0, "T[0,1]=+sinθ 여야 (0,1)의 x가 양수가 된다"
# 같은 각의 CCW 였다면 꼭대기는 반대쪽(−x)으로 갔어야 한다 — 대조군
assert (rot_ccw(theta36) @ [0.0, 1.0])[0] < 0, "CCW 대조군이 −x로 가지 않음 — 컨벤션 검산 실패"

# ── ④ 흔들리는 원 = 전단(shear), 회전 아님 (§6.2 ②) ─────────────
tt = np.linspace(0, 2 * np.pi, 100)
circle = np.vstack((np.sin(tt), np.cos(tt)))               # 책 순서 (sin, cos)
assert np.allclose(np.linalg.norm(circle, axis=0), 1.0)    # 단위원

S = shear(0.0)                                             # φ=0 → 전단계수 최대 1
assert abs(np.linalg.det(S) - 1) < TOL                     # det=1 → 넓이 보존
assert not np.allclose(S.T @ S, np.eye(2))                 # 직교 아님 ⇒ 회전 아님
assert np.linalg.norm(S @ [0.0, 1.0]) > 1.0                # 길이 안 지킴 (회전과 결정적 차이)
assert np.allclose(S @ [1.0, 0.0], [1.0, 0.0])             # x축은 고정 — 전단의 고유벡터
w, _ = np.linalg.eig(S)
assert np.allclose(w, [1.0, 1.0])                          # 고윳값 1 (중근)
assert np.allclose(shear(1.0), np.eye(2))                  # 📌 φ=1이면 T=I (책 명시)

# 원 → 기울어진 타원: 넓이 π 보존, 반지름은 변함(최대>1, 최소<1)
ell = S @ circle
r = np.linalg.norm(ell, axis=0)
assert r.max() > 1.0 and r.min() < 1.0
assert abs(np.linalg.det(S) * np.pi - np.pi) < TOL

# φ 스케줄: phi = linspace(-1, 1-1/40, 40)**2 → 1→~0→0.95 (끝점을 1에서 떼 프레임 중복 회피)
phi = np.linspace(-1, 1 - 1 / 40, 40) ** 2
assert abs(phi[0] - 1.0) < TOL and phi.min() < 1e-3 and phi[-1] < 1.0
assert np.allclose(shear(phi[0]), np.eye(2))               # 첫 프레임 = 원 (T=I)
sc = 1 - phi                                               # 전단계수: 0 → ~1 → 0.05
assert abs(sc[0]) < TOL and sc.max() > 0.999

# ── ④-b [그림 6-3] 책 스캔 실측: 세로 ±1 불변 · 가로만 √(1+s²) 확장 ──
# 720분할이면 t=π/2·π·3π/2 를 정확히 밟아 sin·cos 의 ±1 이 표본에 그대로 들어온다.
tt_fine = np.linspace(0, 2 * np.pi, 721)
circle_fine = np.vstack((np.sin(tt_fine), np.cos(tt_fine)))
assert abs(circle_fine[1].max() - 1.0) < TOL and abs(circle_fine[1].min() + 1.0) < TOL
AREA_TOL = 1e-10                                           # 신발끈 720항 누적오차 상한
area_circle = polygon_area(circle_fine)
assert abs(area_circle - np.pi) < 1e-4, "720각형 넓이가 π 근처가 아니다"

S_SCAN = 0.7            # 스캔 프레임에서 읽은 전단계수 1-φ
MEASURED_HALF_W = 1.22  # 스캔 실측 가로 반폭 (세로는 ±1 그대로)
FRAME_S = (0.0, 0.5, S_SCAN, 1.0)

for s_ in FRAME_S:
    E = shear(1.0 - s_) @ circle_fine
    # (1) y 는 절대 변하지 않는다 — T의 2행이 [0,1] 이라 구조적으로 보장. 전단의 정의 그 자체.
    assert np.allclose(E[1], circle_fine[1], atol=TOL), f"s={s_}: y가 변했다 — 순수 수평 전단이 아니다"
    assert abs(E[1].max() - 1.0) < TOL and abs(E[1].min() + 1.0) < TOL, \
        f"s={s_}: 세로 범위가 ±1을 벗어남 — 축 스케일링이 섞였다"
    # (2) 가로 반폭은 정확히 √(1+s²). x(t)=sin t + s·cos t 의 최대점에서 해석적으로 확인.
    t_star = np.arctan2(1.0, s_)
    assert abs((np.sin(t_star) + s_ * np.cos(t_star)) - np.hypot(1.0, s_)) < TOL, \
        f"s={s_}: 가로 반폭의 해석해가 √(1+s²)가 아니다"
    assert abs(np.abs(E[0]).max() - np.hypot(1.0, s_)) < 1e-4, \
        f"s={s_}: 표본 최대 |x| 가 √(1+s²)에서 벗어남"
    # (3) det=1 ⇒ 넓이 불변 (기울여도 π)
    assert abs(polygon_area(E) - area_circle) < AREA_TOL, f"s={s_}: 넓이가 변했다 — det=1 위배"
    assert abs(np.linalg.det(shear(1.0 - s_)) - 1.0) < TOL

# 가로는 단조 증가(길이 미보존) — 회전과 갈리는 지점
half_w = np.array([np.hypot(1.0, s_) for s_ in FRAME_S])
assert np.all(np.diff(half_w) > 0), "전단계수가 커지는데 가로가 안 늘었다"
# 스캔 실측값(소수 2자리)과의 대조: 정방향·역방향 둘 다
assert abs(np.hypot(1.0, S_SCAN) - MEASURED_HALF_W) < 1e-2, \
    f"s=0.7의 반폭 {np.hypot(1.0, S_SCAN):.4f}가 스캔 실측 {MEASURED_HALF_W}와 불일치"
assert abs(np.sqrt(MEASURED_HALF_W ** 2 - 1.0) - S_SCAN) < 2e-3, \
    "실측 반폭 1.22에서 되짚은 전단계수가 0.7이 아니다"
# 스캔 프레임 s≈0.7 은 애니메이션 φ 스케줄 안에 실제로 등장하는 값이다
assert sc.min() <= S_SCAN <= sc.max(), "s=0.7이 φ 스케줄의 전단계수 범위 밖"

# ── ⑤ 각주 3: 오른쪽-아래 원소를 −1 → 원이 왼쪽으로 흔들린다 ─────
right, left = shear(0.0) @ circle, shear(0.0, flip_y=True) @ circle
assert right[0, np.argmax(right[1])] > 0                   # 위쪽 끝이 오른쪽으로 밀림
assert left[0, np.argmax(left[1])] < 0                     # y 반전 → 위쪽 끝이 왼쪽으로
assert abs(abs(np.linalg.det(shear(0.0, flip_y=True))) - 1) < TOL   # |det|=1, 방향은 반전

# ── ⑥ 책 인쇄 코드 캐비엇 (p.147) ──────────────────────────────
# 원문: plth, = ax.plot(np.cos(x), np.sin(x), 'ko')  ← 변수 x가 정의된 적 없음(NameError).
# 정의된 건 theta 뿐이고, points는 (sin, cos) 순서 → 아래 fig 경로에서 points로 통일해 정정.


# ── 재구성 도판 (규칙 12: 색·격자·원점축·equal aspect·플롯 내부 라벨은 영어) ──
# ⚠️ outdir 기본값은 HERE(이 파일 옆). cwd 상대경로로 저장하면 다른 폴더에서 실행할 때
#    엉뚱한 곳에 PNG가 떨어진다.

def _fig_rotation(plt, path):
    """[그림 6-2] 재구성 — 순수 회전 CW 36°."""
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.8)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.axvline(0, color=GREY, lw=1.0)

    ax.plot(pts_line[0], pts_line[1], "-", color=BLUE, lw=1.2, alpha=0.35)
    ax.plot(rot_line[0], rot_line[1], "-", color=ORANGE, lw=1.2, alpha=0.35)
    ax.plot(pts_line[0], pts_line[1], "o", color=BLUE, ms=7,
            label="before:  P  (21 points on the line x = 0)")
    ax.plot(rot_line[0], rot_line[1], "o", color=ORANGE, ms=7,
            label=r"after:  $T\,P$  (rotated CW by $36^\circ$)")

    # 회전각 호: 90° → 54° (각이 줄어드는 방향 = 시계), 끝에 화살촉으로 방향 확정
    r_arc = 0.45
    a = np.linspace(np.pi / 2, np.pi / 2 - theta36, 80)
    ax.plot(r_arc * np.cos(a), r_arc * np.sin(a), color=GREEN, lw=2.4)
    ax.annotate("", xy=(r_arc * np.cos(a[-1]), r_arc * np.sin(a[-1])),
                xytext=(r_arc * np.cos(a[-6]), r_arc * np.sin(a[-6])),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4, shrinkA=0, shrinkB=0))
    mid = np.pi / 2 - theta36 / 2
    # 지시선에는 화살촉을 달지 않는다 — 호의 화살촉(회전 방향)과 반대로 보여 방향이 모호해진다.
    ax.annotate(r"$\theta = \pi/5 = 36^\circ$" "\n" "CLOCKWISE (CW)" "\n" r"$90^\circ \to 54^\circ$",
                xy=(r_arc * np.cos(mid), r_arc * np.sin(mid)), xytext=(0.62, 0.52),
                color=GREEN, fontsize=10.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))

    # 꼭대기가 +x 로 넘어가는 것이 CW 의 육안 판별근거
    ax.annotate(r"top:  $[0,1] \mapsto [0.5878,\ 0.8090]$" "\n"
                r"it crosses to $+x$   $\Rightarrow$   CW",
                xy=tuple(top), xytext=(1.26, 1.14), fontsize=9.5, color=INK,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color="#868e96", lw=1.0,
                                connectionstyle="arc3,rad=0.25"))

    ax.text(0.015, 0.985,
            "each COLUMN of P is one point (P is 2×N)\n"
            r"$T\,P$ transforms all 21 points at once" "\n"
            r"$T^{\mathsf{T}}T = I$,  $\det T = 1$  $\Rightarrow$  lengths & gaps preserved",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec="#ced4da"))

    # 검증 메타정보는 축 밖으로 — 플롯 내부는 도형만 남긴다.
    # ⚠️ 축 밖 텍스트를 쓸 때 fig.tight_layout() 을 부르면 축이 눌려 찌그러진다. savefig의
    #    bbox_inches="tight" 만으로 충분하다.
    ax.text(0.5, -0.125,
            "book-scan cross-check (translated ed. p.146)\n"
            r"median $x/y$ of the transformed points $= 0.7274 \to 36.03^\circ$"
            r"   vs   $\tan 36^\circ = 0.7265$   ($0.03^\circ$ print gap)",
            transform=ax.transAxes, va="top", ha="center", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff9db", ec="#f0c33c"))

    ax.set(xlim=(-1.3, 1.3), ylim=(-1.35, 1.3), aspect="equal", xlabel="x", ylabel="y")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.set_title("[Fig 6-2] reconstruction — pure rotation, "
                 r"$\theta = \pi/5 = 36^\circ$" "\n"
                 r"$T = [[\cos\theta,\ \sin\theta],\ [-\sin\theta,\ \cos\theta]]$" "\n"
                 "textbook convention = CLOCKWISE = transpose of the standard CCW matrix",
                 fontsize=11.5, pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_shear(plt, path):
    """[그림 6-3] 재구성 — 흔들리는 원 = det=1 수평 전단, 프레임 겹쳐보기."""
    frames = [(1.0, BLUE, "-", 0.10, "unit circle"),
              (0.5, GREEN, "-", 0.10, ""),
              (0.3, PINK, "--", 0.00, "book-scan frame"),
              (0.0, ORANGE, "-", 0.10, "max shear")]

    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.8)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.axvline(0, color=GREY, lw=1.0)

    for y_ in (1.0, -1.0):                                 # 세로 ±1 불변선
        ax.axhline(y_, color="#868e96", ls=":", lw=1.1)
    for x_ in (np.hypot(1.0, S_SCAN), -np.hypot(1.0, S_SCAN)):
        ax.axvline(x_, color=PINK, ls=":", lw=1.1)

    for ph, col, ls, al, note in frames:
        s_ = 1.0 - ph
        E = shear(ph) @ circle_fine
        if al:
            ax.fill(E[0], E[1], color=col, alpha=al, lw=0)
        lab = (rf"$\phi={ph:g}$,  $s = 1-\phi = {s_:g}$"
               + (f"  ({note})" if note else "")
               + rf"   —   area $= {polygon_area(E):.4f}$,  "
                 rf"half-width $=\sqrt{{1+s^2}}= {np.hypot(1.0, s_):.4f}$,  "
                 rf"$\det T = {np.linalg.det(shear(ph)):.1f}$")
        ax.plot(E[0], E[1], ls=ls, color=col, lw=2.0, label=lab)
        ax.plot([s_], [1.0], "o", color=col, ms=6, mec="white", mew=1.0, zorder=6)
        ax.plot([np.hypot(1.0, s_)], [s_ / np.hypot(1.0, s_)], "s",
                color=col, ms=5, mec="white", mew=1.0, zorder=6)

    ax.annotate("", xy=(-np.hypot(1.0, S_SCAN), -1.34), xytext=(np.hypot(1.0, S_SCAN), -1.34),
                arrowprops=dict(arrowstyle="<->", color=PINK, lw=1.4))
    ax.text(0.0, -1.44, r"book scan: half-width $\approx 1.22$ at $s \approx 0.7$   "
                        r"($\sqrt{1+0.7^2} = 1.2207$)",
            ha="center", va="top", fontsize=9, color=PINK)
    ax.text(2.55, 1.10,
            r"$\bullet$  top of a frame $\to$ always $y = +1$" "\n"
            r"$\blacksquare$  widest point $\to x = \sqrt{1+s^2}$, grows",
            ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.text(-2.52, -1.06, r"$y = -1$ likewise fixed  —  no vertical motion at all",
            ha="left", va="top", fontsize=9, color=INK)

    ax.text(0.012, 0.985,
            r"$T = [[1,\ 1-\phi],\ [0,\ 1]]$   (horizontal shear, not a rotation)" "\n"
            r"$x' = x + s\,y$,   $y' = y$    — row 2 of $T$ is $[0,\ 1]$, so $y$ is untouched" "\n"
            r"$\phi:\ 1 \to 0 \to 1$ animation  $\Leftrightarrow$  shear coef "
            r"$s = 1-\phi:\ 0 \to 1 \to 0$" "\n"
            r"$T^{\mathsf{T}}T \neq I$  $\Rightarrow$  lengths NOT preserved, "
            r"but $\det T = 1$  $\Rightarrow$  area is",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec="#ced4da"))

    ax.set(xlim=(-2.6, 2.6), ylim=(-1.75, 1.6), aspect="equal", xlabel="x", ylabel="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=8.5,
              handlelength=2.6, borderaxespad=0.0)
    ax.set_title("[Fig 6-3] reconstruction — the \"wobbly circle\" is a horizontal SHEAR, "
                 "not a rotation\n"
                 r"$\det T = 1$  $\Rightarrow$  area $= \pi \approx 3.1416$ in every frame;  "
                 r"$|y| \leq 1$ frozen while the half-width $\sqrt{1+s^2}$ grows",
                 fontsize=11.5, pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render(outdir=HERE):
    """--fig: [그림 6-2]·[그림 6-3] 재구성 PNG 2장 + '흔들리는 원' 애니메이션 GIF."""
    import matplotlib
    matplotlib.use("Agg")                                  # headless — GUI 창을 띄우지 않는다
    import matplotlib.pyplot as plt
    from matplotlib import animation

    names = ["la6_2_pure_rotation_cw36.png", "la6_3_wobbly_circle_shear.png"]
    # 재구성 PNG는 영어 라벨(규칙 12)이라 DejaVu Sans — AppleGothic은 '['를 전각으로 그린다.
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.unicode_minus": True}):
        _fig_rotation(plt, os.path.join(outdir, names[0]))
        _fig_shear(plt, os.path.join(outdir, names[1]))

    # '흔들리는 원' GIF — 이미 볼트 임베드본과 동일한 산출물이라 렌더 설정을 바꾸지 않는다.
    # 핸들만 갱신(장면 재생성 없음).
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 6))
    plth, = ax.plot(circle[0], circle[1], "ko")            # ⚠️ 책의 x 미정의 정정
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal")

    def aframe(ph):
        P = shear(ph) @ circle
        plth.set_xdata(P[0, :])
        plth.set_ydata(P[1, :])
        return plth,

    anim = animation.FuncAnimation(fig, aframe, phi, interval=100, repeat=True)
    anim.save(os.path.join(outdir, "la6_wobbly_circle.gif"), writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    return names + ["la6_wobbly_circle.gif"]


if __name__ == "__main__":
    print("✅ §6.2 검증 통과 — 직교성(TᵀT=I·det=1·노름보존) · CW=CCW전치 · "
          "그림6-2 재현(90°→54°, 스캔 x/y=0.7274 대조) · "
          "전단(det=1·회전아님·고유벡터[1,0]·φ=1→I·세로±1 불변·가로√(1+s²), 스캔 1.22 대조) · "
          "각주3 좌측 흔들림")
    if "--fig" in sys.argv:
        print("🖼  저장:", ", ".join(render()))
