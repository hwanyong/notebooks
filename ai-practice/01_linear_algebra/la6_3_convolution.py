"""
[실전선대] 6장 — §6.3 이미지 특징 탐지: 2차원 합성곱(커널 슬라이딩 '내적')·가우스 평활

핵심: 커널과 겹친 창 사이의 '내적' = 아다마르곱 후 전 원소 합(p.149 원문: "아다마르곱을 한
     다음 모든 행렬 원소에 대한 합"). 커널을 뒤집지 않으므로 엄밀히는 상관(correlation)이다 —
     원문 대조 완료(p.151 코드 `dotprod = np.sum(pieceOfImg*kernel)` 에 뒤집기 없음).
     커널을 뒤집으면 ±행이 위아래로 뒤바뀌어 [그림 6-4](A) 인쇄값과 어긋난다(assert 로 고정).
     ⚠️ 패딩 모드 판정(이 파일의 핵심 발견): [그림 6-4](A) 인쇄 출력 25칸을 전부 재현하는 것은
        **가장자리 복제(edge/replicate) 패딩**뿐이다. 제로 패딩이면 **첫 행만** 0 → −1 로
        갈리고 나머지 20칸은 완전히 같다. 책 본문(p.149)은 "이미지를 패딩"이라고만 하고
        p.151 코드에도 imagePad 생성 줄이 없어 **어느 쪽인지 원문으로는 미판정**.
     ⚠️ p.151 `range(halfKr, imgN-halfKr)` 의 imgN 은 **패딩된 배열**의 한 변이어야 출력이
        원본 크기가 된다(원본 크기로 읽으면 N−2칸만 채워짐).
     ⓘ 시연 창은 책의 창이 아니다: [그림 6-4](A)가 강조한 창은 좌상단 IMAGE_A[0:3,0:3](값 0),
        이 스크립트가 전개해 보이는 창은 안쪽 IMAGE_A[1:4,1:4](값 −2)다. 값이 0이 아니라 산술이
        드러나고 패딩 모드와 무관해 위 미판정 쟁점에 오염되지 않기 때문 — 책 창의 값 0도 assert 로 고정.
실행: python la6_3_convolution.py          # assert 검증만 (headless 안전, 그림 생성 없음)
      python la6_3_convolution.py --fig    # [그림 6-4] (A)·(B) 규칙12 재구성 PNG 2장
노트: (미작성 — 이 코드가 재료)  ※ 1단계 허브·[08 벡터 응용]의 미결 항목
      "이미지 특징 탐지·2D 필터링 파이프라인 ([실전선대] 6.3)" 이 대상. 렌즈 [28]의 "다음" 자리.
"""
import os
import sys

import numpy as np

TOL = 1e-12
HERE = os.path.dirname(os.path.abspath(__file__))

# 볼트 MOC 정리 규칙 12 팔레트 — 이미지=파랑 · 커널=주황 · 결과(합)=초록 · 음수=분홍
BLUE, ORANGE, GREEN, PINK, GREY = "#1c7ed6", "#e8590c", "#2f9e44", "#d6336c", "#adb5bd"
INK, GRID, PANEL = "#495057", "#dee2e6", "#f8f9fa"

MINUS = "−"          # U+2212 진짜 마이너스 (DejaVu Sans 보유 확인)
CIRC_AST = "∗"       # U+2217 asterisk operator — 책의 ⊛ 안쪽 기호


# ── 합성곱 코어 ────────────────────────────────────────────────

def hadamard_sum(window, kernel):
    """책 p.149 가 따옴표로 '내적'이라 부른 것 = 아다마르곱 후 전 원소 합.

    벡터 내적과 달리 두 **행렬** 사이 연산이라 따옴표가 붙었다. 커널을 뒤집지 않는다.
    """
    assert window.shape == kernel.shape, f"창·커널 shape 불일치: {window.shape} vs {kernel.shape}"
    return float(np.sum(window * kernel))


def conv2d_same(image, kernel, pad_mode="edge"):
    """'same' 크기 2D 합성곱(정확히는 상관). 출력 shape == 입력 shape.

    pad_mode: "edge"(가장자리 복제) | "constant"(제로 패딩)
    """
    kr, kc = kernel.shape
    assert kr % 2 == 1 and kc % 2 == 1, f"커널은 홀수 크기여야 중심이 정의된다: {kernel.shape}"
    hr, hc = kr // 2, kc // 2
    if pad_mode == "constant":
        padded = np.pad(image, ((hr, hr), (hc, hc)), mode="constant", constant_values=0)
    else:
        padded = np.pad(image, ((hr, hr), (hc, hc)), mode=pad_mode)

    out = np.zeros(image.shape, dtype=float)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            out[r, c] = hadamard_sum(padded[r:r + kr, c:c + kc], kernel)
    return out


def book_loop_conv2d(image, kernel, pad_mode="edge"):
    """p.151 이중 for 문 그대로의 전사(轉寫) — 교차 구현 검증용.

    책 코드는 **패딩된 좌표**로 순회하고 결과도 패딩 크기 배열에 쓴다. 그래서 [그림 6-4](A)의
    결과 격자가 7×7(테두리 한 칸 빈칸)로 그려져 있다. 여기서도 그대로 재현한 뒤 안쪽만 잘라
    conv2d_same 과 대조한다.
    """
    halfKr = kernel.shape[0] // 2
    if pad_mode == "constant":
        imagePad = np.pad(image, halfKr, mode="constant", constant_values=0)
    else:
        imagePad = np.pad(image, halfKr, mode=pad_mode)
    imgN = imagePad.shape[0]                       # ⚠️ 원본 크기가 아니라 '패딩된' 크기
    convoutput = np.zeros_like(imagePad, dtype=float)

    for rowi in range(halfKr, imgN - halfKr):      # 행에 대한 for 루프
        for coli in range(halfKr, imgN - halfKr):  # 열에 대한 for 루프
            # 이미지 조각 자르기
            pieceOfImg = imagePad[rowi - halfKr:rowi + halfKr + 1:1,
                                  coli - halfKr:coli + halfKr + 1:1]
            # 내적: 아다마르곱과 합
            dotprod = np.sum(pieceOfImg * kernel)
            # 이 픽셀에 대한 결과를 저장
            convoutput[rowi, coli] = dotprod
    return convoutput


def gaussian_kernel_2d(n=21, span=3.0, sigma=20.0):
    """p.151 코드 그대로: G = exp(−(X²+Y²)/σ) 후 합=1 정규화.

    책 원문 표기: Y,X = np.meshgrid(linspace(-3,3,21), linspace(-3,3,21))
    두 축 격자가 동일하고 X²+Y² 가 방사대칭이라 Y,X 순서 뒤집힘은 값에 영향이 없다(assert 로 확인).
    """
    axis = np.linspace(-span, span, n)
    Y, X = np.meshgrid(axis, axis)
    kernel = np.exp(-(X ** 2 + Y ** 2) / sigma)
    return kernel / np.sum(kernel)                 # 정규화


# ── ① [그림 6-4](A) 이산 예제 — 스캔에서 읽은 값을 하드코딩해 검증 ──────

IMAGE_A = np.array([[0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 1, 1, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1]], dtype=float)

KERNEL_A = np.array([[1, 1, 1],                    # 수평 에지 탐지기
                     [-1, -1, -1],
                     [0, 0, 0]], dtype=float)

# 번역본 p.150 [그림 6-4](A) 결과 격자에 인쇄된 25개 값 (계산 결과가 아니라 스캔 판독값)
BOOK_A_PRINTED = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, -1, -2, -2, -1],
                           [0, 1, 2, 2, 1],
                           [0, 0, 0, 0, 0]], dtype=float)

out_edge = conv2d_same(IMAGE_A, KERNEL_A, pad_mode="edge")
out_zero = conv2d_same(IMAGE_A, KERNEL_A, pad_mode="constant")
out_flip = conv2d_same(IMAGE_A, KERNEL_A[::-1, ::-1], pad_mode="edge")   # 진짜 합성곱(커널 뒤집기)

# 크기: 패딩 덕에 출력이 입력과 같다 (p.149 "결과가 동일한 크기가 되도록")
assert out_edge.shape == IMAGE_A.shape, f"'same' 크기 깨짐: {out_edge.shape}"

# 판정 1 — 뒤집지 않은 상관 + 가장자리 복제 패딩이 인쇄값 25칸을 전부 재현한다
assert np.allclose(out_edge, BOOK_A_PRINTED, atol=TOL), \
    f"edge 패딩 상관이 인쇄값과 불일치:\n{out_edge}"

# 판정 2 — 제로 패딩은 첫 행에서만 갈린다(−1). 나머지 20칸은 동일.
diff = np.argwhere(np.abs(out_zero - BOOK_A_PRINTED) > TOL)
assert set(map(tuple, diff)) == {(0, c) for c in range(5)}, \
    f"제로 패딩 차이가 첫 행이 아닌 곳에도 있다: {diff.tolist()}"
assert np.allclose(out_zero[0], -1.0, atol=TOL), f"제로 패딩 첫 행 기대 −1: {out_zero[0]}"
assert np.allclose(out_zero[1:], BOOK_A_PRINTED[1:], atol=TOL), "제로 패딩 2행 이하가 인쇄값과 불일치"

# 판정 3 — 커널을 뒤집으면(=엄밀한 합성곱) 인쇄값과 어긋난다. ±행이 위로 한 칸 올라간다.
assert not np.allclose(out_flip, BOOK_A_PRINTED, atol=TOL), \
    "커널을 뒤집어도 같다면 판별 실험이 무의미하다(대칭 커널이면 그럴 수 있음)"
assert np.allclose(out_flip[1], [0, 1, 2, 2, 1], atol=TOL), \
    f"뒤집힌 커널의 +행 위치 기대 어긋남: {out_flip[1]}"
assert np.allclose(out_flip[2], [0, -1, -2, -2, -1], atol=TOL), \
    f"뒤집힌 커널의 −행 위치 기대 어긋남: {out_flip[2]}"

# 판정 4 — 교차 구현: 책의 이중 for 문 전사본과 벡터화되지 않은 same 버전이 일치
book_pad_out = book_loop_conv2d(IMAGE_A, KERNEL_A, pad_mode="edge")
assert book_pad_out.shape == (7, 7), f"책 코드는 패딩 크기 배열에 쓴다: {book_pad_out.shape}"
assert np.allclose(book_pad_out[1:6, 1:6], out_edge, atol=TOL), "책 이중 for 문 전사본과 불일치"
assert np.allclose(book_pad_out[0], 0.0, atol=TOL), "패딩 테두리는 손대지 않아 0으로 남아야 한다"

# 판정 5 — 커널 합 = 0 ⇒ 평탄 영역은 0 (에지 탐지기의 정의)
assert abs(KERNEL_A.sum()) < TOL, f"에지 탐지 커널의 합은 0이어야 한다: {KERNEL_A.sum()}"
flat = np.full((5, 5), 7.0)
assert np.allclose(conv2d_same(flat, KERNEL_A, "edge"), 0.0, atol=TOL), "평탄 이미지 응답이 0이 아니다"

# 판정 6 — 이 스크립트가 시연용으로 고른 안쪽 창의 산술을 손으로 검산한 값과 대조.
#   ⚠️ 책이 강조한 창이 아니다. [그림 6-4](A)의 강조 창은 좌상단 [0:3,0:3](→ 판정 7, 값 0)이고,
#      여기 안쪽 창을 쓰는 건 이 파일의 선택이다 — 값이 0이 아니라 산술이 눈에 보이고,
#      이미지 안쪽이라 edge/zero 패딩 미판정 쟁점과 무관하기 때문.
WIN_R0, WIN_C0 = 1, 1                                   # 시연 창 = IMAGE_A[1:4, 1:4], 중심 = (2,2)
window_A = IMAGE_A[WIN_R0:WIN_R0 + 3, WIN_C0:WIN_C0 + 3]
hadamard_A = window_A * KERNEL_A
assert np.allclose(window_A, [[1, 0, 0], [1, 1, 1], [1, 0, 0]], atol=TOL), f"창 판독 오류:\n{window_A}"
assert np.allclose(hadamard_A, [[1, 0, 0], [-1, -1, -1], [0, 0, 0]], atol=TOL), \
    f"아다마르곱 손검산 불일치:\n{hadamard_A}"
assert abs(hadamard_sum(window_A, KERNEL_A) - (-2.0)) < TOL, "시연 창(안쪽)의 '내적'은 −2 여야 한다"
assert abs(out_edge[2, 2] - (-2.0)) < TOL, f"출력 (2,2) 기대 −2: {out_edge[2, 2]}"
assert abs(out_zero[2, 2] - (-2.0)) < TOL, "안쪽 창이라 패딩 모드와 무관해야 한다"

# 판정 7 — 책이 [그림 6-4](A)에서 실제로 강조한 창은 좌상단 IMAGE_A[0:3,0:3] 이고 그 값은 0이다.
#   그 창이 만드는 출력 칸 out[1,1] 도 인쇄값 0과 일치한다.
assert abs(hadamard_sum(IMAGE_A[0:3, 0:3], KERNEL_A) - 0.0) < TOL, "책이 강조한 창(좌상단)의 값은 0"
assert abs(out_edge[1, 1] - 0.0) < TOL, f"책 강조 창의 출력 칸 out[1,1] 기대 0: {out_edge[1, 1]}"

# 판정 8 — 부호 의미. ⚠️ 직관이 틀리기 쉬운 지점이라 인덱스로 못 박는다.
#   out[i,j] = (i−1행 3칸 합) − (i행 3칸 합). 커널 +1행이 '한 칸 위'에, −1행이 '중심 행'에 걸린다.
#   ⇒ 밝은 가로줄(=이미지 2행, 'H'의 가로대)이 있는 **바로 그 행**이 음수(−2)이고,
#     그 **한 칸 아래 행**이 양수(+2)다. "위가 음수/아래가 양수"가 아니다.
BAR_ROW = int(np.argmax(IMAGE_A.sum(axis=1)))
assert BAR_ROW == 2, f"밝은 가로줄은 이미지 2행: {BAR_ROW}"
assert int(np.argmin(out_edge.sum(axis=1))) == BAR_ROW, \
    f"음수 응답은 가로줄과 같은 행이어야 한다: {out_edge.sum(axis=1)}"
assert int(np.argmax(out_edge.sum(axis=1))) == BAR_ROW + 1, \
    f"양수 응답은 가로줄 바로 아래 행이어야 한다: {out_edge.sum(axis=1)}"
assert np.allclose(out_edge[BAR_ROW], [0, -1, -2, -2, -1], atol=TOL), \
    f"가로줄 행의 응답 기대 [0,−1,−2,−2,−1]: {out_edge[BAR_ROW]}"
assert np.allclose(out_edge[BAR_ROW + 1], [0, 1, 2, 2, 1], atol=TOL), \
    f"가로줄 바로 아래 행의 응답 기대 [0,1,2,2,1]: {out_edge[BAR_ROW + 1]}"


# ── ② [그림 6-4](B) 가우스 평활 ────────────────────────────────

KN, KSPAN, KSIGMA = 21, 3.0, 20.0
KERNEL_B = gaussian_kernel_2d(KN, KSPAN, KSIGMA)

assert KERNEL_B.shape == (KN, KN), f"커널 shape: {KERNEL_B.shape}"
assert abs(KERNEL_B.sum() - 1.0) < TOL, f"정규화 후 합=1 이어야 한다: {KERNEL_B.sum()}"

# meshgrid 인자 순서(Y,X)가 값에 영향이 없음을 직접 확인 — 방사대칭이라 무해
_ax = np.linspace(-KSPAN, KSPAN, KN)
_XY, _YX = np.meshgrid(_ax, _ax), np.meshgrid(_ax, _ax)[::-1]
_k1 = np.exp(-(_XY[0] ** 2 + _XY[1] ** 2) / KSIGMA)
_k2 = np.exp(-(_YX[0] ** 2 + _YX[1] ** 2) / KSIGMA)
assert np.allclose(_k1, _k2, atol=TOL), "Y,X 순서가 값을 바꾼다면 방사대칭 가정이 틀린 것"

# 대칭·중심 최대
assert np.allclose(KERNEL_B, KERNEL_B.T, atol=TOL), "가우스 커널은 전치 대칭이어야 한다"
assert np.allclose(KERNEL_B, KERNEL_B[::-1, :], atol=TOL), "상하 대칭이어야 한다"
assert np.allclose(KERNEL_B, KERNEL_B[:, ::-1], atol=TOL), "좌우 대칭이어야 한다"
assert np.unravel_index(np.argmax(KERNEL_B), KERNEL_B.shape) == (KN // 2, KN // 2), "최댓값은 중심"

# ⚠️ 이 커널은 σ=20 에 격자가 ±3 뿐이라 "1σ도 못 미치게" 잘린 거의 평평한 원반이다.
#    모서리/중심 비 = exp(−(3²+3²)/20) = exp(−0.9) ≈ 0.4066. 그림에서 좁아 보이는 건
#    imshow 의 자동 대비 스케일링(min→검정) 때문이지 실제 감쇠가 커서가 아니다.
assert abs(KERNEL_B.min() / KERNEL_B.max() - np.exp(-0.9)) < 1e-12, \
    f"모서리/중심 비 기대 exp(−0.9): {KERNEL_B.min() / KERNEL_B.max()}"

RNG_SEED, IMG_N = 6, 40
rng = np.random.default_rng(RNG_SEED)               # 고정 시드 — 그림 재현성
IMAGE_B = rng.standard_normal((IMG_N, IMG_N))
SMOOTH_B = conv2d_same(IMAGE_B, KERNEL_B, pad_mode="edge")

assert SMOOTH_B.shape == IMAGE_B.shape, f"'same' 크기 깨짐: {SMOOTH_B.shape}"
# 합=1 커널 ⇒ 상수 이미지는 그대로 통과(가중 평균의 단위원 성질)
assert np.allclose(conv2d_same(np.ones((12, 12)), KERNEL_B, "edge"), 1.0, atol=1e-12), \
    "합=1 커널인데 상수 이미지가 보존되지 않는다"
# 평활 = 분산 축소. 눈에 띄게 줄어야 한다.
VAR_IN, VAR_OUT = IMAGE_B.var(), SMOOTH_B.var()
assert VAR_OUT < VAR_IN / 10, f"평활 후 분산이 충분히 줄지 않았다: {VAR_IN:.4f} → {VAR_OUT:.4f}"
# 평균은 대체로 보존(가중 평균이므로)
assert abs(SMOOTH_B.mean() - IMAGE_B.mean()) < 0.25, \
    f"평균이 크게 이동: {IMAGE_B.mean():.4f} → {SMOOTH_B.mean():.4f}"


# ── 재구성 도판 (규칙 12: 색·격자·재현 스크립트 보존·플롯 내부 라벨은 영어) ──
# ⚠️ outdir 기본값은 HERE(이 파일 옆). cwd 상대경로로 저장하면 볼트에서 실행할 때
#    엉뚱한 곳(볼트 루트)에 PNG 가 떨어진다.

FIG_A = "LA6-4a image-convolution-mechanism (figure).png"
FIG_B = "LA6-4b gaussian-smoothing-random-matrix (figure).png"


def _fmt(v):
    """−1 을 U+2212 로. DejaVu Sans 는 U+2212 를 갖고 AppleGothic 은 갖지 않는다(실측)."""
    s = f"{v:g}"
    return s.replace("-", MINUS)


def _draw_grid(ax, plt, vals, x0, y0, cell, face_fn, fontsize, lw=1.1, ec="#495057"):
    """(r,c) 셀이 [x0+c·cell, y0−(r+1)·cell] ~ [x0+(c+1)·cell, y0−r·cell] 를 차지하는 격자."""
    from matplotlib.patches import Rectangle
    nr, nc = vals.shape
    for r in range(nr):
        for c in range(nc):
            fc, tc, weight = face_fn(r, c, vals[r, c])
            ax.add_patch(Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell,
                                   facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
            ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, _fmt(vals[r, c]),
                    ha="center", va="center", fontsize=fontsize, color=tc,
                    fontweight=weight, zorder=3)


def _op_circle(ax, xy, symbol, radius=0.52, fontsize=17):
    from matplotlib.patches import Circle
    ax.add_patch(Circle(xy, radius, facecolor="white", edgecolor=INK, lw=1.6, zorder=4))
    ax.text(xy[0], xy[1], symbol, ha="center", va="center", fontsize=fontsize,
            color=INK, fontweight="bold", zorder=5)


def _fig_mechanism(plt, path):
    """[그림 6-4](A) 재구성 — 이산 5×5 예제로 본 합성곱 메커니즘."""
    from matplotlib.patches import Rectangle

    CELL, KCELL, SCELL = 1.0, 1.35, 0.62
    IX0, IY0 = 0.0, 0.0                       # 이미지 격자 좌상단
    KX0 = 8.8                                 # 커널 격자 좌단
    KY0 = -2.5 + 1.5 * KCELL                  # 커널 격자 상단(세로 중심 −2.5 에 정렬)
    RX0, RY0 = 15.8, 0.0                      # 결과 격자 좌상단
    OP_CONV, OP_SUM = (7.3, -2.5), (14.3, -2.5)
    HR, HC = WIN_R0, WIN_C0                   # 시연 창(책이 아니라 이 파일의 선택) 좌상단 (1,1)
    OR_, OC_ = HR + 1, HC + 1                 # 그 창이 만드는 출력 칸 (2,2)

    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    ax.set(xlim=(-1.75, 21.6), ylim=(-10.30, 1.55), aspect="equal")
    ax.axis("off")

    # 패딩 링(7×7) — 출력 크기를 입력과 같게 만드는 장치. 이미지 쪽에 그린다.
    ax.add_patch(Rectangle((IX0 - CELL, IY0 - 6 * CELL), 7 * CELL, 7 * CELL,
                           facecolor="none", edgecolor=GREY, lw=1.3, ls=(0, (5, 4)), zorder=1))
    ax.text(IX0 + 2.5, IY0 - 6 * CELL - 0.32, "zero / edge padding ring  (keeps the output 5×5)",
            ha="center", va="top", fontsize=9, color="#868e96")

    # ⚠️ 규칙 12(책 원본 1:1 대조)를 위해 반드시 남긴다 — 이 재구성이 박스로 친 창은 책의 창이
    #    아니다. 이 한 줄이 없으면 노트 본문(책 창 = 값 0)과 그림(안쪽 창 = 값 −2)이 말없이 갈린다.
    ax.text(IX0 - CELL, IY0 - 6 * CELL - 0.62,
            f"the book boxes the top-left window [0:3,0:3] (sum = 0); this reconstruction boxes "
            f"the inner [1:4,1:4] (sum = {MINUS}2) instead: non-zero, and independent of the padding mode",
            ha="left", va="top", fontsize=9, color="#868e96")

    # ── 이미지 ──
    def img_face(r, c, v):
        inside = HR <= r < HR + 3 and HC <= c < HC + 3
        if inside:
            return ("#a5d8ff" if v else "#e7f5ff"), "#0b4f8a", "bold"
        return ("#dbeafe" if v else "white"), INK, "normal"

    _draw_grid(ax, plt, IMAGE_A, IX0, IY0, CELL, img_face, 15)
    ax.add_patch(Rectangle((IX0 + HC * CELL, IY0 - (HR + 3) * CELL), 3 * CELL, 3 * CELL,
                           facecolor="none", edgecolor=BLUE, lw=2.6, zorder=6))
    ax.text(IX0 + 2.5, IY0 + 0.14, 'image  (5×5)   —  an "H": two vertical lines + one bright bar',
            ha="center", va="bottom", fontsize=12, color=BLUE, fontweight="bold")

    # ── 커널 ──
    def ker_face(r, c, v):
        if v > 0:
            return "#ffd8a8", "#8a3d00", "bold"
        if v < 0:
            return "#ffdeeb", "#8a1140", "bold"
        return "#f1f3f5", "#868e96", "normal"

    _draw_grid(ax, plt, KERNEL_A, KX0, KY0, KCELL, ker_face, 18, lw=1.3)
    ax.text(KX0 + 1.5 * KCELL, KY0 + 0.42, "kernel  (3×3)", ha="center", va="bottom",
            fontsize=12.5, color=ORANGE, fontweight="bold")
    ax.text(KX0 + 1.5 * KCELL, KY0 - 3 * KCELL - 0.30,
            "horizontal edge detector\n"
            "weights sum to 0  →  flat areas give 0",
            ha="center", va="top", fontsize=9.5, color=ORANGE)

    # 시연 창 → 커널 확대 지시선 (콜아웃 어법만 책과 같고, 창 위치는 다르다)
    for (px, py), (qx, qy) in [((IX0 + HC, IY0 - HR), (KX0, KY0)),
                               ((IX0 + HC, IY0 - HR - 3), (KX0, KY0 - 3 * KCELL))]:
        ax.plot([px, qx], [py, qy], ls=(0, (4, 4)), color="#adb5bd", lw=1.1, zorder=7)

    # 연산자
    _op_circle(ax, OP_CONV, CIRC_AST, fontsize=21)
    _op_circle(ax, OP_SUM, "Σ")
    for sy in (KY0, KY0 - 3 * KCELL):
        ax.plot([KX0 + 3 * KCELL, OP_SUM[0]], [sy, OP_SUM[1]],
                ls=(0, (4, 4)), color="#adb5bd", lw=1.1, zorder=1)

    # ── 결과 ──
    def res_face(r, c, v):
        if r == OR_ and c == OC_:
            return "#b2f2bb", "#0b4a1e", "bold"
        if v > 0:
            return "#d3f9d8", "#1c6b32", "bold"
        if v < 0:
            return "#ffdeeb", "#8a1140", "bold"
        return "white", INK, "normal"

    _draw_grid(ax, plt, out_edge, RX0, RY0, CELL, res_face, 15)
    ax.add_patch(Rectangle((RX0 + OC_ * CELL, RY0 - (OR_ + 1) * CELL), CELL, CELL,
                           facecolor="none", edgecolor=GREEN, lw=2.6, zorder=6))
    ax.text(RX0 + 2.5, RY0 + 0.42, "result  (5×5, same size as the input)",
            ha="center", va="bottom", fontsize=12.5, color=GREEN, fontweight="bold")

    # 첫 행 = 패딩 모드가 갈리는 유일한 곳
    ax.add_patch(Rectangle((RX0, RY0 - CELL), 5 * CELL, CELL, facecolor="none",
                           edgecolor="#f0a500", lw=1.8, ls=(0, (3, 3)), zorder=6))

    # Σ → 시연 창이 만드는 출력 칸. 화살촉은 칸의 **좌상단 모서리**에 멈춘다 — 옆 칸 숫자를 덮지 않게.
    # 살짝 위로 휘어 지나간다(책과 같은 파선 지시선 어법).
    ax.annotate("", xy=(RX0 + OC_ * CELL + 0.03, RY0 - OR_ * CELL - 0.03),
                xytext=(OP_SUM[0] + 0.6, OP_SUM[1]),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, ls="dashed",
                                shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=-0.24"),
                zorder=8)
    ax.text(RX0 + 2.5 * CELL, RY0 - 5 * CELL - 0.34,
            "one window  →  exactly one output cell", ha="center", va="top",
            fontsize=10, color=GREEN, fontweight="bold")

    # ── 아래: 시연 창의 산술 전개 ──
    ay0 = -7.60          # 위에 규칙 12 대조 문구 한 줄이 들어가 0.55 내렸다(캐비엇 상자도 함께 이동)

    def plain(_r, _c, v):
        if v > 0:
            return "#d3f9d8", "#1c6b32", "bold"
        if v < 0:
            return "#ffdeeb", "#8a1140", "bold"
        return "white", INK, "normal"

    def win_face(_r, _c, v):
        return ("#a5d8ff" if v else "#e7f5ff"), "#0b4f8a", "bold"

    _draw_grid(ax, plt, window_A, 0.35, ay0, SCELL, win_face, 11, lw=0.9)
    ax.text(0.35 + 1.5 * SCELL, ay0 + 0.16, "window", ha="center", va="bottom",
            fontsize=9.5, color=BLUE)
    ax.text(2.55, ay0 - 1.5 * SCELL, "⊙", ha="center", va="center", fontsize=17, color=INK)
    _draw_grid(ax, plt, KERNEL_A, 3.05, ay0, SCELL, ker_face, 11, lw=0.9)
    ax.text(3.05 + 1.5 * SCELL, ay0 + 0.16, "kernel", ha="center", va="bottom",
            fontsize=9.5, color=ORANGE)
    ax.text(5.25, ay0 - 1.5 * SCELL, "=", ha="center", va="center", fontsize=17, color=INK)
    _draw_grid(ax, plt, hadamard_A, 5.75, ay0, SCELL, plain, 11, lw=0.9)
    ax.text(5.75 + 1.5 * SCELL, ay0 + 0.16, "Hadamard product", ha="center", va="bottom",
            fontsize=9.5, color=INK)
    ax.text(7.95, ay0 - 1.5 * SCELL, "Σ", ha="center", va="center", fontsize=17, color=INK)
    ax.text(8.55, ay0 - 1.5 * SCELL,
            f"= (1+0+0) + ({MINUS}1{MINUS}1{MINUS}1) + 0 = {MINUS}2",
            ha="left", va="center", fontsize=12.5, color=GREEN, fontweight="bold")
    ax.text(0.35, ay0 - 3 * SCELL - 0.22,
            "this window sits fully inside the image, so the padding mode cannot affect it",
            ha="left", va="top", fontsize=9, color="#868e96")

    # ── 오른쪽 아래: 두 개의 캐비엇 상자 ──
    ax.text(11.3, ay0 + 0.30,
            '"dot product" here (p.149) = Hadamard product, then sum over the window\n'
            f"the kernel is NOT flipped  →  strictly this is CORRELATION, not convolution\n"
            f"flipping it swaps the +1 / {MINUS}1 rows and breaks the printed output",
            ha="left", va="top", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec="#ced4da"))
    ax.text(11.3, ay0 - 1.32,
            "padding mode changes ONLY row 1 (orange dashes); the other 20 cells are identical\n"
            f"    edge / replicate pad  →  [0 0 0 0 0]      = what the book prints\n"
            f"    zero pad              →  [{MINUS}1 {MINUS}1 {MINUS}1 {MINUS}1 {MINUS}1]\n"
            "the book never states which one it used (p.149 prose, p.151 code omit the pad line)",
            ha="left", va="top", fontsize=9.5, color="#8a5a00",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff9db", ec="#f0c33c"))

    ax.set_title("[Fig 6-4](A) reconstruction  —  image convolution: slide the kernel, "
                 "Hadamard-multiply the overlapping window, sum\n"
                 "5×5 binary image  ∗  3×3 horizontal edge detector  →  the bright bar answers "
                 f"{MINUS}2 on its OWN row and +2 one row BELOW  "
                 "(the +1 row of the kernel reads the row above the centre)",
                 fontsize=12.5, pad=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_gaussian(plt, path):
    """[그림 6-4](B) 재구성 — 난수 행렬을 가우스 커널로 평활."""
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 0.52, 1.0], height_ratios=[1, 1.5, 1],
                          wspace=0.30, hspace=0.0, bottom=0.24, top=0.90)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_ker = fig.add_subplot(gs[1, 1])
    ax_out = fig.add_subplot(gs[:, 2])

    for ax, data, title, color in [
        (ax_img, IMAGE_B, f"image  ({IMG_N}×{IMG_N} random matrix, seed={RNG_SEED})", BLUE),
        (ax_ker, KERNEL_B, f"kernel  ({KN}×{KN} gaussian)", ORANGE),
        (ax_out, SMOOTH_B, "result  (smoothed, same size)", GREEN),
    ]:
        ax.imshow(data, cmap="gray", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, color=color, fontweight="bold", pad=8)
        for s in ax.spines.values():
            s.set_edgecolor(color), s.set_linewidth(1.8)

    ax_img.set_xlabel(f"std = {IMAGE_B.std():.3f}", fontsize=10, color=INK, labelpad=6)
    ax_ker.set_xlabel("sum = 1  (weighted average)", fontsize=10, color=INK, labelpad=6)
    ax_out.set_xlabel(f"std = {SMOOTH_B.std():.3f}"
                      f"   →   variance ×{VAR_OUT / VAR_IN:.3f}",
                      fontsize=10, color=INK, labelpad=6)

    fig.text(0.408, 0.50, CIRC_AST, ha="center", va="center", fontsize=24, color=INK)
    fig.text(0.600, 0.50, "=", ha="center", va="center", fontsize=24, color=INK)

    fig.text(0.5, 0.015,
             f"G = exp( {MINUS}(X² + Y²) / σ ),  σ = 20,  "
             f"grid = linspace({MINUS}3, 3, 21)   then   kernel /= kernel.sum()\n"
             f"the kernel looks narrow only because imshow autoscales: its true corner/center "
             f"ratio is exp({MINUS}0.9) = {np.exp(-0.9):.4f}, i.e. it is a nearly flat disc\n"
             "sum = 1 makes every output pixel a weighted average of its neighbours "
             "→ the data keeps its original scale, the noise averages away\n"
             f"edges use replicate padding, so the outer {KN // 2}-pixel band is partly an echo "
             "of the border rows/columns — that is the streaking near the frame",
             ha="center", va="bottom", fontsize=9.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec="#ced4da"))

    fig.suptitle("[Fig 6-4](B) reconstruction  —  gaussian smoothing of a random matrix "
                 "(the 2-D version of the §3.2 time-series smoothing)",
                 fontsize=12.5, y=1.03)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render(outdir=HERE):
    """--fig: [그림 6-4] (A)·(B) 재구성 PNG 2장."""
    import matplotlib
    matplotlib.use("Agg")                              # headless — GUI 창을 띄우지 않는다
    import matplotlib.pyplot as plt

    # 규칙 12 "플롯 내부 라벨은 영어" ⇒ DejaVu Sans.
    # 실측 근거: DejaVu Sans 는 ⊛(U+229B)·∗(U+2217)·−(U+2212) 를 갖지만 한글이 없고,
    # AppleGothic 은 한글만 있고 위 세 글자가 전부 없다(→ 한글로 쓰면 연산자가 두부박스가 된다).
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.unicode_minus": True, "font.size": 11}):
        _fig_mechanism(plt, os.path.join(outdir, FIG_A))
        _fig_gaussian(plt, os.path.join(outdir, FIG_B))
    return [FIG_A, FIG_B]


if __name__ == "__main__":
    print("✅ §6.3 검증 통과")
    print(f"   (A) 상관(커널 미뒤집기)+가장자리복제 패딩 = [그림 6-4](A) 인쇄값 25/25 일치")
    print(f"       제로 패딩은 첫 행만 갈림: {out_zero[0].astype(int).tolist()} vs 인쇄값 [0,0,0,0,0]")
    print(f"       커널 뒤집기(진짜 합성곱)는 ±행이 한 칸 위로 → 인쇄값 불일치")
    print(f"       책 p.151 이중 for 문 전사본과 교차 일치(7×7 배열 안쪽 5×5)")
    print(f"       책이 강조한 창은 좌상단 IMAGE_A[0:3,0:3] "
          f"'내적' = {hadamard_sum(IMAGE_A[0:3, 0:3], KERNEL_A):.0f} = out[1,1] = {out_edge[1, 1]:.0f}")
    print(f"       이 스크립트의 시연 창은 안쪽 IMAGE_A[1:4,1:4] (값이 0이 아니고 패딩 모드 무관): "
          f"'내적' = {hadamard_sum(window_A, KERNEL_A):.0f} = out[2,2] = {out_edge[2, 2]:.0f}")
    print(f"   (B) 가우스 커널 {KERNEL_B.shape} 합={KERNEL_B.sum():.12f}, 대칭·중심최대, "
          f"모서리/중심={KERNEL_B.min() / KERNEL_B.max():.4f}=exp(-0.9)")
    print(f"       평활 분산 {VAR_IN:.4f} → {VAR_OUT:.4f} (×{VAR_OUT / VAR_IN:.4f}), "
          f"평균 {IMAGE_B.mean():+.4f} → {SMOOTH_B.mean():+.4f}")
    if "--fig" in sys.argv:
        print("🖼  저장:", ", ".join(render()))
