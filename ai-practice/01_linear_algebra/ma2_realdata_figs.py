"""[필수수학AI] §2.4 실제 데이터 예시 — 그림 2-1·2-2·2-3·2-6 재구성.

데이터 출처: 번역자 GitHub 저장소(옮긴이 저장소 — 책의 oreil.ly 단축링크와 동일 데이터,
Kaggle 로그인 불필요):
  https://github.com/EmjayAhn/essential-mathematics-for-ai/tree/main/chapter02/data

  - 500_Person_Gender_Height_Weight_Index.csv (n=500, Height=cm, Weight=kg, Index=0~5 BMI 구간)
    → oreil.ly/pxgwe 대응. 그림 2-1(전체) · 그림 2-3(Index==3 층화)
  - weight-height.csv (n=10,000, Height=inch, Weight=lb)
    → oreil.ly/8bE36 대응. 그림 2-2(전체) · 그림 2-6(Gender=='Female', n=5,000 → oreil.ly/rZNBS 대응)

⚠️ 단위 불일치 주의: dataset1은 cm/kg, dataset2는 inch/lb — 두 데이터셋을 같은 축에 얹거나
   비교하면 안 됨. 책·노트 어디에도 이 단위 차이가 명시돼 있지 않음.

검증(2026-07-21 실행 기준):
  전체(그림 2-1 대응) 상관계수      = 0.0    (책의 "패턴 없음" 서술과 정확히 일치)
  Index==3(그림 2-3 대응) 상관계수  = 0.913
  dataset2 전체(그림 2-2 대응) 상관계수 = 0.925
  dataset2 Female n                = 5,000  ("실제 여성 5,000명" 서술과 정확히 일치)

노트: 1단계/19 실제 vs 시뮬레이션 데이터 — 키-체중·조건부 층화·노이즈
그림: 1단계/attachments/MA그림2-1·2-2·2-3·2-6 (재구성)  · 시뮬 쪽(2-4·2-5)은 ma2_simulation.py

실행하면 이 스크립트와 같은 폴더에 fig2-1/2-2/2-3/2-6.png가 생성됨 — 그걸 Obsidian
vault의 1단계/attachments/로 복사하고 "MA그림2-N ... (재구성).png"로 이름을 바꾸면 됨
(2026-07-22 세션에서 생성한 현재 vault 파일도 이 스크립트의 출력물).
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=1.0)

URL1 = "https://raw.githubusercontent.com/EmjayAhn/essential-mathematics-for-ai/main/chapter02/data/500_Person_Gender_Height_Weight_Index.csv"
URL2 = "https://raw.githubusercontent.com/EmjayAhn/essential-mathematics-for-ai/main/chapter02/data/weight-height.csv"

df1 = pd.read_csv(URL1)   # Gender, Height(cm), Weight(kg), Index
df2 = pd.read_csv(URL2)   # Gender, Height(inch), Weight(lb)

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
OUT = "."  # 스크립트와 같은 폴더 — vault로는 수동(또는 별도 스텝) 복사


def jointplot(data, x, y, xlab, ylab, title, out_name):
    g = sns.jointplot(data=data, x=x, y=y, kind="reg", height=5.5,
                       scatter_kws={"s": 15, "color": BLUE, "alpha": 0.6},
                       line_kws={"color": ORANGE, "linewidth": 2})
    g.ax_joint.set_xlabel(xlab)
    g.ax_joint.set_ylabel(ylab)
    g.fig.suptitle(title, y=1.02)
    g.savefig(f"{OUT}/{out_name}", dpi=130, bbox_inches="tight")
    plt.close(g.fig)


# 검증: 책이 주장하는 상관관계 패턴이 실제로 재현되는지 먼저 확인
assert round(df1["Height"].corr(df1["Weight"]), 2) == 0.0
assert df1[df1.Index == 3]["Height"].corr(df1[df1.Index == 3]["Weight"]) > 0.85
assert df2["Height"].corr(df2["Weight"]) > 0.85
assert (df2.Gender == "Female").sum() == 5000

# 그림 2-1: dataset1 전체 — 패턴 없음
jointplot(df1, "Height", "Weight", "height (cm)", "weight (kg)",
          "Real data: no visible pattern (n=500)",
          "MA그림2-1 키체중 산점도 무패턴 (재구성).png")

# 그림 2-2: dataset2 전체 — 선형 + 쌍봉 마진(가우스 혼합 신호)
jointplot(df2, "Height", "Weight", "height (inch)", "weight (lb)",
          "Real data: linear pattern, bimodal marginals (n=10,000)",
          "MA그림2-2 키체중 선형관계 쌍봉 (재구성).png")

# 그림 2-3: dataset1, Index==3 층화 — 선형 회복(조건화의 힘)
sub = df1[df1.Index == 3]
jointplot(sub, "Height", "Weight", "height (cm)", "weight (kg)",
          f"Real data: Index=3 stratum -> linear (n={len(sub)})",
          "MA그림2-3 Index3 층화 선형 (재구성).png")

# 그림 2-6: dataset2 Female만 — 시뮬레이션 그림 2-5와 직접 비교용(같은 플레인 스캐터 스타일)
fem = df2[df2.Gender == "Female"]
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(fem.Height, fem.Weight, s=8, alpha=0.5, color=BLUE)
sns.regplot(data=fem, x="Height", y="Weight", scatter=False, ax=ax,
            line_kws={"color": ORANGE, "linewidth": 2})
ax.set_xlabel("height (inch)")
ax.set_ylabel("weight (lb)")
ax.set_title(f"Real data: female only (n={len(fem)}) - compare to simulated Fig 2-5")
fig.savefig(f"{OUT}/MA그림2-6 여성만 선형 (재구성).png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("통과 — 그림 2-1·2-2·2-3·2-6 재구성 완료, 책의 상관관계 서술과 수치 일치 확인.")
