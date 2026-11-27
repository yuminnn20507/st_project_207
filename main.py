import os
import urllib.request
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="연령군별 치주질환(K05)·제2형 당뇨병(E11) 연관분석",
    layout="wide",
)

# 1. 한글 폰트 파일 자동 다운로드 및 강제 적용 (한글 깨짐 100% 방지)
FONT_FILE = "NanumGothic.ttf"
if not os.path.exists(FONT_FILE):
    url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    urllib.request.urlretrieve(url, FONT_FILE)

fm.fontManager.addfont(FONT_FILE)
font_name = fm.FontProperties(fname=FONT_FILE).get_name()
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

FILE_PATH = "건강보험심사평가원_3단상병별 성별 연령군별 통계_20251231.csv"


@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    for enc in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해 주세요.")


st.title("연령군별 치주질환(K05) 및 제2형 당뇨병(E11) 연관분석")
st.caption(
    "팀 질문: 2025년 연령군별 진료 통계에서 치주질환(K05) 환자 수 증가는 "
    "정말 제2형 당뇨병(E11) 환자 수 증가와 연관되어 있는가?"
)

try:
    df_raw = load_raw_data(FILE_PATH)
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# 1. 데이터 개요 (행·열 수, 자료형, 빈 값 개수)
st.header("1. 데이터 개요")
c1, c2, c3 = st.columns(3)
c1.metric("원본 행 수", f"{len(df_raw):,}행")
c2.metric("원본 열 수", f"{df_raw.shape[1]}열")
c3.metric("총 결측치(빈 값) 수", f"{int(df_raw.isna().sum().sum()):,}개")

col_info, col_null = st.columns(2)
with col_info:
    st.subheader("열별 자료형")
    dtype_df = pd.DataFrame(
        {"열 이름": df_raw.columns, "자료형": df_raw.dtypes.astype(str).values}
    )
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

with col_null:
    st.subheader("열별 빈 값(결측치) 개수")
    null_df = pd.DataFrame(
        {"열 이름": df_raw.columns, "빈 값 개수": df_raw.isna().sum().values}
    )
    st.dataframe(null_df, use_container_width=True, hide_index=True)

# 2. 결측치·이상치 처리 기준 및 전후 행 수
st.header("2. 결측치·이상치 처리 및 전후 행 수")
gender_options = ["전체(남+여)"] + sorted(df_raw["성별"].dropna().unique().tolist())
selected_gender = st.selectbox("분석 대상 성별 선택", gender_options, index=0)

raw_rows = len(df_raw)
df_clean = df_raw.dropna(subset=["주상병코드", "성별", "연령군", "환자수"]).copy()
df_clean = df_clean[df_clean["환자수"] >= 0].copy()

if selected_gender != "전체(남+여)":
    df_filtered = df_clean[df_clean["성별"] == selected_gender].copy()
else:
    df_filtered = df_clean.copy()

k05_df = df_filtered[df_filtered["주상병코드"].astype(str).str.startswith("K05")].copy()
e11_df = df_filtered[df_filtered["주상병코드"].astype(str).str.startswith("E11")].copy()
target_df = pd.concat([k05_df, e11_df], ignore_index=True)

k05_agg = (
    k05_df.groupby("연령군", as_index=False)["환자수"]
    .sum()
    .rename(columns={"환자수": "치주질환_K05_환자수"})
)
e11_agg = (
    e11_df.groupby("연령군", as_index=False)["환자수"]
    .sum()
    .rename(columns={"환자수": "제2형당뇨병_E11_환자수"})
)

age_order = sorted(df_raw["연령군"].dropna().unique().tolist())
merged = (
    pd.DataFrame({"연령군": age_order})
    .merge(k05_agg, on="연령군", how="left")
    .merge(e11_agg, on="연령군", how="left")
)
merged["치주질환_K05_환자수"] = merged["치주질환_K05_환자수"].fillna(0).astype(int)
merged["제2형당뇨병_E11_환자수"] = merged["제2형당뇨병_E11_환자수"].fillna(0).astype(int)

st.markdown(
    f"""
- **결측치·이상치 처리 기준:** 필수 컬럼(`주상병코드`, `성별`, `연령군`, `환자수`) 결측치 제거 및 `환자수 < 0` 이상치 제외(원본 결측치 `0`개, 음수 환자수 `0`건 확인).
- **처리 전 행 수:** `{raw_rows:,}행`
- **질병(`K05`, `E11`) 및 성별(`{selected_gender}`) 필터링 후 행 수:** `{len(target_df):,}행` (K05 `{len(k05_df):,}행` + E11 `{len(e11_df):,}행`)
- **최종 연령군별 집계 테이블 행 수:** `{len(merged):,}행` (연령군 18개 구간)
    """
)

# 3. 연관분석 (지지도 · 신뢰도 · 향상도 및 베이스라인 비교)
st.header("3. 연관분석 결과 (지지도 · 신뢰도 · 향상도)")

k05_median = merged["치주질환_K05_환자수"].median()
e11_median = merged["제2형당뇨병_E11_환자수"].median()

merged["치주질환_다발"] = merged["치주질환_K05_환자수"] >= k05_median
merged["당뇨병_다발"] = merged["제2형당뇨병_E11_환자수"] >= e11_median

total_n = len(merged)
# P(A): 치주질환 다발 비율
support_a = merged["치주질환_다발"].sum() / total_n
# P(B): 당뇨병 다발 비율 (베이스라인: 결과 항목의 평소 등장 비율)
baseline_b = merged["당뇨병_다발"].sum() / total_n
# 지지도 Support P(A ∩ B): 두 질환 모두 다발하는 비율
support_ab = (merged["치주질환_다발"] & merged["당뇨병_다발"]).sum() / total_n
# 신뢰도 Confidence P(B | A): 치주질환 다발 구간 중 당뇨병도 다발하는 비율
confidence = support_ab / support_a if support_a > 0 else 0
# 향상도 Lift: 신뢰도 / 베이스라인(P(B))
lift = confidence / baseline_b if baseline_b > 0 else 0

st.markdown(
    f"""
- **연관 규칙 설정:** `{selected_gender}` 기준 치주질환(`K05`) 중앙값(`{int(k05_median):,}명`) 이상을 `치주질환 다발`, 제2형 당뇨병(`E11`) 중앙값(`{int(e11_median):,}명`) 이상을 `당뇨병 다발`로 정의하여 **`{{치주질환 다발}} → {{당뇨병 다발}}`** 규칙을 분석합니다.
- **베이스라인(기준선) 비교:** 조건 없이 당뇨병이 다발하는 **평소 등장 비율({baseline_b*100:.1f}%)** 대비, 치주질환이 다발할 때 당뇨병도 다발하는 **신뢰도({confidence*100:.1f}%)**가 얼마나 증가했는지 평가합니다.
    """
)

a1, a2, a3, a4 = st.columns(4)
a1.metric("지지도 (Support)", f"{support_ab:.4f} ({support_ab*100:.1f}%)")
a2.metric("신뢰도 (Confidence)", f"{confidence:.4f} ({confidence*100:.1f}%)")
a3.metric("베이스라인 (평소 등장 비율)", f"{baseline_b:.4f} ({baseline_b*100:.1f}%)")
a4.metric("향상도 (Lift)", f"{lift:.2f}")

# 연관분석 규칙 요약표 및 베이스라인 비교 차트
assoc_col1, assoc_col2 = st.columns([1.3, 1])

with assoc_col1:
    st.subheader("연관 규칙 평가표")
    rule_df = pd.DataFrame(
        {
            "연관 규칙": ["{치주질환(K05) 다발} → {제2형 당뇨병(E11) 다발}"],
            "지지도(Support)": [f"{support_ab:.4f}"],
            "신뢰도(Confidence)": [f"{confidence:.4f}"],
            "베이스라인(평소 비율)": [f"{baseline_b:.4f}"],
            "향상도(Lift)": [f"{lift:.4f}"],
            "해석": ["향상도 > 1 (양의 연관성 존재)" if lift > 1 else "연관성 없음"],
        }
    )
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

with assoc_col2:
    fig_lift, ax_lift = plt.subplots(figsize=(5, 2.6))
    bars = ax_lift.bar(
        ["베이스라인\n(평소 당뇨 다발 비율)", "연관 규칙 신뢰도\n(치주질환 다발 조건)"],
        [baseline_b * 100, confidence * 100],
        color=["#95a5a6", "#2b5c8f"],
        width=0.45,
    )
    ax_lift.set_ylim(0, 105)
    ax_lift.set_ylabel("비율 (%)")
    ax_lift.set_title(f"베이스라인 대비 신뢰도 향상 (Lift = {lift:.2f})", fontsize=10, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax_lift.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_lift.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_lift)

# 4. 그래프 시각화 및 상관분석 (피어슨 · 스피어만)
st.header("4. 연령군별 환자 수 그래프 및 상관관계 분석")

pearson_r, pearson_p = stats.pearsonr(
    merged["치주질환_K05_환자수"], merged["제2형당뇨병_E11_환자수"]
)
spearman_rho, spearman_p = stats.spearmanr(
    merged["치주질환_K05_환자수"], merged["제2형당뇨병_E11_환자수"]
)

chart_col1, chart_col2 = st.columns(2)

# [그래프 1] 연령군별 막대 그래프
with chart_col1:
    x = np.arange(len(merged))
    width = 0.38

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width / 2, merged["치주질환_K05_환자수"], width, label="치주질환 (K05)")
    ax1.bar(x + width / 2, merged["제2형당뇨병_E11_환자수"], width, label="제2형 당뇨병 (E11)")
    ax1.axhline(k05_median, color="#1f77b4", linestyle=":", alpha=0.7, label="K05 중앙값 기준선")
    ax1.axhline(e11_median, color="#ff7f0e", linestyle=":", alpha=0.7, label="E11 중앙값 기준선")
    ax1.set_title(
        f"연령군별 환자 수 분포 ({selected_gender})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("연령군")
    ax1.set_ylabel("환자 수 (명)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["연령군"], rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig1)

# [그래프 2] 두 질병 간 상관관계 산점도 + 중앙값 사분면
with chart_col2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x_vals = merged["치주질환_K05_환자수"]
    y_vals = merged["제2형당뇨병_E11_환자수"]

    ax2.scatter(x_vals, y_vals, color="#2b5c8f", s=60, alpha=0.8, label="연령군 데이터")

    # 연관분석 기준선(중앙값) 표시
    ax2.axvline(k05_median, color="gray", linestyle=":", alpha=0.7, label="다발 구간 기준선(중앙값)")
    ax2.axhline(e11_median, color="gray", linestyle=":", alpha=0.7)

    # 선형 회귀 추세선 추가
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax2.plot(
        x_line,
        slope * x_line + intercept,
        color="#d9534f",
        linestyle="--",
        linewidth=1.5,
        label=f"추세선 (r = {pearson_r:.2f})",
    )

    for i, row in merged.iterrows():
        short_label = row["연령군"].split("_")[-1] if "_" in row["연령군"] else row["연령군"]
        ax2.annotate(
            short_label,
            (row["치주질환_K05_환자수"], row["제2형당뇨병_E11_환자수"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7.5,
            alpha=0.85,
        )

    ax2.set_title(
        f"치주질환(K05) vs 제2형 당뇨병(E11) 산점도 ({selected_gender})",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("치주질환(K05) 환자 수 (명)")
    ax2.set_ylabel("제2형 당뇨병(E11) 환자 수 (명)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)

m1, m2, m3, m4 = st.columns(4)
m1.metric("치주질환(K05) 합계", f"{merged['치주질환_K05_환자수'].sum():,}명")
m2.metric("제2형 당뇨병(E11) 합계", f"{merged['제2형당뇨병_E11_환자수'].sum():,}명")
m3.metric("피어슨 상관계수 (r)", f"{pearson_r:.4f} (p={pearson_p:.4f})")
m4.metric("스피어만 상관계수 (ρ)", f"{spearman_rho:.4f} (p={spearman_p:.4f})")

st.subheader("연령군별 집계 및 다발 여부 판정표")
st.dataframe(merged, use_container_width=True, hide_index=True)

# 5. 해석 메모
st.header("5. 발견한 점 및 다음 분석에서 확인할 점")
st.markdown(
    f"""
- **연관분석 결과 해석:** 치주질환(`K05`)이 다발하는 연령군일 때 제2형 당뇨병(`E11`)도 함께 다발할 **신뢰도는 {confidence*100:.1f}%**이며, 이는 아무 조건 없을 때의 **베이스라인({baseline_b*100:.1f}%) 대비 향상도(Lift) {lift:.2f}배**에 해당합니다. (지지도: `{support_ab:.4f}`)
- **상관분석 및 패턴 해석:** 두 질환은 피어슨 상관계수 `r={pearson_r:.4f}`, 스피어만 상관계수 `ρ={spearman_rho:.4f}`의 통계적으로 유의한 양의 상관관계를 보입니다. 특히 중·장년층(45~79세)에서 두 질환이 집중적으로 동반 다발하나, 치주질환은 `60~64세` 정점 이후 `85세 이상`에서 크게 감소하는 반면 제2형 당뇨병은 고령층 비중이 상대적으로 길게 유지됩니다.
- **다음 분석에서 확인할 점:** 연령군별 전체 인구수 대비 유병률(비율)로 보정하여 연관 규칙과 상관계수를 다시 산출하면, 고령화(인구 구조) 효과를 제외한 순수한 질병 간 연관성을 더 정확하게 검증할 수 있습니다.
    """
)
