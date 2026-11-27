import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="연령군별 치주질환(K05)·제2형 당뇨병(E11) 비교",
    layout="wide",
)

# 한글 폰트 설정 (환경에 맞게 필요시 조정)
plt.rcParams["font.family"] = "Malgun Gothic"
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


st.title("연령군별 치주질환(K05) 및 제2형 당뇨병(E11) 환자 수 분석")
st.caption(
    "팀 질문: 2025년 연령군별 진료 통계에서 치주질환(K05) 환자 수와 "
    "제2형 당뇨병(E11) 환자 수는 연령대에 따라 어떤 관계와 차이를 보이는가?"
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

# 3. 그래프 및 상관분석 결과
st.header("3. 연령군별 환자 수 그래프 및 상관계수")

x = np.arange(len(merged))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width / 2, merged["치주질환_K05_환자수"], width, label="치주질환 (K05)")
ax.bar(x + width / 2, merged["제2형당뇨병_E11_환자수"], width, label="제2형 당뇨병 (E11)")
ax.set_title(
    f"연령군별 치주질환(K05) 및 제2형 당뇨병(E11) 환자 수 ({selected_gender})",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("연령군")
ax.set_ylabel("환자 수 (명)")
ax.set_xticks(x)
ax.set_xticklabels(merged["연령군"], rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
st.pyplot(fig)

# 상관계수 계산
pearson_r, pearson_p = stats.pearsonr(
    merged["치주질환_K05_환자수"], merged["제2형당뇨병_E11_환자수"]
)
spearman_rho, spearman_p = stats.spearmanr(
    merged["치주질환_K05_환자수"], merged["제2형당뇨병_E11_환자수"]
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("치주질환(K05) 합계", f"{merged['치주질환_K05_환자수'].sum():,}명")
m2.metric("제2형 당뇨병(E11) 합계", f"{merged['제2형당뇨병_E11_환자수'].sum():,}명")
m3.metric("피어슨 상관계수 (r)", f"{pearson_r:.4f} (p={pearson_p:.4f})")
m4.metric("스피어만 상관계수 (ρ)", f"{spearman_rho:.4f} (p={spearman_p:.4f})")

st.subheader("연령군별 집계표")
st.dataframe(merged, use_container_width=True, hide_index=True)

# 4. 해석 메모 (예상과 달랐던 점 / 다음 확인 사항)
st.header("4. 발견한 점 및 다음 분석에서 확인할 점")
st.markdown(
    """
    - **근거 수치로 본 패턴:** 두 질환 모두 중·장년층 구간에서 환자 수가 크게 늘어나 연령군별 분포 간 양의 상관관계를 보이지만, 치주질환(`K05`)은 `13_60~64세` 구간에서 정점을 찍은 뒤 `18_85세 이상`에서 크게 줄어들고, 제2형 당뇨병(`E11`)은 `14_65~69세` 구간 정점 이후에도 고령층 비중이 상대적으로 길게 이어집니다.
    - **예상과 달랐던 점:** 막연히 고령층일수록 두 질환 환자 수가 계속 증가할 것으로 예상했으나, `70세 이상` 구간에서는 두 질환 모두 절대 환자 수가 감소하며 특히 치주질환 감소 폭이 큽니다.
    - **다음 분석에서 확인할 점:** 연령군별 인구수 대비 환자 비율이나 성별(`남`/`여`) 차이를 함께 비교하면 고령층에서 환자 수가 줄어드는 이유를 더 정확히 파악할 수 있습니다.
    """
) d
