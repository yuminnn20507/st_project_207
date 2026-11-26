import re
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 치주질환(K05)과 2형 당뇨병(E11) 환자 수 상관관계 탐구
# 실행 파일: code/main.py
# ============================================================

st.set_page_config(
    page_title="치주질환과 2형 당뇨병",
    page_icon="🦷",
    layout="wide",
)

YEARS = list(range(2021, 2026))
PERIODONTAL = "치주질환(K05)"
DIABETES = "2형 당뇨병(E11)"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.title("🦷 치주질환과 2형 당뇨병 환자 수의 관계")
st.markdown(
    """
    **우리 팀의 탐구 질문**  
    최근 5년간(2021~2025년) 우리나라 성인의 연령대별 치주질환 환자 수와
    2형 당뇨병 환자 수 사이에는 어떤 상관관계가 있을까?

    같은 연도·연령대에서 두 질환의 환자 수를 짝지어 비교합니다.
    상관관계는 두 변수가 함께 변하는 정도를 뜻하며, 인과관계를 증명하지는 않습니다.
    """
)


# ------------------------- 공통 함수 -------------------------

def read_csv_bytes(file_bytes):
    errors = []
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding=encoding, low_memory=False)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV를 읽지 못했습니다. 파일 형식과 인코딩을 확인해 주세요.")


def norm_name(value):
    return re.sub(r"\s+", "", str(value)).lower()


def find_column(columns, candidates):
    normalized = {norm_name(col): col for col in columns}
    for candidate in candidates:
        key = norm_name(candidate)
        if key in normalized:
            return normalized[key]
    for col in columns:
        key = norm_name(col)
        for candidate in candidates:
            if norm_name(candidate) in key:
                return col
    return None


def parse_year(series):
    extracted = series.astype(str).str.extract(r"(20\d{2})", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def parse_age(value):
    text = str(value).strip().replace(" ", "")
    if not text or text.lower() == "nan" or "미만" in text:
        return np.nan
    found = re.search(r"\d+", text)
    return int(found.group()) if found else np.nan


def clean_code(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def correlation_result(frame, method):
    valid = frame[[PERIODONTAL, DIABETES]].dropna()
    if len(valid) < 3:
        return None
    if valid[PERIODONTAL].nunique() < 2 or valid[DIABETES].nunique() < 2:
        return None
    if method == "Pearson":
        r, p = pearsonr(valid[PERIODONTAL], valid[DIABETES])
    else:
        r, p = spearmanr(valid[PERIODONTAL], valid[DIABETES])
    return {"r": float(r), "p": float(p), "n": int(len(valid))}


def strength_text(r):
    absolute = abs(r)
    if absolute < 0.2:
        strength = "매우 약한"
    elif absolute < 0.4:
        strength = "약한"
    elif absolute < 0.6:
        strength = "중간 정도의"
    elif absolute < 0.8:
        strength = "강한"
    else:
        strength = "매우 강한"
    direction = "양의" if r > 0 else "음의" if r < 0 else "거의 없는"
    return f"{strength} {direction} 상관관계"


def make_process_log(file_name, original, cleaned, steps, missing_summary, outlier_count):
    lines = [
        "치주질환·2형 당뇨병 데이터 분석 기록",
        f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"사용 파일: {file_name}",
        f"원본 데이터 크기: {original.shape[0]}행 × {original.shape[1]}열",
        f"전처리 후 분석 자료: {cleaned.shape[0]}행 × {cleaned.shape[1]}열",
        "",
        "[전처리 단계별 행 수]",
    ]
    lines.extend([f"- {name}: {count:,}행" for name, count in steps])
    lines += [
        "",
        "[원본 열별 결측치]",
        missing_summary.to_string(index=False),
        "",
        f"[이상치 점검] IQR 기준 이상치 표시 건수(두 질환 환자 수 열 합계): {outlier_count:,}건",
        "이상치는 자동 삭제하지 않고 표시·검토 대상으로 남겼습니다.",
        "",
        "[전처리 기준]",
        "- 분석 기간: 2021~2025년",
        "- 성인 기준: 연령대 시작 나이 만 20세 이상",
        "- 치주질환 코드: K05로 시작하는 코드",
        "- 2형 당뇨병 코드: E11로 시작하는 코드",
        "- 환자 수가 결측이거나 숫자로 변환되지 않는 행은 분석에서 제외",
        "- 환자 수가 음수인 행은 비정상 값으로 제외",
        "- 0명은 실제 값일 수 있으므로 삭제하지 않음",
        "- 성별 자료가 남녀로 나뉘어 있으면 남녀를 합산",
        "- 같은 연도·연령대·질환에 여러 행이 있으면 합산",
        "- 두 질환의 값이 모두 있는 동일 연도·연령대만 상관관계 분석에 사용",
        "- IQR 이상치는 임의 삭제하지 않고 원자료와 함께 확인",
        "",
        "[해석 주의]",
        "- 환자 수는 연령대별 인구 규모의 영향을 받습니다.",
        "- 상관관계는 인과관계를 뜻하지 않습니다.",
        "- 보험 청구자료는 실제 유병자 전체와 차이가 있을 수 있습니다.",
    ]
    return "\n".join(lines)


# ------------------------- 데이터 선택 -------------------------

st.sidebar.header("데이터 불러오기")
uploaded = st.sidebar.file_uploader("CSV 파일 직접 업로드", type=["csv"])
available_files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []

if uploaded is not None:
    file_name = uploaded.name
    try:
        raw = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:
        st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()
elif available_files:
    chosen = st.sidebar.selectbox(
        "data 폴더의 CSV 파일 선택",
        available_files,
        format_func=lambda path: path.name,
    )
    file_name = chosen.name
    try:
        raw = read_csv_bytes(chosen.read_bytes())
    except Exception as exc:
        st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()
else:
    st.warning(
        "분석할 CSV 파일이 없습니다. 왼쪽에서 CSV를 업로드하거나 "
        "GitHub 저장소의 data 폴더에 CSV 파일을 넣어 주세요."
    )
    st.info("필요한 열: 진료년도, 주상병코드, 연령군, 환자수. 성별 열은 선택 사항입니다.")
    st.stop()

# ------------------------- 데이터 개요 -------------------------

st.header("1. 데이터 개요")
st.write(
    f"**파일명:** `{file_name}`  ·  **원본 크기:** "
    f"{raw.shape[0]:,}행 × {raw.shape[1]:,}열"
)
st.write("**열별 자료형과 결측치 개수**")
overview = pd.DataFrame({
    "열 이름": raw.columns.astype(str),
    "자료형": [str(dtype) for dtype in raw.dtypes],
    "결측치 개수": [int(raw[col].isna().sum()) for col in raw.columns],
    "결측치 비율(%)": [
        round(raw[col].isna().mean() * 100, 2) for col in raw.columns
    ],
})
st.dataframe(overview, use_container_width=True, hide_index=True)

with st.expander("원본 데이터 미리 보기"):
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)

# ------------------------- 열 자동 인식 -------------------------

year_col = find_column(raw.columns, ["진료년도", "진료연도", "연도", "년도"])
code_col = find_column(raw.columns, ["주상병코드", "상병코드", "질병코드", "상병"])
age_col = find_column(raw.columns, ["연령군", "연령구분", "연령"])
count_col = find_column(raw.columns, ["환자수", "진료인원", "환자 수", "환자수(명)"])
sex_col = find_column(raw.columns, ["성별"])

st.header("2. 변수 선택")
st.caption("열 이름을 자동으로 찾았습니다. 자동 인식이 틀리면 아래 선택 상자에서 직접 지정하세요.")
all_columns = ["선택 안 함"] + list(raw.columns)

def column_picker(label, detected, required=True):
    default_index = all_columns.index(detected) if detected in all_columns else 0
    selected = st.selectbox(label, all_columns, index=default_index, key=f"pick_{label}")
    if selected == "선택 안 함" and required:
        st.error(f"{label} 열을 선택해야 합니다.")
        st.stop()
    return None if selected == "선택 안 함" else selected

c1, c2, c3 = st.columns(3)
with c1:
    year_col = column_picker("진료년도 열", year_col)
    code_col = column_picker("주상병코드 열", code_col)
with c2:
    age_col = column_picker("연령군 열", age_col)
    count_col = column_picker("환자 수 열", count_col)
with c3:
    sex_col = column_picker("성별 열(선택)", sex_col, required=False)

st.write(
    f"선택된 변수: 연도=`{year_col}`, 상병코드=`{code_col}`, "
    f"연령=`{age_col}`, 환자 수=`{count_col}`, "
    f"성별=`{sex_col if sex_col else '없음'}`"
)

# ------------------------- 전처리 -------------------------

st.header("3. 전처리 및 결측치·이상치 점검")

work = raw.copy()
steps = [("원본", len(work))]

work["_연도"] = parse_year(work[year_col])
work["_코드"] = work[code_col].map(clean_code)
work["_연령"] = work[age_col].astype(str).str.strip()
work["_연령시작"] = work["_연령"].map(parse_age)
work["_환자수"] = pd.to_numeric(
    work[count_col].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("명", "", regex=False),
    errors="coerce",
)
steps.append(("필수 변수 정리 후(행 수 유지)", len(work)))

before_valid = len(work)
work = work.dropna(subset=["_연도", "_연령시작", "_환자수"])
steps.append(("연도·연령·환자 수 결측/변환 불가 행 제외", len(work)))

before_year = len(work)
work = work[work["_연도"].isin(YEARS)]
steps.append(("2021~2025년 자료만 선택", len(work)))

before_adult = len(work)
work = work[work["_연령시작"] >= 20]
steps.append(("만 20세 이상 자료만 선택", len(work)))

before_code = len(work)
work["_질환"] = np.select(
    [work["_코드"].str.startswith("K05"), work["_코드"].str.startswith("E11")],
    [PERIODONTAL, DIABETES],
    default="",
)
work = work[work["_질환"] != ""]
steps.append(("K05·E11 코드만 선택", len(work)))

before_negative = len(work)
work = work[work["_환자수"] >= 0]
steps.append(("음수 환자 수 제외(0명은 유지)", len(work)))

if work.empty:
    st.error("전처리 후 자료가 없습니다. 연도·상병코드·연령·환자 수 열을 확인해 주세요.")
    st.stop()

# 남녀가 따로 존재하면 남녀 행만 합산하고, 합계 행은 중복 집계하지 않음
if sex_col is not None:
    sex_text = work[sex_col].astype(str).str.strip()
    male = sex_text.str.contains(r"남|male", case=False, regex=True)
    female = sex_text.str.contains(r"여|female", case=False, regex=True)
    total = sex_text.str.contains(r"계|전체|합계|total", case=False, regex=True)
    if male.any() and female.any():
        work = work[male | female]
    elif total.any():
        work = work[total]

grouped = (
    work.groupby(["_연도", "_연령", "_연령시작", "_질환"], as_index=False)["_환자수"]
    .sum()
    .rename(columns={
        "_연도": "연도",
        "_연령": "연령대",
        "_연령시작": "연령시작",
        "_질환": "질환",
        "_환자수": "환자수",
    })
)

paired = grouped.pivot_table(
    index=["연도", "연령대", "연령시작"],
    columns="질환",
    values="환자수",
    aggfunc="sum",
).reset_index()
paired.columns.name = None
for disease in [PERIODONTAL, DIABETES]:
    if disease not in paired.columns:
        paired[disease] = np.nan

paired_before = len(paired)
paired = paired.dropna(subset=[PERIODONTAL, DIABETES]).copy()
paired = paired.sort_values(["연도", "연령시작"]).reset_index(drop=True)
steps.append(("두 질환 모두 값이 있는 연도·연령대 짝", len(paired)))

if paired.empty:
    st.error("같은 연도·연령대에 두 질환의 값이 모두 존재하는 자료가 없습니다.")
    st.stop()

# IQR 이상치 플래그: 삭제하지 않고 개수만 점검
outlier_flags = {}
outlier_total = 0
for col in [PERIODONTAL, DIABETES]:
    q1 = paired[col].quantile(0.25)
    q3 = paired[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    flags = (paired[col] < low) | (paired[col] > high)
    outlier_flags[col] = flags
    outlier_total += int(flags.sum())

paired["이상치_점검"] = (
    outlier_flags[PERIODONTAL] | outlier_flags[DIABETES]
)
paired["연도"] = paired["연도"].astype(int)

# 단계별 처리 결과 표시
st.subheader("전처리 단계별 행 수")
step_df = pd.DataFrame(steps, columns=["처리 단계", "남은 행 수"])
step_df["제외 행 수(직전 단계 대비)"] = step_df["남은 행 수"].shift(1) - step_df["남은 행 수"]
step_df["제외 행 수(직전 단계 대비)"] = step_df["제외 행 수(직전 단계 대비)"].fillna(0).astype(int)
st.dataframe(step_df, use_container_width=True, hide_index=True)

a, b, c, d = st.columns(4)
a.metric("원본 행 수", f"{len(raw):,}")
b.metric("분석에 사용한 짝", f"{len(paired):,}")
c.metric("이상치 표시 행", f"{int(paired['이상치_점검'].sum()):,}")
d.metric("분석 연도", f"{paired['연도'].nunique()}개")

st.info(
    "전처리 기준: 2021~2025년, 만 20세 이상, K05·E11 코드, "
    "환자 수 0 이상. 결측·변환 불가·음수 값은 제외했습니다. "
    "IQR 기준 이상치는 임의로 삭제하지 않고 표시하여 검토합니다."
)

with st.expander("이상치로 표시된 자료 확인"):
    st.write("IQR(사분위 범위) 기준으로 이상치가 표시된 행입니다. 자동 삭제하지 않았습니다.")
    st.dataframe(
        paired[paired["이상치_점검"]],
        use_container_width=True,
        hide_index=True,
    )

# ------------------------- 시각화 -------------------------

st.header("4. 그래프 분석")

st.subheader("그래프 1. 연령대별 두 질환 환자 수 비교")
selected_year = st.selectbox(
    "비교할 연도",
    sorted(paired["연도"].unique()),
    index=len(sorted(paired["연도"].unique())) - 1,
)
one_year = paired[paired["연도"] == selected_year].sort_values("연령시작")
long_year = one_year.melt(
    id_vars=["연령대", "연령시작"],
    value_vars=[PERIODONTAL, DIABETES],
    var_name="질환",
    value_name="환자 수(명)",
)
fig1 = px.bar(
    long_year,
    x="연령대",
    y="환자 수(명)",
    color="질환",
    barmode="group",
    title=f"{selected_year}년 연령대별 치주질환(K05)·2형 당뇨병(E11) 환자 수",
    labels={"연령대": "연령대", "환자 수(명)": "환자 수(명)", "질환": "질환"},
)
fig1.update_layout(xaxis_title="연령대", yaxis_title="환자 수(명)")
st.plotly_chart(fig1, use_container_width=True)
st.caption("해석할 점: 연령대가 높아질수록 두 질환의 환자 수가 함께 증가하는지, 증가 폭은 같은지 살펴보세요.")

st.subheader("그래프 2. 치주질환 환자 수와 2형 당뇨병 환자 수의 상관관계")
scatter = px.scatter(
    paired,
    x=PERIODONTAL,
    y=DIABETES,
    color=paired["연도"].astype(str),
    hover_data=["연도", "연령대", "이상치_점검"],
    title="2021~2025년 연도·연령대별 치주질환 환자 수와 2형 당뇨병 환자 수",
    labels={
        PERIODONTAL: "치주질환 환자 수(명)",
        DIABETES: "2형 당뇨병 환자 수(명)",
        "color": "진료년도",
    },
)
scatter.update_layout(
    xaxis_title="치주질환 환자 수(명)",
    yaxis_title="2형 당뇨병 환자 수(명)",
)
st.plotly_chart(scatter, use_container_width=True)
st.caption("점 하나는 특정 연도의 특정 연령대입니다. 오른쪽 위로 모일수록 두 환자 수가 함께 큰 경향입니다.")

st.subheader("그래프 3. 연도별 연령대 자료의 상관계수")
annual_rows = []
for year in sorted(paired["연도"].unique()):
    subset = paired[paired["연도"] == year]
    for method in ["Pearson", "Spearman"]:
        result = correlation_result(subset, method)
        if result:
            annual_rows.append({
                "연도": year,
                "방법": method,
                "상관계수": result["r"],
                "p-value": result["p"],
                "자료 수": result["n"],
            })
annual = pd.DataFrame(annual_rows)

if not annual.empty:
    fig3 = px.line(
        annual,
        x="연도",
        y="상관계수",
        color="방법",
        markers=True,
        title="연도별 연령대 환자 수 상관계수 변화",
        hover_data=["p-value", "자료 수"],
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="gray")
    fig3.update_yaxes(range=[-1.05, 1.05], title="상관계수")
    fig3.update_xaxes(title="진료년도")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(annual.round(4), use_container_width=True, hide_index=True)
else:
    st.warning("연도별 상관계수를 계산하기 위한 연령대 자료가 충분하지 않습니다.")

# ------------------------- 지표와 해석 -------------------------

st.header("5. 사용한 지표와 결과 해석")
method = st.radio("전체 자료 상관계수", ["Pearson", "Spearman", "두 방법 비교"], horizontal=True)

pearson = correlation_result(paired, "Pearson")
spearman = correlation_result(paired, "Spearman")

if method == "Pearson":
    chosen_result = pearson
    chosen_name = "Pearson"
elif method == "Spearman":
    chosen_result = spearman
    chosen_name = "Spearman"
else:
    chosen_result = None
    chosen_name = "두 방법"

m1, m2 = st.columns(2)
for col, label, result in [
    (m1, "피어슨 상관계수", pearson),
    (m2, "스피어만 상관계수", spearman),
]:
    with col:
        if result:
            st.metric(label, f"{result['r']:.3f}")
            st.write(f"p-value: **{result['p']:.4g}** · 자료 수: **{result['n']}개**")
            st.write(strength_text(result["r"]))
        else:
            st.warning(f"{label}: 계산 가능한 자료가 부족합니다.")

st.markdown(
    """
    - **피어슨 상관계수:** 두 환자 수 사이의 직선(선형) 관계를 측정합니다.
    - **스피어만 상관계수:** 값의 순위가 함께 증가하거나 감소하는 관계를 측정합니다.
    - **기준선 0:** 선형 또는 순위 상관이 없다는 기준입니다.
    - **p-value:** 관찰된 상관이 우연만으로 나타났다고 보기 어려운지 판단할 때 참고합니다.
      일반적으로 0.05 미만을 통계적으로 유의한 결과의 참고 기준으로 사용합니다.
    """
)

# 동적으로 계산한 인사이트
st.subheader("그래프에서 찾을 수 있는 근거 수치")
if pearson:
    st.write(
        f"- 전체 자료의 피어슨 상관계수는 **{pearson['r']:.3f}**이며, "
        f"기준선 0과 비교하면 {strength_text(pearson['r'])}입니다."
    )
if spearman:
    st.write(
        f"- 전체 자료의 스피어만 상관계수는 **{spearman['r']:.3f}**입니다. "
        "피어슨 값과 비교해 선형 관계와 순위 관계가 비슷한지 확인할 수 있습니다."
    )

age_summary = (
    paired.groupby("연령대", as_index=False)
    .agg(
        평균_치주질환환자수=(PERIODONTAL, "mean"),
        평균_당뇨환자수=(DIABETES, "mean"),
        자료수=("연도", "count"),
    )
    .sort_values("평균_치주질환환자수", ascending=False)
)
if not age_summary.empty:
    top_age = age_summary.iloc[0]
    st.write(
        f"- 분석 자료에서 치주질환 환자 수 평균이 가장 큰 연령대는 "
        f"**{top_age['연령대']}**이며, 평균 환자 수는 "
        f"**{top_age['평균_치주질환환자수']:,.1f}명**입니다."
    )
    st.dataframe(age_summary.round(1), use_container_width=True, hide_index=True)

# 기대와 다른 점을 찾도록 실제 값 기반으로 보여 줌
st.subheader("예상과 다른 점 및 다음 분석 질문")
if len(paired) >= 2:
    paired["환자수_차이_절댓값"] = (
        paired[PERIODONTAL] - paired[DIABETES]
    ).abs()
    most_different = paired.loc[paired["환자수_차이_절댓값"].idxmax()]
    st.write(
        f"- 두 질환 환자 수 차이가 가장 크게 나타난 자료는 "
        f"**{int(most_different['연도'])}년 {most_different['연령대']}**입니다. "
        f"치주질환 {most_different[PERIODONTAL]:,.0f}명, "
        f"2형 당뇨병 {most_different[DIABETES]:,.0f}명으로 "
        f"차이는 {most_different['환자수_차이_절댓값']:,.0f}명입니다."
    )
st.markdown(
    """
    **다음 분석에서 확인할 질문**
    1. 연령대가 높아지는 효과만으로 두 질환의 상관관계가 높게 나타난 것은 아닐까?
    2. 연령대별 인구 수로 나눈 환자 비율을 비교하면 결과가 달라질까?
    3. 전체 5년을 합친 상관관계와 각 연도별 상관관계는 같은 방향일까?
    """
)

# ------------------------- 다운로드와 기록 -------------------------

st.header("6. 전처리 자료 및 분석 기록")
export_data = paired.drop(columns=["환자수_차이_절댓값"], errors="ignore")
st.download_button(
    "전처리된 분석 자료 CSV 다운로드",
    data=export_data.to_csv(index=False).encode("utf-8-sig"),
    file_name="periodontal_diabetes_cleaned.csv",
    mime="text/csv",
)

missing_summary = overview[["열 이름", "결측치 개수", "결측치 비율(%)"]]
process_log = make_process_log(
    file_name, raw, paired, steps, missing_summary, outlier_total
)
st.download_button(
    "전처리·분석 기록(log) 다운로드",
    data=process_log.encode("utf-8-sig"),
    file_name="analysis_log.txt",
    mime="text/plain",
)
with st.expander("전처리 기준 및 분석 기록 보기"):
    st.text(process_log)

st.subheader("분석 결과를 발표할 때 주의할 점")
st.markdown(
    """
    - 환자 수는 연령대별 인구 규모의 영향을 받습니다. 인구 수로 나눈 환자 비율과는 다를 수 있습니다.
    - 여러 연도와 연령대를 함께 분석하면 연령과 시간의 영향이 상관계수에 섞일 수 있습니다.
    - 건강보험 청구자료의 환자 수는 실제 전체 환자 수와 완전히 같지 않을 수 있습니다.
    - 상관관계가 높아도 치주질환이 당뇨병을 일으켰다거나 그 반대라고 결론 내릴 수 없습니다.
    - 두 질환의 환자 수를 비교하는 분석이며 개인별로 두 질환이 함께 발생했는지를 확인하는 분석은 아닙니다.
    """
)

st.caption(
    "자료 출처: 업로드한 건강보험심사평가원 자료. "
    "분석 기간 2021~2025년, 성인 기준 만 20세 이상. "
    "실제 분석 수치는 업로드한 원본 CSV에 따라 달라집니다."
)
