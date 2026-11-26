import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, spearmanr


# ==============================
# 기본 설정
# ==============================

st.set_page_config(
    page_title="잇몸병과 당뇨병의 상관관계",
    page_icon="🦷",
    layout="wide"
)

st.title("🦷 치주질환과 2형 당뇨병의 상관관계")

st.markdown("""
### 탐구 질문

**2025년 우리나라 성인의 연령대별 치주질환 환자 수와
2형 당뇨병 환자 수 사이에는 어떤 상관관계가 있을까?**

건강보험심사평가원의 통계자료를 활용하여
연령대별 환자 수를 비교하고 상관관계를 분석합니다.
""")

PERIODONTAL = "치주질환(K05)"
DIABETES = "2형 당뇨병(E11)"

DATA_DIR = Path(__file__).resolve().parent / "data"


# ==============================
# CSV 불러오기
# ==============================

@st.cache_data
def load_csv(file_bytes):

    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:

        try:

            return pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding,
                low_memory=False
            )

        except Exception:
            continue

    raise ValueError("CSV 파일을 읽을 수 없습니다.")


# ==============================
# 연령 처리
# ==============================

def extract_age(value):

    text = str(value).strip()

    match = re.search(r"(\d+)", text)

    if match:
        return int(match.group(1))

    return None


# ==============================
# 데이터 전처리
# ==============================

def prepare_data(df):

    required_columns = [
        "진료년도",
        "주상병코드",
        "성별",
        "연령군",
        "환자수"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"필수 열이 없습니다: {column}"
            )

    data = df.copy()

    # 2025년 데이터만 추출
    data = data[
        data["진료년도"].astype(str).str.contains("2025")
    ].copy()

    # 코드 문자열 정리
    data["주상병코드"] = (
        data["주상병코드"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 연령군에서 숫자 추출
    data["연령시작"] = data["연령군"].apply(
        extract_age
    )

    # 환자수 숫자 변환
    data["환자수"] = pd.to_numeric(
        data["환자수"],
        errors="coerce"
    )

    # 성인 20세 이상만 추출
    data = data[
        data["연령시작"] >= 20
    ].copy()

    # 질환 코드 구분
    data["질환"] = np.select(
        [
            data["주상병코드"].str.startswith("K05"),
            data["주상병코드"].str.startswith("E11")
        ],
        [
            PERIODONTAL,
            DIABETES
        ],
        default=""
    )

    # 두 질환만 선택
    data = data[
        data["질환"] != ""
    ].copy()

    # 남녀 환자 수 합산
    grouped = (
        data.groupby(
            ["연령군", "연령시작", "질환"],
            as_index=False
        )["환자수"]
        .sum()
    )

    # 질환별로 열 분리
    paired = grouped.pivot_table(
        index=["연령군", "연령시작"],
        columns="질환",
        values="환자수",
        aggfunc="sum"
    ).reset_index()

    paired.columns.name = None

    # 두 질환의 열이 존재하는지 확인
    for disease in [PERIODONTAL, DIABETES]:

        if disease not in paired.columns:

            paired[disease] = np.nan

    # 두 질환의 값이 모두 있는 연령대만 분석
    paired = paired.dropna(
        subset=[PERIODONTAL, DIABETES]
    )

    # 연령순 정렬
    paired = paired.sort_values(
        "연령시작"
    ).reset_index(drop=True)

    return paired, grouped


# ==============================
# 상관계수 계산
# ==============================

def calculate_correlation(data, method):

    clean = data[
        [PERIODONTAL, DIABETES]
    ].dropna()

    if len(clean) < 3:

        return None

    if (
        clean[PERIODONTAL].nunique() < 2
        or clean[DIABETES].nunique() < 2
    ):

        return None

    if method == "Pearson":

        r, p = pearsonr(
            clean[PERIODONTAL],
            clean[DIABETES]
        )

    else:

        r, p = spearmanr(
            clean[PERIODONTAL],
            clean[DIABETES]
        )

    return {
        "r": r,
        "p": p,
        "n": len(clean)
    }


# ==============================
# 데이터 파일 선택
# ==============================

st.sidebar.header("데이터 불러오기")

uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 업로드",
    type=["csv"]
)

csv_files = (
    list(DATA_DIR.glob("*.csv"))
    if DATA_DIR.exists()
    else []
)


if uploaded_file is not None:

    raw_data = load_csv(
        uploaded_file.getvalue()
    )

    file_name = uploaded_file.name

elif csv_files:

    selected_file = st.sidebar.selectbox(
        "data 폴더의 CSV 선택",
        csv_files,
        format_func=lambda x: x.name
    )

    raw_data = load_csv(
        selected_file.read_bytes()
    )

    file_name = selected_file.name

else:

    st.warning(
        "CSV 파일을 업로드하거나 "
        "GitHub의 data 폴더에 파일을 넣어주세요."
    )

    st.stop()


st.caption(
    f"사용 파일: {file_name}"
)


# ==============================
# 전처리 실행
# ==============================

try:

    paired, grouped = prepare_data(
        raw_data
    )

except Exception as error:

    st.error(
        f"데이터 처리 오류: {error}"
    )

    st.stop()


if paired.empty:

    st.error(
        "분석할 데이터가 없습니다. "
        "2025년 K05와 E11 코드 및 "
        "20세 이상 연령대 자료를 확인해주세요."
    )

    st.stop()


st.success(
    f"전처리 완료! {len(paired)}개 연령대 분석"
)


# ==============================
# 요약
# ==============================

col1, col2, col3 = st.columns(3)

col1.metric(
    "분석 연령대 수",
    f"{len(paired)}개"
)

col2.metric(
    "분석 연도",
    "2025년"
)

col3.metric(
    "분석 대상",
    "20세 이상"
)


# ==============================
# 연령대별 환자 수 비교
# ==============================

st.subheader("1. 연령대별 환자 수 비교")


chart_data = paired.melt(
    id_vars=["연령군", "연령시작"],
    value_vars=[PERIODONTAL, DIABETES],
    var_name="질환",
    value_name="환자 수"
)


fig = px.bar(
    chart_data,
    x="연령군",
    y="환자 수",
    color="질환",
    barmode="group",
    title="2025년 연령대별 치주질환과 2형 당뇨병 환자 수",
    labels={
        "연령군": "연령대",
        "환자 수": "환자 수(명)",
        "질환": "질환"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==============================
# 산점도
# ==============================

st.subheader("2. 두 질환의 환자 수 관계")


scatter_fig = px.scatter(
    paired,
    x=PERIODONTAL,
    y=DIABETES,
    hover_data=["연령군"],
    title="연령대별 치주질환과 2형 당뇨병 환자 수 관계",
    labels={
        PERIODONTAL: "치주질환 환자 수(명)",
        DIABETES: "2형 당뇨병 환자 수(명)"
    }
)

st.plotly_chart(
    scatter_fig,
    use_container_width=True
)


# ==============================
# 상관계수 분석
# ==============================

st.subheader("3. 상관계수 분석")


pearson_result = calculate_correlation(
    paired,
    "Pearson"
)

spearman_result = calculate_correlation(
    paired,
    "Spearman"
)


col1, col2 = st.columns(2)


if pearson_result is not None:

    with col1:

        st.metric(
            "피어슨 상관계수",
            f"{pearson_result['r']:.3f}"
        )

        st.write(
            f"p-value: {pearson_result['p']:.4g}"
        )


if spearman_result is not None:

    with col2:

        st.metric(
            "스피어만 상관계수",
            f"{spearman_result['r']:.3f}"
        )

        st.write(
            f"p-value: {spearman_result['p']:.4g}"
        )


st.markdown("""
- **피어슨 상관계수:** 두 변수 사이의 선형 관계를 측정합니다.
- **스피어만 상관계수:** 두 변수의 순위에 따른 관계를 측정합니다.
- **p-value:** 관찰된 관계가 통계적으로 유의한지 판단할 때 참고합니다.

상관관계가 높다고 해서 한 질병이 다른 질병을
직접 유발한다고 단정할 수는 없습니다.
""")


# ==============================
# 분석 자료 확인
# ==============================

st.subheader("4. 전처리된 분석 데이터")


st.dataframe(
    paired,
    use_container_width=True,
    hide_index=True
)


# ==============================
# 다운로드
# ==============================

csv_download = paired.to_csv(
    index=False
).encode("utf-8-sig")


st.download_button(
    "분석 데이터 CSV 다운로드",
    data=csv_download,
    file_name="2025_치주질환_당뇨병_분석자료.csv",
    mime="text/csv"
)


# ==============================
# 한계
# ==============================

st.subheader("5. 분석의 한계")


st.markdown("""
1. 2025년 단일 연도의 자료이므로 최근 5년간의 증가 추세는 분석하지 못합니다.
2. 연령대별 환자 수만 비교하므로 인구 규모의 차이가 결과에 영향을 줄 수 있습니다.
3. 환자 수가 많다는 사실이 해당 연령대의 질병 발생 위험이 높다는 의미는 아닙니다.
4. 건강보험 청구자료는 의료기관을 이용한 기록을 바탕으로 집계되므로 실제 질병 유병 규모와 차이가 있을 수 있습니다.
5. 연령대별 환자 수의 상관관계는 연령이라는 공통 요인의 영향을 받을 수 있습니다.
6. 본 분석은 상관관계를 탐구하며 인과관계를 증명하는 연구가 아닙니다.
""")


st.caption(
    "자료 출처: 건강보험심사평가원 "
    "3단상병별 성별 연령군별 통계(2025년)"
)
