years_found = sorted(paired["연도"].dropna().astype(int).unique().tolist())
st.success(f"전처리 완료: {len(paired):,}개의 연도-연령군 짝을 찾았습니다.")
if years_found != YEARS:
    st.warning(f"자료에서 확인된 연도는 {years_found}입니다. 2021~2025년 전체가 포함되어 있는지 확인해 주세요.")

with st.expander("원본 데이터에서 자동으로 찾은 열 확인"):
    labels = ["진료년도", "주상병코드", "성별", "연령군", "환자수"]
    for label, col in zip(labels, detected):
        st.write(f"- {label}: {col if col is not None else '해당 열 없음'}")

# ---------- 요약 ----------
periodontal = "치주질환(K05)"
diabetes = "2형 당뇨병(E11)"
total_rows = len(paired)
c1, c2, c3 = st.columns(3)
c1.metric("분석 연도", f"{len(years_found)}개")
c2.metric("분석 연령군-연도 짝", f"{total_rows:,}개")
c3.metric("분석 연령 범위", f"{int(paired['연령시작'].min())}세 이상")

st.subheader("1. 연령대별 환자 수 비교")
selected_year = st.selectbox("연도 선택", years_found, index=len(years_found)-1)
year_data = paired[paired["연도"] == selected_year].sort_values("연령시작")
if not year_data.empty:
    long_year = year_data.melt(
        id_vars=["연령군", "연령시작"],
        value_vars=[periodontal, diabetes],
        var_name="질환", value_name="환자수"
    )
    fig_bar = px.bar(
        long_year, x="연령군", y="환자수", color="질환", barmode="group",
        title=f"{selected_year}년 연령대별 치주질환·2형 당뇨병 환자 수",
        labels={"연령군": "연령대", "환자수": "환자 수(명)", "질환": "질환"}
    )
    fig_bar.update_layout(xaxis_title="연령대", yaxis_title="환자 수(명)", legend_title="질환")
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("2. 두 질환 환자 수의 산점도")
plot_data = paired.copy()
fig_scatter = px.scatter(
    plot_data, x=periodontal, y=diabetes, color=plot_data["연도"].astype(str),
    hover_data=["연도", "연령군"], trendline="ols",
    labels={
        periodontal: "치주질환 환자 수(명)",
        diabetes: "2형 당뇨병 환자 수(명)",
        "color": "진료년도"
    },
    title="연도·연령군별 치주질환 환자 수와 2형 당뇨병 환자 수"
)
fig_scatter.update_layout(xaxis_title="치주질환 환자 수(명)", yaxis_title="2형 당뇨병 환자 수(명)")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("3. 상관계수 분석")
method_choice = st.radio("상관계수 표시", ["Pearson", "Spearman", "둘 다"], horizontal=True)
pooled_results = {}
if method_choice in ("Pearson", "둘 다"):
    pooled_results["Pearson"] = corr_result(paired, periodontal, diabetes, "Pearson")
if method_choice in ("Spearman", "둘 다"):
    pooled_results["Spearman"] = corr_result(paired, periodontal, diabetes, "Spearman")

cols = st.columns(max(1, len(pooled_results)))
for col, (method, result) in zip(cols, pooled_results.items()):
    with col:
        if result is None:
            st.warning(f"{method}: 계산에 필요한 자료가 부족합니다.")
        else:
            st.metric(f"{method} 상관계수", f"{result['r']:.3f}")
            st.write(f"p-value: **{result['p']:.4g}**")
            st.write(f"표본 수: **{result['n']}개**")

st.caption("상관계수 0은 선형 관계가 없음을 기준으로 삼는 값입니다. p-value는 통계적 유의성을 살펴보는 참고값이며, 인과관계를 증명하지 않습니다.")

st.subheader("4. 연도별 상관관계")
annual_rows = []
for year in years_found:
    subset = paired[paired["연도"] == year]
    for method in ("Pearson", "Spearman"):
        result = corr_result(subset, periodontal, diabetes, method)
        if result is not None:
            annual_rows.append({"연도": year, "방법": method, "상관계수": result["r"], "p-value": result["p"], "표본 수": result["n"]})
annual_df = pd.DataFrame(annual_rows)
if not annual_df.empty:
    st.dataframe(annual_df.round({"상관계수": 3, "p-value": 4}), use_container_width=True, hide_index=True)
    fig_annual = px.line(
        annual_df, x="연도", y="상관계수", color="방법", markers=True,
        title="연도별 연령군 간 상관계수 변화",
        labels={"연도": "진료년도", "상관계수": "상관계수", "방법": "분석 방법"}
    )
    fig_annual.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_annual.update_yaxes(range=[-1.05, 1.05])
    st.plotly_chart(fig_annual, use_container_width=True)
else:
    st.info("연도별 상관계수를 계산하기에 연령군 자료가 충분하지 않습니다.")

st.subheader("5. 전처리된 분석 자료")
st.dataframe(
    paired.rename(columns={
        periodontal: "치주질환 환자 수",
        diabetes: "2형 당뇨병 환자 수"
    }),
    use_container_width=True,
    hide_index=True
)
csv_out = paired.to_csv(index=False).encode("utf-8-sig")
st.download_button("전처리 결과 CSV 다운로드", csv_out, "periodontal_diabetes_paired.csv", "text/csv")

st.subheader("해석할 때 주의할 점")
st.markdown(
    """
    - 이 분석은 **환자 수의 상관관계**를 확인하며, 치주질환이 당뇨병을 일으킨다거나 그 반대라고 증명하지 않습니다.
    - 연령대별 인구 규모가 다르므로 환자 수가 많다는 사실만으로 해당 연령대의 질병 위험이 더 높다고 단정할 수 없습니다.
    - 인구 대비 발생 비율 또는 유병률을 분석하려면 연령대별 인구 자료가 추가로 필요합니다.
    - 건강보험 청구자료의 환자 수는 실제 전체 환자 수나 질병의 중증도를 완벽하게 나타내지 않을 수 있습니다.
    - 전체 연도·연령군을 한꺼번에 계산한 상관계수는 연령과 시간의 영향을 받을 수 있으므로, 연도별 결과와 함께 해석해야 합니다.
    """
)

st.caption("자료 출처: 건강보험심사평가원, 공공데이터포털의 3단상병별 성별 연령군별 통계. 분석 기간은 2021~2025년으로 설정되어 있습니다.")
