
# 추가 그래프: 치주질환과 제2형 당뇨병 환자 수의 상관관계 산점도

st.subheader("연령대별 두 질환 환자 수의 상관관계")

fig2, ax2 = plt.subplots(figsize=(10, 6))

# 산점도
ax2.scatter(
    merged["치주질환_K05_환자수"],
    merged["제2형당뇨병_E11_환자수"],
    s=80,
    alpha=0.7,
    label="연령대별 환자 수"
)

# 각 점에 연령대 표시
for _, row in merged.iterrows():
    ax2.annotate(
        row["연령군"],
        (
            row["치주질환_K05_환자수"],
            row["제2형당뇨병_E11_환자수"]
        ),
        fontsize=8,
        xytext=(5, 5),
        textcoords="offset points"
    )

# 선형 추세선
x_values = merged["치주질환_K05_환자수"].to_numpy()
y_values = merged["제2형당뇨병_E11_환자수"].to_numpy()

if len(merged) >= 2 and np.ptp(x_values) > 0:
    slope, intercept = np.polyfit(x_values, y_values, 1)

    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    line_y = slope * line_x + intercept

    ax2.plot(
        line_x,
        line_y,
        linestyle="--",
        label="선형 추세선"
    )

ax2.set_title(
    "치주질환과 제2형 당뇨병 환자 수의 상관관계",
    fontsize=14,
    fontweight="bold"
)
ax2.set_xlabel("치주질환(K05) 환자 수(명)")
ax2.set_ylabel("제2형 당뇨병(E11) 환자 수(명)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)
