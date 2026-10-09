import os
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()
st.set_page_config(page_title="HR Attrition Analytics", layout="wide", page_icon="🏢")

# ---- Custom styling: bigger metric font ----
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1e2530;
    border: 1px solid #2d3648;
    border-radius: 10px;
    padding: 15px;
}
div[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {"No": "#3B82F6", "Yes": "#EF4444"}

with open("models/attrition_model.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
encoders = saved["encoders"]
columns = saved["columns"]

@st.cache_data(ttl=600)
def load_data():
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )
    df = pd.read_sql("SELECT * FROM hr_attrition_final;", conn)
    conn.close()
    return df

df = load_data()

st.title("🏢 HR Attrition Analytics Dashboard")

tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predict Attrition"])

with tab1:
    st.header("Attrition Overview")

    # ---- Filters row ----
    f1, f2 = st.columns([2, 2])
    with f1:
        dept_filter = st.multiselect("Filter by Department", options=sorted(df["DEPARTMENT"].unique()),
                                      default=list(df["DEPARTMENT"].unique()))
    with f2:
        min_yrs, max_yrs = int(df["YEARS_AT_COMPANY"].min()), int(df["YEARS_AT_COMPANY"].max())
        yrs_range = st.slider("Tenure Range (Years at Company)", min_yrs, max_yrs, (min_yrs, max_yrs))

    fdf = df[
        df["DEPARTMENT"].isin(dept_filter)
        & df["YEARS_AT_COMPANY"].between(yrs_range[0], yrs_range[1])
    ]

    # ---- Cross-filter: click a bar in the Department chart to drill down ----
    st.caption("💡 Click a bar in the Department chart below to cross-filter all other charts.")

    total = len(fdf)
    attr_yes = (fdf['ATTRITION'] == 'Yes').sum()
    rate = attr_yes / total * 100 if total else 0
    avg_income = fdf['MONTHLY_INCOME'].mean() if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Employees Left", f"{attr_yes:,}")
    col3.metric("Attrition Rate", f"{rate:.1f}%")
    col4.metric("Avg Monthly Income", f"₹{avg_income:,.0f}")

    st.divider()

    # ---- Department chart with click-selection (drives cross-filter) ----
    c1, c2 = st.columns([2, 1])
    with c1:
        dept_counts = fdf.groupby(["DEPARTMENT", "ATTRITION"]).size().reset_index(name="count")
        fig1 = px.bar(dept_counts, x="DEPARTMENT", y="count", color="ATTRITION",
                      barmode="group", text="count", color_discrete_map=COLOR_MAP,
                      title="Attrition by Department (click a bar to filter)")
        fig1.update_traces(textposition="outside")
        selection = st.plotly_chart(fig1, use_container_width=True,
                                     on_select="rerun", selection_mode="points",
                                     key="dept_chart")
    with c2:
        pie_data = fdf["ATTRITION"].value_counts().reset_index()
        pie_data.columns = ["ATTRITION", "count"]
        fig_pie = px.pie(pie_data, names="ATTRITION", values="count", hole=0.5,
                          color="ATTRITION", color_discrete_map=COLOR_MAP,
                          title="Overall Attrition Split")
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- Apply cross-filter from department chart click ----
    clicked_depts = []
    if selection and selection.get("selection", {}).get("points"):
        clicked_depts = [p["x"] for p in selection["selection"]["points"]]
    if clicked_depts:
        fdf = fdf[fdf["DEPARTMENT"].isin(clicked_depts)]
        st.info(f"🔍 Cross-filtered to: {', '.join(clicked_depts)} — click empty space or rerun to reset.")

    # ---- Row 2: Overtime + Age distribution ----
    c3, c4 = st.columns(2)
    with c3:
        ot_counts = fdf.groupby(["OVERTIME", "ATTRITION"]).size().reset_index(name="count")
        fig2 = px.bar(ot_counts, x="OVERTIME", y="count", color="ATTRITION", text="count",
                      barmode="group", color_discrete_map=COLOR_MAP, title="Attrition by OverTime")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    with c4:
        fig_age = px.histogram(fdf, x="AGE", color="ATTRITION", nbins=20, marginal="box",
                                color_discrete_map=COLOR_MAP, title="Age Distribution by Attrition")
        st.plotly_chart(fig_age, use_container_width=True)

    # ---- Row 3: Income + Satisfaction ----
    c5, c6 = st.columns(2)
    with c5:
        fig3 = px.violin(fdf, x="ATTRITION", y="MONTHLY_INCOME", color="ATTRITION", box=True,
                          points="all", color_discrete_map=COLOR_MAP, title="Monthly Income vs Attrition")
        st.plotly_chart(fig3, use_container_width=True)
    with c6:
        sat_counts = fdf.groupby(["JOB_SATISFACTION", "ATTRITION"]).size().reset_index(name="count")
        fig4 = px.bar(sat_counts, x="JOB_SATISFACTION", y="count", color="ATTRITION", text="count",
                      barmode="group", color_discrete_map=COLOR_MAP, title="Job Satisfaction vs Attrition")
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    # ---- Row 4: Feature importance ----
    st.subheader("🔑 Top Factors Driving Attrition (Model-based)")
    importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False).head(8)
    fig_imp = px.bar(importance[::-1], orientation="h", text=importance[::-1].round(3),
                      labels={"value": "Importance", "index": "Feature"},
                      color=importance[::-1], color_continuous_scale="Blues")
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.header("Predict Employee Attrition Risk")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 60, 30)
        department = st.selectbox("Department", encoders['DEPARTMENT'].classes_)
        job_role = st.selectbox("Job Role", encoders['JOB_ROLE'].classes_)
        gender = st.selectbox("Gender", encoders['GENDER'].classes_)
        marital_status = st.selectbox("Marital Status", encoders['MARITAL_STATUS'].classes_)
        performance = st.slider("Performance Rating", 1, 4, 3)
    with col2:
        overtime = st.selectbox("OverTime", encoders['OVERTIME'].classes_)
        distance = st.number_input("Distance From Home (km)", 1, 30, 10)
        income = st.number_input("Monthly Income", 20000, 100000, 50000)
        years = st.number_input("Years At Company", 0, 20, 3)
        satisfaction = st.slider("Job Satisfaction", 1, 4, 3)

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([{
            "AGE": age,
            "DEPARTMENT": encoders['DEPARTMENT'].transform([department])[0],
            "JOB_ROLE": encoders['JOB_ROLE'].transform([job_role])[0],
            "GENDER": encoders['GENDER'].transform([gender])[0],
            "MARITAL_STATUS": encoders['MARITAL_STATUS'].transform([marital_status])[0],
            "PERFORMANCE_RATING": performance,
            "OVERTIME": encoders['OVERTIME'].transform([overtime])[0],
            "DISTANCE_FROM_HOME": distance,
            "MONTHLY_INCOME": income,
            "YEARS_AT_COMPANY": years,
            "JOB_SATISFACTION": satisfaction,
        }])[columns]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.progress(int(prob * 100))
        if pred == 1:
            st.error(f"⚠️ High Attrition Risk — {prob*100:.1f}% probability")
        else:
            st.success(f"✅ Low Attrition Risk — {prob*100:.1f}% probability")