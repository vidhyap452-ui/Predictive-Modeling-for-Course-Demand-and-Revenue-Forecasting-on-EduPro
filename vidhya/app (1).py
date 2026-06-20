import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.generate_data import get_merged, get_monthly_revenue, get_courses, get_teachers
from models.ml_models import train_models, get_feature_importance, predict_single, FEATURE_COLS

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main { background: #0d0f1a; }
[data-testid="stSidebar"] {
    background: #111320 !important;
    border-right: 1px solid #1e2235;
}

.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #141728 100%);
    border: 1px solid #2a2d4a;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card h3 { color: #8b92c9; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 0.4rem; }
.metric-card h2 { color: #e8ecff; font-size: 1.9rem; margin: 0; font-family: 'Syne', sans-serif; }
.metric-card p  { color: #5fe6a0; font-size: 0.78rem; margin: 0.3rem 0 0; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #c8ceff;
    border-left: 4px solid #6c8bff;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

.pred-box {
    background: linear-gradient(135deg, #1a2540, #111828);
    border: 1px solid #2a3a60;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.pred-box h4 { color: #8b9ec9; font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 0.5rem; }
.pred-box h1 { color: #5fe6a0; font-size: 2.4rem; margin: 0; font-family: 'Syne', sans-serif; }
.pred-box p  { color: #6c8bff; font-size: 0.85rem; margin: 0.4rem 0 0; }

.insight-tag {
    display: inline-block;
    background: #1e2842;
    color: #6c8bff;
    border: 1px solid #2a3860;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    margin: 0.2rem;
}

stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Train Models ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all():
    df = get_merged()
    monthly = get_monthly_revenue()
    return df, monthly

@st.cache_resource(show_spinner=False)
def load_models():
    enroll_models, enroll_results, Xe, ye = train_models("EnrollmentCount")
    rev_models,    rev_results,    Xr, yr = train_models("TotalRevenue")
    return enroll_models, enroll_results, rev_models, rev_results

with st.spinner("🔄 Training predictive models…"):
    df, monthly_rev = load_all()
    enroll_models, enroll_results, rev_models, rev_results = load_models()

BEST_ENROLL = "Random Forest"
BEST_REV    = "Random Forest"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduPro")
    st.markdown("**Predictive Analytics Platform**")
    st.divider()

    st.markdown("### 🔍 Global Filters")
    sel_category = st.multiselect("Course Category",
        options=df["CourseCategory"].unique().tolist(),
        default=df["CourseCategory"].unique().tolist())
    sel_level = st.multiselect("Course Level",
        options=["Beginner", "Intermediate", "Advanced"],
        default=["Beginner", "Intermediate", "Advanced"])

    st.divider()
    st.markdown("### ⚙️ Model Selection")
    enroll_model_choice = st.selectbox("Enrollment Model", list(enroll_results.keys()), index=2)
    rev_model_choice    = st.selectbox("Revenue Model", list(rev_results.keys()), index=2)

    st.divider()
    st.markdown(
        "<small style='color:#555'>Synthetic data. Replace `data/generate_data.py` with your real CSVs.</small>",
        unsafe_allow_html=True)

# ─── Filter Data ──────────────────────────────────────────────────────────────
fdf = df[df["CourseCategory"].isin(sel_category) & df["CourseLevel"].isin(sel_level)]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 0.5rem'>
<h1 style='font-family:Syne,sans-serif;font-size:2.2rem;color:#e8ecff;margin:0'>
📊 EduPro Predictive Intelligence
</h1>
<p style='color:#6272a4;font-size:0.95rem;margin:0.3rem 0 0'>
Course Demand & Revenue Forecasting Dashboard
</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("Total Courses", f"{len(fdf):,}", "in selection"),
    ("Total Enrollments", f"{int(fdf['EnrollmentCount'].sum()):,}", "transactions"),
    ("Total Revenue", f"${fdf['TotalRevenue'].sum():,.0f}", "all time"),
    ("Avg Course Rating", f"{fdf['CourseRating'].mean():.2f} ⭐", "out of 5.0"),
    ("Avg Revenue/Course", f"${fdf['TotalRevenue'].mean():,.0f}", "per course"),
]
for col, (title, val, sub) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f"""
    <div class='metric-card'>
      <h3>{title}</h3><h2>{val}</h2><p>{sub}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Overview",
    "💰 Revenue Forecast",
    "🔮 Predict a Course",
    "🧠 Model Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMAND OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Enrollment Distribution by Category</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        cat_enroll = fdf.groupby("CourseCategory")["EnrollmentCount"].sum().reset_index().sort_values("EnrollmentCount", ascending=True)
        fig = px.bar(cat_enroll, x="EnrollmentCount", y="CourseCategory", orientation="h",
                     color="EnrollmentCount", color_continuous_scale="Blues",
                     labels={"EnrollmentCount": "Total Enrollments", "CourseCategory": ""})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#c8ceff", coloraxis_showscale=False,
                          margin=dict(l=0, r=20, t=10, b=10))
        fig.update_xaxes(gridcolor="#1e2235")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        level_enroll = fdf.groupby("CourseLevel")["EnrollmentCount"].sum().reset_index()
        fig2 = px.pie(level_enroll, names="CourseLevel", values="EnrollmentCount",
                      color_discrete_sequence=["#6c8bff","#5fe6a0","#f97bff"],
                      hole=0.55)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c8ceff",
                           margin=dict(l=0, r=0, t=20, b=0),
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig2.update_traces(textfont_color="#e8ecff")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<center style='color:#6272a4;font-size:0.8rem'>Enrollment share by level</center>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Price vs Enrollment (by Category)</div>", unsafe_allow_html=True)
    fig3 = px.scatter(fdf[fdf["EnrollmentCount"] > 0],
                      x="CoursePrice", y="EnrollmentCount",
                      color="CourseCategory", size="TotalRevenue",
                      hover_data=["CourseID", "CourseLevel", "CourseRating"],
                      opacity=0.8,
                      labels={"CoursePrice": "Course Price ($)", "EnrollmentCount": "Enrollments"})
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#c8ceff", legend=dict(bgcolor="rgba(0,0,0,0)"),
                       margin=dict(l=0, r=0, t=10, b=10))
    fig3.update_xaxes(gridcolor="#1e2235"); fig3.update_yaxes(gridcolor="#1e2235")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Top 10 Courses by Enrollment</div>", unsafe_allow_html=True)
    top10 = fdf[fdf["EnrollmentCount"] > 0].nlargest(10, "EnrollmentCount")[
        ["CourseID","CourseCategory","CourseLevel","CoursePrice","EnrollmentCount","TotalRevenue","CourseRating"]]
    top10["CoursePrice"] = top10["CoursePrice"].map("${:.2f}".format)
    top10["TotalRevenue"] = top10["TotalRevenue"].map("${:,.0f}".format)
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Monthly Revenue Trend by Category</div>", unsafe_allow_html=True)
    mr = monthly_rev[monthly_rev["CourseCategory"].isin(sel_category)].copy()
    mr["Month"] = pd.to_datetime(mr["Month"])
    fig4 = px.line(mr, x="Month", y="Amount", color="CourseCategory",
                   labels={"Amount": "Revenue ($)", "Month": ""},
                   line_shape="spline")
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#c8ceff", legend=dict(bgcolor="rgba(0,0,0,0)"),
                       margin=dict(l=0, r=0, t=10, b=10))
    fig4.update_xaxes(gridcolor="#1e2235"); fig4.update_yaxes(gridcolor="#1e2235")
    st.plotly_chart(fig4, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Revenue by Price Band</div>", unsafe_allow_html=True)
        pb = fdf.groupby("PriceBand")["TotalRevenue"].sum().reset_index()
        fig5 = px.bar(pb, x="PriceBand", y="TotalRevenue",
                      color="PriceBand",
                      color_discrete_map={"Low":"#5fe6a0","Medium":"#6c8bff","High":"#f97bff"},
                      labels={"TotalRevenue":"Revenue ($)","PriceBand":"Price Band"})
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c8ceff", showlegend=False,
                           margin=dict(l=0,r=0,t=10,b=10))
        fig5.update_yaxes(gridcolor="#1e2235")
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Revenue by Course Level</div>", unsafe_allow_html=True)
        lv = fdf.groupby("CourseLevel")["TotalRevenue"].sum().reset_index()
        fig6 = px.bar(lv, x="CourseLevel", y="TotalRevenue",
                      color="CourseLevel",
                      color_discrete_map={"Beginner":"#5fe6a0","Intermediate":"#6c8bff","Advanced":"#f97bff"},
                      labels={"TotalRevenue":"Revenue ($)","CourseLevel":""})
        fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c8ceff", showlegend=False,
                           margin=dict(l=0,r=0,t=10,b=10))
        fig6.update_yaxes(gridcolor="#1e2235")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("<div class='section-header'>Category Revenue Heatmap</div>", unsafe_allow_html=True)
    heat = fdf.groupby(["CourseCategory","CourseLevel"])["TotalRevenue"].sum().unstack(fill_value=0)
    fig7 = px.imshow(heat, color_continuous_scale="Blues", aspect="auto",
                     labels=dict(color="Revenue ($)"))
    fig7.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#c8ceff", margin=dict(l=0,r=0,t=10,b=10))
    st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT A COURSE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🔮 Enter Course Details to Get Predictions</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6272a4'>Fill in the details below and click <b>Predict</b> to forecast enrollments and revenue.</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_category  = st.selectbox("Course Category", df["CourseCategory"].unique())
            p_level     = st.selectbox("Course Level", ["Beginner", "Intermediate", "Advanced"])
            p_type      = st.selectbox("Course Type", ["Video", "Live", "Hybrid"])
        with col2:
            p_price     = st.slider("Course Price ($)", 9, 200, 49)
            p_duration  = st.slider("Duration (hours)", 2, 60, 20)
            p_rating    = st.slider("Expected Course Rating", 3.0, 5.0, 4.2, 0.1)
        with col3:
            p_teacher_rating = st.slider("Teacher Rating", 3.0, 5.0, 4.0, 0.1)
            p_experience     = st.slider("Teacher Experience (years)", 1, 20, 5)

        submitted = st.form_submit_button("🚀 Predict Demand & Revenue", use_container_width=True)

    if submitted:
        # Build price band etc.
        price_band = "Low" if p_price < 49 else ("Medium" if p_price < 99 else "High")
        dur_bucket = "Short" if p_duration <= 10 else ("Medium" if p_duration <= 30 else "Long")
        rat_tier   = "Low" if p_rating < 3.5 else ("Medium" if p_rating < 4.2 else "High")
        exp_bucket = "Junior" if p_experience <= 5 else ("Mid" if p_experience <= 10 else "Senior")

        inp = {
            "CoursePrice": p_price, "CourseDuration": p_duration,
            "CourseRating": p_rating, "TeacherRating": p_teacher_rating,
            "YearsOfExperience": p_experience,
            "CourseCategory": p_category, "CourseLevel": p_level,
            "CourseType": p_type, "PriceBand": price_band,
            "DurationBucket": dur_bucket, "RatingTier": rat_tier,
            "ExperienceBucket": exp_bucket,
        }

        pred_enroll = max(0, predict_single(inp, enroll_models[enroll_model_choice]))
        pred_rev    = max(0, predict_single(inp, rev_models[rev_model_choice]))
        pred_rev_per = pred_rev / pred_enroll if pred_enroll > 0 else 0

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.markdown(f"""
        <div class='pred-box'>
          <h4>Predicted Enrollments</h4>
          <h1>{pred_enroll:.0f}</h1>
          <p>students expected</p>
        </div>""", unsafe_allow_html=True)
        r2.markdown(f"""
        <div class='pred-box'>
          <h4>Predicted Revenue</h4>
          <h1>${pred_rev:,.0f}</h1>
          <p>total course revenue</p>
        </div>""", unsafe_allow_html=True)
        r3.markdown(f"""
        <div class='pred-box'>
          <h4>Revenue per Enrollment</h4>
          <h1>${pred_rev_per:.2f}</h1>
          <p>avg per student</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Compare to category average
        cat_avg_e = fdf[fdf["CourseCategory"] == p_category]["EnrollmentCount"].mean()
        cat_avg_r = fdf[fdf["CourseCategory"] == p_category]["TotalRevenue"].mean()
        diff_e = ((pred_enroll - cat_avg_e) / cat_avg_e * 100) if cat_avg_e > 0 else 0
        diff_r = ((pred_rev - cat_avg_r) / cat_avg_r * 100) if cat_avg_r > 0 else 0

        st.info(f"""
        **📊 Category Benchmark ({p_category}):**
        Average enrollment = **{cat_avg_e:.0f}** | Your prediction is **{diff_e:+.1f}%** vs average.
        Average revenue = **${cat_avg_r:,.0f}** | Your prediction is **{diff_r:+.1f}%** vs average.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Enrollment Prediction Models**")
        enroll_df = pd.DataFrame(enroll_results).T.reset_index().rename(columns={"index":"Model"})
        fig8 = px.bar(enroll_df.melt(id_vars="Model", value_vars=["MAE","RMSE","R2"]),
                      x="Model", y="value", color="variable", barmode="group",
                      color_discrete_sequence=["#6c8bff","#5fe6a0","#f97bff"],
                      labels={"value":"Score","variable":"Metric"})
        fig8.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c8ceff", legend=dict(bgcolor="rgba(0,0,0,0)"),
                           xaxis_tickangle=-20, margin=dict(l=0,r=0,t=10,b=0))
        fig8.update_yaxes(gridcolor="#1e2235")
        st.plotly_chart(fig8, use_container_width=True)
        st.dataframe(enroll_df.set_index("Model"), use_container_width=True)

    with col2:
        st.markdown("**Revenue Prediction Models**")
        rev_df = pd.DataFrame(rev_results).T.reset_index().rename(columns={"index":"Model"})
        fig9 = px.bar(rev_df.melt(id_vars="Model", value_vars=["MAE","RMSE","R2"]),
                      x="Model", y="value", color="variable", barmode="group",
                      color_discrete_sequence=["#6c8bff","#5fe6a0","#f97bff"],
                      labels={"value":"Score","variable":"Metric"})
        fig9.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c8ceff", legend=dict(bgcolor="rgba(0,0,0,0)"),
                           xaxis_tickangle=-20, margin=dict(l=0,r=0,t=10,b=0))
        fig9.update_yaxes(gridcolor="#1e2235")
        st.plotly_chart(fig9, use_container_width=True)
        st.dataframe(rev_df.set_index("Model"), use_container_width=True)

    st.markdown("<div class='section-header'>Feature Importance — What Drives Demand?</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Enrollment drivers**")
        fi_e = get_feature_importance(enroll_models[enroll_model_choice])
        fig10 = px.bar(fi_e, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale="Blues")
        fig10.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#c8ceff", coloraxis_showscale=False,
                            margin=dict(l=0,r=0,t=10,b=10))
        fig10.update_xaxes(gridcolor="#1e2235")
        st.plotly_chart(fig10, use_container_width=True)

    with c2:
        st.markdown("**Revenue drivers**")
        fi_r = get_feature_importance(rev_models[rev_model_choice])
        fig11 = px.bar(fi_r, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale="Greens")
        fig11.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#c8ceff", coloraxis_showscale=False,
                            margin=dict(l=0,r=0,t=10,b=10))
        fig11.update_xaxes(gridcolor="#1e2235")
        st.plotly_chart(fig11, use_container_width=True)

    st.markdown("<div class='section-header'>📌 Business Insights</div>", unsafe_allow_html=True)
    top_feat_e = fi_e.iloc[0]["Feature"] if not fi_e.empty else "CoursePrice"
    top_feat_r = fi_r.iloc[0]["Feature"] if not fi_r.empty else "CoursePrice"
    best_cat = fdf.groupby("CourseCategory")["TotalRevenue"].sum().idxmax()
    best_level = fdf.groupby("CourseLevel")["EnrollmentCount"].sum().idxmax()

    insights = [
        f"🏆 <b>{best_cat}</b> is the highest revenue category",
        f"📚 <b>{best_level}</b> courses attract the most enrollments",
        f"🔑 <b>{top_feat_e}</b> is the #1 driver of enrollment demand",
        f"💵 <b>{top_feat_r}</b> is the #1 driver of course revenue",
        "📈 Higher teacher ratings correlate with better enrollment performance",
        "💡 Medium-priced courses ($49–$99) often balance volume and revenue",
    ]
    for ins in insights:
        st.markdown(f"<div style='background:#1a1d2e;border:1px solid #2a2d4a;border-radius:10px;padding:0.8rem 1.2rem;margin:0.4rem 0;color:#c8ceff;font-size:0.9rem'>{ins}</div>", unsafe_allow_html=True)
