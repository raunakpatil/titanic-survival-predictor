import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import os
from model import train_model, predict_survival, run_survival_simulator, build_passenger_features
from data_utils import load_data, engineer_features, get_historical_twin, search_passengers

st.set_page_config(
    page_title="Titanic Survival Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme & Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Base */
  .stApp { background-color: #07090f; }
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Hide default header padding */
  .block-container { padding-top: 0 !important; }

  /* ── Hero Banner ── */
  .hero {
    background: linear-gradient(160deg, #0d1420 0%, #07090f 50%, #0d1117 100%);
    border-bottom: 1px solid #1a2235;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(30,60,120,0.18) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-ship {
    position: absolute; right: 2rem; bottom: 0;
    opacity: 0.07; font-size: 11rem; line-height: 1;
    user-select: none; pointer-events: none;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 800;
    color: #e8d5b7; letter-spacing: -1px;
    margin: 0 0 0.25rem; line-height: 1.1;
  }
  .hero-sub {
    font-size: 0.95rem; color: #5a7a9a; font-weight: 400;
    margin: 0 0 1.5rem; letter-spacing: 0.03em;
  }
  .hero-stats {
    display: flex; gap: 2rem; flex-wrap: wrap;
  }
  .hstat { text-align: left; }
  .hstat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 700;
    line-height: 1;
  }
  .hstat-num.green { color: #4ade80; }
  .hstat-num.red   { color: #f87171; }
  .hstat-num.blue  { color: #60a5fa; }
  .hstat-num.amber { color: #fbbf24; }
  .hstat-lbl { font-size: 0.72rem; color: #4a6070; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

  /* ── Metric Cards ── */
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.5rem; }
  .mcard {
    background: #0e1320; border-radius: 12px;
    padding: 1rem 1.25rem;
    border-left: 3px solid transparent;
  }
  .mcard.green  { border-color: #166534; }
  .mcard.red    { border-color: #7f1d1d; }
  .mcard.blue   { border-color: #1e3a5f; }
  .mcard.amber  { border-color: #78350f; }
  .mcard-num { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: #e8d5b7; }
  .mcard-lbl { font-size: 0.72rem; color: #4a6070; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

  /* ── Verdict Badge ── */
  .survived-badge {
    display: inline-block;
    background: #052e16; color: #4ade80;
    border: 1px solid #166534; border-radius: 8px;
    padding: 0.45rem 1.2rem; font-weight: 600; font-size: 1rem;
    animation: fadein 0.4s ease;
  }
  .perished-badge {
    display: inline-block;
    background: #1c0505; color: #f87171;
    border: 1px solid #7f1d1d; border-radius: 8px;
    padding: 0.45rem 1.2rem; font-weight: 600; font-size: 1rem;
    animation: fadein 0.4s ease;
  }
  @keyframes fadein { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform: none; } }

  /* ── Waterfall SHAP chart label ── */
  .shap-label { font-size: 0.8rem; color: #6b7280; }

  /* ── Section header ── */
  .section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700; color: #e8d5b7;
    border-bottom: 1px solid #1a2235; padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* ── Passenger card (search result / twin) ── */
  .pcard {
    background: #0e1320; border: 1px solid #1a2235;
    border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.75rem;
  }
  .pcard-name { font-family: 'Playfair Display', serif; font-size: 1.1rem; color: #e8d5b7; }
  .pcard-sub  { font-size: 0.8rem; color: #5a7a9a; margin-top: 2px; }
  .pcard-fate-survived { color: #4ade80; font-weight: 600; }
  .pcard-fate-perished { color: #f87171; font-weight: 600; }

  /* ── Comparison panel ── */
  .compare-badge {
    font-size: 0.7rem; background: #1a2235; color: #60a5fa;
    border-radius: 4px; padding: 2px 6px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
  }

  /* Streamlit widget overrides */
  div[data-testid="stMetricValue"] { color: #e8d5b7 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #0e1320; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #5a7a9a !important; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: #1a2235 !important; color: #e8d5b7 !important; }
  div[data-testid="stSidebar"] { background: #0a0d16; border-right: 1px solid #1a2235; }

  /* Loading spinner message */
  .loading-msg {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; color: #5a7a9a; text-align: center; padding: 2rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model_and_data():
    train_df, test_df = load_data()
    X_train, y_train, X_test, feature_names = engineer_features(train_df, test_df)
    model, explainer, cv_mean, cv_std = train_model(X_train, y_train)
    return model, explainer, train_df, X_train, y_train, feature_names, cv_mean, cv_std


with st.spinner("Loading passenger manifest & training model…"):
    model, explainer, train_df, X_train, y_train, feature_names, cv_mean, cv_std = get_model_and_data()


# ── Hero Banner ───────────────────────────────────────────────────────────────
n_total = len(train_df)
n_survived = int(train_df["Survived"].sum())
n_perished = n_total - n_survived
avg_age = train_df["Age"].mean()

st.markdown(f"""
<div class="hero">
  <div class="hero-ship">🚢</div>
  <p class="hero-title">Titanic Survival Explorer</p>
  <p class="hero-sub">Predicting Titanic Survival with XGBoost + SHAP &nbsp;·&nbsp; Made with ❤️ by <a href="https://github.com/raunakpatil" target="_blank" style="color:#60a5fa;text-decoration:none;font-weight:500;">@raunakpatil</a></p>
  <div class="hero-stats">
    <div class="hstat">
      <div class="hstat-num blue">{n_total:,}</div>
      <div class="hstat-lbl">Passengers</div>
    </div>
    <div class="hstat">
      <div class="hstat-num green">{n_survived}</div>
      <div class="hstat-lbl">Survived</div>
    </div>
    <div class="hstat">
      <div class="hstat-num red">{n_perished}</div>
      <div class="hstat-lbl">Perished</div>
    </div>
    <div class="hstat">
      <div class="hstat-num amber">{cv_mean:.1%} <span style="font-size:1rem;color:#78350f">±{cv_std:.1%}</span></div>
      <div class="hstat-lbl">Model accuracy (5-fold CV)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Would You Survive?",
    "🔄 Compare Profiles",
    "🔎 Passenger Search",
    "📊 Data Story",
    "🧠 Model Insights"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Survival Predictor
# ═══════════════════════════════════════════════════════════════════════════════
def shap_waterfall_fig(shap_dict, base_prob):
    labels = list(shap_dict.keys())
    values = list(shap_dict.values())
    colors = ["#4ade80" if v > 0 else "#f87171" for v in values]

    # Compute running total from base
    base = base_prob - sum(values)
    running = base
    measures = []
    for v in values:
        measures.append("relative")
        running += v

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + measures,
        x=[base] + values,
        y=["Base rate"] + labels,
        connector={"line": {"color": "#1a2235", "width": 1}},
        increasing={"marker": {"color": "#4ade80"}},
        decreasing={"marker": {"color": "#f87171"}},
        totals={"marker": {"color": "#60a5fa"}},
        text=[""] + [f"{'+'if v>0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#9ca3af", "size": 11},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(t=10, b=10, l=10, r=70),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#1a2235",
                   tickfont={"color": "#6b7280"}, range=[0, 1]),
        yaxis=dict(showgrid=False, tickfont={"color": "#9ca3af"}),
        font_color="#e8d5b7",
        showlegend=False,
    )
    return fig


with tab1:
    st.markdown('<div class="section-header">Build your 1912 passenger profile</div>', unsafe_allow_html=True)
    st.caption("Adjust inputs to see your survival probability and what drove it.")

    c1, c2, c3 = st.columns(3)
    with c1:
        sex = st.selectbox("Gender", ["female", "male"])
        pclass = st.selectbox("Ticket class", [1, 2, 3],
                              format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} class")
        embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                                format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
    with c2:
        age = st.slider("Age", 1, 80, 28)
        fare = st.slider("Fare paid (£)", 5, 300, 30, step=5)
    with c3:
        sibsp = st.slider("Siblings / spouses aboard", 0, 5, 0)
        parch = st.slider("Parents / children aboard", 0, 5, 0)

    st.markdown("---")

    prob, shap_vals, input_df = predict_survival(
        model, explainer, sex, pclass, age, fare, sibsp, parch, embarked, feature_names
    )

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("#### Survival probability")

        # Animated gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"color": "#e8d5b7", "size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#374151", "tickfont": {"color": "#4a6070"}},
                "bar": {"color": "#4ade80" if prob > 0.5 else "#f87171", "thickness": 0.25},
                "bgcolor": "#0e1320",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],  "color": "#1c0505"},
                    {"range": [35, 65], "color": "#0e1320"},
                    {"range": [65, 100],"color": "#052e16"},
                ],
                "threshold": {
                    "line": {"color": "#e8d5b7", "width": 2},
                    "thickness": 0.75, "value": 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(t=20, b=10, l=30, r=30),
            font_color="#e8d5b7", transition_duration=500
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        verdict = "✅ Likely survived" if prob > 0.5 else "❌ Likely perished"
        badge_class = "survived-badge" if prob > 0.5 else "perished-badge"
        st.markdown(f'<div style="text-align:center;margin-top:0.5rem"><span class="{badge_class}">{verdict}</span></div>',
                    unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### What drove this prediction?")
        st.caption("SHAP waterfall — each bar shows how much a feature pushed the probability up or down from the baseline.")
        st.plotly_chart(shap_waterfall_fig(shap_vals, prob), use_container_width=True)

    # ── Simulator ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎲 Survival simulator — 100 passengers like you")
    st.caption("Randomly samples 100 historical passengers with your class, sex, and similar age to show a distribution of model-predicted survival odds.")

    sim_probs = run_survival_simulator(model, train_df, sex, pclass, age, feature_names, n=100)
    if sim_probs:
        sim_arr = np.array(sim_probs)
        pct_survive = (sim_arr > 0.5).mean() * 100

        sim_col1, sim_col2 = st.columns([2, 1])
        with sim_col1:
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Histogram(
                x=sim_arr, nbinsx=20,
                marker_color=["#4ade80" if p > 0.5 else "#f87171" for p in sim_arr],
                marker_line_width=0, opacity=0.85
            ))
            fig_sim.add_vline(x=prob, line_dash="dash", line_color="#e8d5b7",
                              annotation_text=f" You ({prob:.0%})", annotation_font_color="#e8d5b7",
                              annotation_position="top right")
            fig_sim.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=200, margin=dict(t=20, b=30, l=40, r=20),
                xaxis=dict(title="Predicted survival probability", tickformat=".0%",
                           tickfont={"color":"#6b7280"}, showgrid=False),
                yaxis=dict(title="Count", tickfont={"color":"#6b7280"}, showgrid=False),
                font_color="#9ca3af", showlegend=False,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        with sim_col2:
            st.metric("Similar passengers who'd survive", f"{pct_survive:.0f}%")
            st.metric("Your predicted probability", f"{prob:.0%}")
            avg_sim = np.mean(sim_arr)
            st.metric("Avg for this group", f"{avg_sim:.0%}")

    # ── Historical twin ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Your historical twin")
    twin = get_historical_twin(train_df, sex, pclass, age)
    if twin is not None:
        twin_survived = twin["Survived"] == 1
        fate_class = "pcard-fate-survived" if twin_survived else "pcard-fate-perished"
        fate_text = "Survived ✅" if twin_survived else "Perished ❌"
        name = str(twin.get("Name", "Unknown"))
        name_short = name[:40] + "…" if len(name) > 40 else name
        st.markdown(f"""
        <div class="pcard">
          <div class="pcard-name">{name_short}</div>
          <div class="pcard-sub">
            Age {twin['Age']:.0f} &nbsp;·&nbsp;
            Class {int(twin['Pclass'])}{'st' if twin['Pclass']==1 else 'nd' if twin['Pclass']==2 else 'rd'} &nbsp;·&nbsp;
            <span class="{fate_class}">{fate_text}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Closest match from the Titanic passenger manifest by gender, class, and age.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compare Two Profiles
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Compare two passenger profiles side-by-side</div>', unsafe_allow_html=True)
    st.caption("Build two profiles and see how changing a single attribute (e.g. gender, class) affects survival odds.")

    p_col1, p_col2 = st.columns(2)

    def profile_inputs(col, label, key_prefix, default_sex="female", default_class=1, default_age=28):
        with col:
            st.markdown(f'<span class="compare-badge">{label}</span>', unsafe_allow_html=True)
            sex_v = st.selectbox("Gender", ["female", "male"], key=f"{key_prefix}_sex",
                                 index=0 if default_sex == "female" else 1)
            pclass_v = st.selectbox("Ticket class", [1, 2, 3], key=f"{key_prefix}_class",
                                    format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} class",
                                    index=default_class - 1)
            age_v = st.slider("Age", 1, 80, default_age, key=f"{key_prefix}_age")
            fare_v = st.slider("Fare (£)", 5, 300, 30, step=5, key=f"{key_prefix}_fare")
            sibsp_v = st.slider("Siblings/spouses", 0, 5, 0, key=f"{key_prefix}_sibsp")
            parch_v = st.slider("Parents/children", 0, 5, 0, key=f"{key_prefix}_parch")
            emb_v = st.selectbox("Embarkation", ["S", "C", "Q"], key=f"{key_prefix}_emb",
                                 format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
            return sex_v, pclass_v, age_v, fare_v, sibsp_v, parch_v, emb_v

    p1 = profile_inputs(p_col1, "Passenger A", "p1", "female", 1, 28)
    p2 = profile_inputs(p_col2, "Passenger B", "p2", "male", 3, 28)

    st.markdown("---")

    prob1, shap1, _ = predict_survival(model, explainer, *p1, feature_names)
    prob2, shap2, _ = predict_survival(model, explainer, *p2, feature_names)

    cmp1, cmp2, cmp3 = st.columns([2, 1, 2])

    with cmp1:
        badge1 = "survived-badge" if prob1 > 0.5 else "perished-badge"
        verdict1 = "✅ Likely survived" if prob1 > 0.5 else "❌ Likely perished"
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem;background:#0e1320;border-radius:12px;border-left:3px solid {'#166534' if prob1>0.5 else '#7f1d1d'}">
          <div style="font-family:'Playfair Display',serif;font-size:3rem;color:{'#4ade80' if prob1>0.5 else '#f87171'};font-weight:700">{prob1:.0%}</div>
          <div style="margin-top:0.5rem"><span class="{badge1}">{verdict1}</span></div>
          <div style="font-size:0.8rem;color:#4a6070;margin-top:0.75rem">Passenger A</div>
        </div>
        """, unsafe_allow_html=True)

    with cmp2:
        delta = prob1 - prob2
        delta_color = "#4ade80" if delta > 0 else "#f87171"
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem;height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center">
          <div style="font-size:0.75rem;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Difference</div>
          <div style="font-family:'Playfair Display',serif;font-size:2.2rem;color:{delta_color};font-weight:700">
            {'+'if delta>0 else ''}{delta:.0%}
          </div>
          <div style="font-size:0.8rem;color:#4a6070;margin-top:0.5rem">in favour of A</div>
        </div>
        """, unsafe_allow_html=True)

    with cmp3:
        badge2 = "survived-badge" if prob2 > 0.5 else "perished-badge"
        verdict2 = "✅ Likely survived" if prob2 > 0.5 else "❌ Likely perished"
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem;background:#0e1320;border-radius:12px;border-right:3px solid {'#166534' if prob2>0.5 else '#7f1d1d'}">
          <div style="font-family:'Playfair Display',serif;font-size:3rem;color:{'#4ade80' if prob2>0.5 else '#f87171'};font-weight:700">{prob2:.0%}</div>
          <div style="margin-top:0.5rem"><span class="{badge2}">{verdict2}</span></div>
          <div style="font-size:0.8rem;color:#4a6070;margin-top:0.75rem">Passenger B</div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP comparison bars
    st.markdown("---")
    st.markdown("#### Feature contribution comparison")

    all_feats = sorted(set(list(shap1.keys()) + list(shap2.keys())))
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        name="Passenger A",
        x=all_feats,
        y=[shap1.get(f, 0) for f in all_feats],
        marker_color="#4ade80", opacity=0.8
    ))
    comp_fig.add_trace(go.Bar(
        name="Passenger B",
        x=all_feats,
        y=[shap2.get(f, 0) for f in all_feats],
        marker_color="#60a5fa", opacity=0.8
    ))
    comp_fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(t=10, b=60, l=40, r=20),
        xaxis=dict(tickfont={"color":"#9ca3af"}, showgrid=False),
        yaxis=dict(tickfont={"color":"#6b7280"}, showgrid=False,
                   zeroline=True, zerolinecolor="#1a2235"),
        font_color="#e8d5b7",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#9ca3af"),
    )
    st.plotly_chart(comp_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Passenger Search
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Search the passenger manifest</div>', unsafe_allow_html=True)
    st.caption("Find any of the 891 passengers by name. See their actual fate vs the model's prediction.")

    search_query = st.text_input("Search by name", placeholder='e.g. "Astor", "Brown", "Smith"…')

    if search_query:
        results = search_passengers(train_df, search_query)
        if results.empty:
            st.info("No passengers found. Try a partial surname.")
        else:
            st.markdown(f"**{len(results)} passenger{'s' if len(results)>1 else ''} found**")
            for _, row in results.head(10).iterrows():
                # Build model prediction for this passenger
                sex_r = row.get("Sex", "male")
                age_r = row["Age"] if not pd.isna(row.get("Age")) else 30
                fare_r = row["Fare"] if not pd.isna(row.get("Fare")) else 30
                emb_r = row.get("Embarked") or "S"
                sibsp_r = int(row.get("SibSp", 0) or 0)
                parch_r = int(row.get("Parch", 0) or 0)
                pclass_r = int(row.get("Pclass", 3))

                pred_prob, _, _ = predict_survival(
                    model, explainer, sex_r, pclass_r, age_r, fare_r,
                    sibsp_r, parch_r, emb_r, feature_names
                )
                actual_survived = row["Survived"] == 1
                model_correct = (pred_prob > 0.5) == actual_survived

                actual_label = "✅ Survived" if actual_survived else "❌ Perished"
                actual_class = "pcard-fate-survived" if actual_survived else "pcard-fate-perished"
                correct_label = "🟢 Model correct" if model_correct else "🔴 Model wrong"
                name_full = str(row.get("Name","Unknown"))

                col_card, col_pred = st.columns([3, 1])
                with col_card:
                    st.markdown(f"""
                    <div class="pcard">
                      <div class="pcard-name">{name_full}</div>
                      <div class="pcard-sub">
                        Age {age_r:.0f} &nbsp;·&nbsp;
                        Class {pclass_r}{'st' if pclass_r==1 else 'nd' if pclass_r==2 else 'rd'} &nbsp;·&nbsp;
                        {'Female' if sex_r=='female' else 'Male'} &nbsp;·&nbsp;
                        £{fare_r:.0f} fare &nbsp;·&nbsp;
                        <span class="{actual_class}">{actual_label}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_pred:
                    st.metric("Model prediction", f"{pred_prob:.0%}", help=correct_label)

            if len(results) > 10:
                st.caption(f"Showing top 10 of {len(results)} results. Refine your search to narrow down.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Story
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">The human story behind the numbers</div>', unsafe_allow_html=True)

    # Coloured metric cards
    st.markdown(f"""
    <div class="metric-grid">
      <div class="mcard blue">
        <div class="mcard-num">{n_total:,}</div>
        <div class="mcard-lbl">Total passengers</div>
      </div>
      <div class="mcard green">
        <div class="mcard-num">{n_survived} <span style="font-size:1.1rem;color:#166534">({train_df['Survived'].mean():.0%})</span></div>
        <div class="mcard-lbl">Survivors</div>
      </div>
      <div class="mcard amber">
        <div class="mcard-num">{train_df['Age'].mean():.0f} yrs</div>
        <div class="mcard-lbl">Average age</div>
      </div>
      <div class="mcard red">
        <div class="mcard-num">£{train_df['Fare'].mean():.0f}</div>
        <div class="mcard-lbl">Average fare</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("##### Survival by class & gender")
        grp = train_df.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
        grp["Class"] = grp["Pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})
        fig1 = px.bar(grp, x="Class", y="Survived", color="Sex", barmode="group",
                      color_discrete_map={"female": "#4ade80", "male": "#60a5fa"},
                      labels={"Survived": "Survival rate"}, text_auto=".0%")
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#9ca3af", yaxis_tickformat=".0%",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig1.update_traces(textfont_color="#e8d5b7")
        st.plotly_chart(fig1, use_container_width=True)

    with v2:
        st.markdown("##### Age distribution: survivors vs non-survivors")
        survived = train_df[train_df["Survived"] == 1]["Age"].dropna()
        perished = train_df[train_df["Survived"] == 0]["Age"].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=survived, name="Survived", marker_color="#4ade80", opacity=0.7, nbinsx=30))
        fig2.add_trace(go.Histogram(x=perished, name="Perished", marker_color="#f87171", opacity=0.7, nbinsx=30))
        fig2.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

    v3, v4 = st.columns(2)
    with v3:
        st.markdown("##### Fare vs survival (by class)")
        fig3 = px.box(train_df, x="Pclass", y="Fare", color="Survived",
                      color_discrete_map={0: "#f87171", 1: "#4ade80"},
                      labels={"Pclass": "Class", "Survived": "Survived"},
                      category_orders={"Pclass": [1, 2, 3]})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#9ca3af", legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3, use_container_width=True)

    with v4:
        st.markdown("##### Family size vs survival rate")
        train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
        fs = train_df.groupby("FamilySize")["Survived"].mean().reset_index()
        fig4 = px.line(fs, x="FamilySize", y="Survived", markers=True,
                       labels={"FamilySize": "Family size (incl. self)", "Survived": "Survival rate"},
                       color_discrete_sequence=["#fbbf24"])
        fig4.update_traces(line_width=2, marker_size=8)
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#9ca3af", yaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Missing data audit")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ["Feature", "Missing values"]
    missing["% missing"] = (missing["Missing values"] / len(train_df) * 100).round(1).astype(str) + "%"
    st.dataframe(missing, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Model transparency & fairness</div>', unsafe_allow_html=True)

    preds = model.predict(X_train)
    tp = ((preds == 1) & (y_train == 1)).sum()
    fp = ((preds == 1) & (y_train == 0)).sum()
    fn = ((preds == 0) & (y_train == 1)).sum()
    tn = ((preds == 0) & (y_train == 0)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    train_acc = (preds == y_train).mean()

    # Coloured metric cards
    st.markdown(f"""
    <div class="metric-grid">
      <div class="mcard amber">
        <div class="mcard-num">{cv_mean:.1%} <span style="font-size:0.9rem;color:#78350f">±{cv_std:.1%}</span></div>
        <div class="mcard-lbl">CV accuracy (5-fold)</div>
      </div>
      <div class="mcard blue">
        <div class="mcard-num">{precision:.1%}</div>
        <div class="mcard-lbl">Precision</div>
      </div>
      <div class="mcard green">
        <div class="mcard-num">{recall:.1%}</div>
        <div class="mcard-lbl">Recall</div>
      </div>
      <div class="mcard red">
        <div class="mcard-num">{f1:.1%}</div>
        <div class="mcard-lbl">F1 Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    i1, i2 = st.columns(2)

    with i1:
        st.markdown("##### Global feature importance (SHAP)")
        shap_values = explainer(X_train)
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": mean_shap})
        imp_df = imp_df.sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#1c0505", "#f87171", "#4ade80"])
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#9ca3af", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with i2:
        st.markdown("##### Fairness audit — survival rate by group")
        st.caption("Does the model reflect historical bias, or amplify it?")
        groups = {
            "1st class female": train_df[(train_df.Pclass==1)&(train_df.Sex=="female")]["Survived"].mean(),
            "2nd class female": train_df[(train_df.Pclass==2)&(train_df.Sex=="female")]["Survived"].mean(),
            "3rd class female": train_df[(train_df.Pclass==3)&(train_df.Sex=="female")]["Survived"].mean(),
            "1st class male":   train_df[(train_df.Pclass==1)&(train_df.Sex=="male")]["Survived"].mean(),
            "2nd class male":   train_df[(train_df.Pclass==2)&(train_df.Sex=="male")]["Survived"].mean(),
            "3rd class male":   train_df[(train_df.Pclass==3)&(train_df.Sex=="male")]["Survived"].mean(),
        }
        fair_df = pd.DataFrame({"Group": list(groups.keys()), "Actual survival rate": list(groups.values())})
        fig_fair = px.bar(fair_df, x="Actual survival rate", y="Group", orientation="h",
                          color="Actual survival rate",
                          color_continuous_scale=["#7f1d1d","#f87171","#4ade80","#052e16"],
                          text_auto=".0%")
        fig_fair.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#9ca3af", coloraxis_showscale=False)
        fig_fair.update_traces(textfont_color="#e8d5b7")
        st.plotly_chart(fig_fair, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Confusion matrix")
    cm_data = [[int(tn), int(fp)], [int(fn), int(tp)]]
    fig_cm = px.imshow(cm_data,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=["Predicted: No", "Predicted: Yes"],
                       y=["Actual: No", "Actual: Yes"],
                       color_continuous_scale=["#07090f", "#1e3a5f", "#4ade80"],
                       text_auto=True)
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#9ca3af", width=440)
    st.plotly_chart(fig_cm)

    st.markdown("---")
    st.markdown("##### Model card")
    st.markdown(f"""
| | |
|---|---|
| **Model** | XGBoost Classifier |
| **Training data** | Titanic train.csv (891 passengers) |
| **Features used** | Sex, Pclass, Age, Fare, SibSp, Parch, Embarked, FamilySize, IsAlone, Title, FarePerPerson |
| **CV Accuracy** | {cv_mean:.1%} ± {cv_std:.1%} (5-fold) |
| **Precision / Recall** | {precision:.1%} / {recall:.1%} |
| **Known limitations** | Class and gender strongly predict survival — reflects historical reality, not a model flaw |
| **Intended use** | Educational / portfolio exploration only |
| **Not intended for** | Any real-world decision-making |
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Made with ❤️ by **[@raunakpatil](https://github.com/raunakpatil)** · Built with Streamlit · XGBoost · SHAP · Plotly")
