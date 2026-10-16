import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import history
from data_utils import (PORTS, build_features, clean_data, get_historical_twin, load_data,
                        passenger_frame, search_passengers)
from model import explain_passenger, global_importance, make_explainer, run_survival_simulator, train_model
from ship import berth_for, ship_berth, ship_sinking
from ui import (AMBER, BLUE, DIM, GREEN, PINK, RED, card, chart, esc, hint, label, md, prob_bar, row,
                setup, stat, verdict_chip)

st.set_page_config(page_title="Titanic Survival Explorer", page_icon="🚢", layout="wide",
                   initial_sidebar_state="collapsed")
setup()


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model_and_data():
    raw_df = load_data()
    df = clean_data(raw_df)
    X = build_features(df)
    y = df["Survived"].to_numpy()
    model, oof_probs, cv_mean, cv_std = train_model(X, y)
    explainer = make_explainer(model, X)
    importance = global_importance(explainer, X)
    return raw_df, df, y, model, explainer, oof_probs, cv_mean, cv_std, importance


try:
    with st.spinner("Loading passenger manifest & training model…"):
        raw_df, df, y_train, model, explainer, oof_probs, cv_mean, cv_std, importance = get_model_and_data()
except RuntimeError as err:
    st.error(str(err))
    st.stop()

n_total = len(raw_df)


def ordinal(n):
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n, 'th') }"


def pts(v):
    return f"{v * 100:+.1f} pts"


def tone(v):
    return "flat" if abs(v) < 0.005 else ("up" if v > 0 else "down")


# ── Top bar ───────────────────────────────────────────────────────────────────
md(f"""
<div class="topbar">
  <div class="brand"><b>Titanic</b><span>survival-explorer</span></div>
  <div class="top-pills">
    <span class="top-pill"><span class="dot"></span>Model ready · {cv_mean:.0%} accurate</span>
    <a class="top-pill" href="https://github.com/raunakpatil/titanic-survival-predictor" target="_blank">GitHub ↗</a>
  </div>
</div>
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Predict", "Compare", "Passengers", "Data story", "Model"])


# ── Shared widgets ────────────────────────────────────────────────────────────
PROFILE_DEFAULTS = {"sex": "female", "pclass": 1, "embarked": "S", "age": 28, "fare": 30, "sibsp": 0, "parch": 0}

PRESETS = {
    "Rose, 17 · 1st class": dict(sex="female", pclass=1, embarked="S", age=17, fare=150, sibsp=0, parch=1),
    "Jack, 20 · 3rd class": dict(sex="male", pclass=3, embarked="S", age=20, fare=5, sibsp=0, parch=0),
    "Molly Brown, 44": dict(sex="female", pclass=1, embarked="C", age=44, fare=30, sibsp=0, parch=0),
    "Boy of 6 · 2nd class": dict(sex="male", pclass=2, embarked="S", age=6, fare=30, sibsp=1, parch=1),
    "Family of six · 3rd": dict(sex="female", pclass=3, embarked="S", age=40, fare=30, sibsp=1, parch=4),
}


def _apply_preset(prefix):
    choice = st.session_state.get(f"{prefix}_preset")
    if choice:
        for k, v in PRESETS[choice].items():
            st.session_state[f"{prefix}_{k}"] = v


def profile_inputs(prefix, presets=False, **overrides):
    """Passenger profile widgets; returns a one-row passenger frame."""
    defaults = {**PROFILE_DEFAULTS, **overrides}
    for k, v in defaults.items():
        st.session_state.setdefault(f"{prefix}_{k}", v)

    if presets:
        st.pills("Try a passenger", list(PRESETS), key=f"{prefix}_preset",
                 on_change=_apply_preset, args=(prefix,))

    def seg(lbl, options, k, fmt=str):
        return st.segmented_control(lbl, options, key=f"{prefix}_{k}", format_func=fmt,
                                    width="stretch") or defaults[k]

    sex = seg("Gender", ["female", "male"], "sex", str.title)
    pclass = seg("Ticket class", [1, 2, 3], "pclass", ordinal)
    embarked = seg("Boarded at", list(PORTS), "embarked", PORTS.get)
    age = st.slider("Age", 1, 80, key=f"{prefix}_age")
    fare = st.slider("Fare paid (£)", 5, 300, step=5, key=f"{prefix}_fare")
    c1, c2 = st.columns(2)
    sibsp = c1.slider("Siblings / spouses", 0, 5, key=f"{prefix}_sibsp")
    parch = c2.slider("Parents / children", 0, 5, key=f"{prefix}_parch")
    return passenger_frame(sex, pclass, age, fare, sibsp, parch, embarked)


FACTOR_NOTES = {
    "Gender": "“Women and children first” was enforced at the lifeboats",
    "Title (Mr/Mrs/Master)": "Mr, Mrs, Miss or Master: age and gender rolled into one",
    "Ticket class": "Upper-deck cabins were closest to the lifeboats",
    "Fare paid": "A proxy for wealth and where your cabin was",
    "Fare per person": "Ticket price split across your travelling party",
    "Age": "Young children were prioritised; older passengers fared worse",
    "Siblings/spouses": "Travelling with a partner or siblings",
    "Parents/children": "Travelling with parents or children",
    "Family size": "Small groups stuck together; large families struggled",
    "Travelling alone": "Nobody to look out for you, or to wait for",
    "Embarked Cherbourg": "Port of boarding, which shaped who you sailed with",
    "Embarked Queenstown": "Port of boarding, which shaped who you sailed with",
    "Embarked Southampton": "Port of boarding, which shaped who you sailed with",
    "Other features": "Several small effects combined",
}


def factor_value(name, p):
    family = p["SibSp"] + p["Parch"] + 1
    port = {"Embarked Cherbourg": "C", "Embarked Queenstown": "Q", "Embarked Southampton": "S"}
    if name in port:
        return "yes" if p["Embarked"] == port[name] else "no"
    title = "Mrs / Miss" if p["Sex"] == "female" else ("Master" if p["Age"] < 16 else "Mr")
    return {
        "Gender": p["Sex"].title(), "Title (Mr/Mrs/Master)": title,
        "Ticket class": f"{ordinal(p['Pclass'])} class", "Age": f"{p['Age']} years",
        "Fare paid": f"£{p['Fare']}", "Fare per person": f"£{p['Fare'] / family:.0f}",
        "Siblings/spouses": p["SibSp"], "Parents/children": p["Parch"],
        "Family size": family, "Travelling alone": "yes" if family == 1 else "no",
    }.get(name, "")


def factor_rows(contribs, base, prob, passenger):
    p = passenger.iloc[0].to_dict()
    out = [row("Average passenger", "Where everyone starts before your details count", f"{base:.0%}",
               "flat", glyph="≈")]
    for name, v in contribs.items():
        value = factor_value(name, p)
        title = f"{name} <span style='color:{DIM};font-weight:400'>· {esc(value)}</span>" if value != "" else name
        out.append(row(title, FACTOR_NOTES.get(name, ""), pts(v), tone(v)))
    out.append(row("Your prediction", "Average passenger plus every factor above", f"{prob:.0%}",
                   "up" if prob > 0.5 else "down", glyph="=", highlight=True))
    return '<div class="rows">' + "".join(out) + "</div>"


def shap_waterfall_fig(contribs, base, prob):
    values = list(contribs.values())
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(values) + ["total"],
        x=[base] + values + [prob],
        y=["Average passenger"] + list(contribs) + ["Your prediction"],
        connector={"line": {"color": "#2a2a2a", "width": 1}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": BLUE}},
        text=[f"{base:.0%}"] + [f"{v:+.1%}" for v in values] + [f"{prob:.0%}"],
        textposition="outside",
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=60), showlegend=False,
                      xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed", showgrid=False))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.75, 1])

    with right, card("setup"):
        label("user", "Your passenger", "1912")
        hint("Pick a famous passenger or build your own. Everything updates instantly.")
        you = profile_inputs("you", presets=True)
    p_you = you.iloc[0]

    prob, base_prob, contribs = explain_passenger(model, explainer, you)
    better_than = (oof_probs < prob).mean()

    with left, card("predict"):
        label("activity", "Your survival odds", f"model trained on {n_total} real passengers")
        md(f'<div class="headline">{prob:.0%} chance of survival</div>{verdict_chip(prob)}')
        md(prob_bar(prob, marker=base_prob,
                    left=f"Better odds than {better_than:.0%} of the people on board",
                    right=f"│ average passenger {base_prob:.0%}"))
        md('<div style="height:1.1rem"></div>')
        label("sliders", "What drove it")
        md(factor_rows(contribs, base_prob, prob, you))
        with st.expander("Show SHAP waterfall chart"):
            chart(shap_waterfall_fig(contribs, base_prob, prob), height=360)

    left, right = st.columns([1.75, 1])

    with left, card("berth"):
        _, where, decks = berth_for(p_you["Pclass"], p_you["Sex"])
        label("ship", "Where you'd be on board", "drag to explore")
        md(f'<div class="headline" style="font-size:2rem">{decks} decks '
           f'<span style="color:var(--muted);font-size:1rem">between your berth and the lifeboats</span></div>')
        hint(f"{ordinal(p_you['Pclass'])}-class {'men' if p_you['Sex'] == 'male' else 'women'} slept on "
             f"{where}. When the order came, the climb to the boat deck was "
             f"{'a short one' if decks <= 2 else 'a long one, through crowded corridors and stairways'}.")
        ship_berth(p_you["Pclass"], p_you["Sex"], GREEN if prob > 0.5 else RED, height=380, key="ship_berth")

    with right, card("twin"):
        label("clock", "Your historical twin")
        twin = get_historical_twin(raw_df, p_you["Sex"], p_you["Pclass"], p_you["Age"])
        if twin is not None:
            survived = twin["Survived"] == 1
            md(f"""
            <div style="margin-top:1rem;font-size:1.25rem;font-weight:400;letter-spacing:-.01em">{esc(twin['Name'])}</div>
            <div class="hint">Age {twin['Age']:.0f} · {ordinal(int(twin['Pclass']))} class ·
              boarded at {PORTS.get(twin['Embarked'], 'an unknown port')} · paid £{twin['Fare']:.0f}</div>
            <div style="margin-top:.75rem"><span class="chip {'green' if survived else 'red'}">
              {'Survived' if survived else 'Perished'}</span></div>
            """)
            group = raw_df[(raw_df["Sex"] == p_you["Sex"]) & (raw_df["Pclass"] == p_you["Pclass"])]
            md('<div class="stats">'
               + stat(f"{group['Survived'].mean():.0%}",
                      f"of {p_you['Sex']} {ordinal(p_you['Pclass'])}-class passengers actually survived")
               + stat(len(group), "people in that group")
               + "</div>")
            md('<p class="tiny" style="margin-top:1.2rem">Closest real passenger by gender, class and age.</p>')

    sim_probs = run_survival_simulator(model, df, p_you["Sex"], p_you["Pclass"], p_you["Age"], n=100)
    with card("sim"):
        n_sim = 0 if sim_probs is None else len(sim_probs)
        label("users", "Passengers like you", f"{n_sim} real passengers")
        hint(f"Real {p_you['Sex']} passengers in {ordinal(p_you['Pclass'])} class aged within 15 years of you, "
             "and how the model rates each of their chances.")
        if sim_probs is not None:
            c_chart, c_stats = st.columns([2.2, 1])
            with c_chart:
                fig_sim = go.Figure()
                for name, mask, color in [("Predicted to survive", sim_probs > 0.5, GREEN),
                                          ("Predicted to perish", sim_probs <= 0.5, RED)]:
                    fig_sim.add_trace(go.Histogram(x=sim_probs[mask], name=name, marker_color=color,
                                                   opacity=0.85, xbins=dict(start=0, end=1, size=0.05)))
                fig_sim.add_vline(x=prob, line_dash="dot", line_color="#ededed",
                                  annotation_text=f" you · {prob:.0%}", annotation_font_color="#ededed",
                                  annotation_position="top right")
                fig_sim.update_layout(barmode="overlay", margin=dict(t=30, b=10, l=10, r=10),
                                      xaxis=dict(tickformat=".0%", range=[0, 1]),
                                      yaxis=dict(title="passengers"))
                chart(fig_sim, height=230)
            with c_stats:
                md('<div class="stats" style="grid-template-columns:1fr">'
                   + stat(f"{(sim_probs > 0.5).mean():.0%}", "predicted to survive")
                   + stat(f"{sim_probs.mean():.0%}", "average odds in this group")
                   + stat(f"{(prob - sim_probs.mean()) * 100:+.0f}", "points you are above or below them")
                   + "</div>")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compare
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)
    results = {}
    for col, name, prefix, defaults in [(col_a, "Passenger A", "a", dict(sex="female", pclass=1)),
                                        (col_b, "Passenger B", "b", dict(sex="male", pclass=3, fare=10))]:
        with col, card(f"cmp_{prefix}"):
            label("user", name)
            head = st.container()
            passenger = profile_inputs(prefix, **defaults)
            p, _, c = explain_passenger(model, explainer, passenger)
            results[name] = (p, c)
            with head:
                md(f'<div class="headline">{p:.0%}</div>{verdict_chip(p)}')
                md(prob_bar(p, left="chance of survival", right=""))

    (prob_a, shap_a), (prob_b, shap_b) = results.values()
    delta = prob_a - prob_b

    with card("cmp_diff"):
        label("columns", "What separates them")
        if abs(delta) < 0.005:
            md('<div class="headline">Dead even</div>')
        else:
            leader = "A" if delta > 0 else "B"
            md(f'<div class="headline">{abs(delta) * 100:.0f} points '
               f'<span style="color:var(--muted);font-size:1.1rem">in favour of passenger {leader}</span></div>')
        hint("Percentage points each factor added to or removed from each passenger's odds.")
        feats = [f for f in dict.fromkeys([*shap_a, *shap_b]) if f != "Other features"]
        fig_cmp = go.Figure()
        for name, contribs_v, color in [("Passenger A", shap_a, GREEN), ("Passenger B", shap_b, BLUE)]:
            fig_cmp.add_trace(go.Bar(name=name, y=feats, x=[contribs_v.get(f, 0) for f in feats],
                                     orientation="h", marker_color=color))
        fig_cmp.update_layout(barmode="group", margin=dict(t=30, b=10, l=10, r=10),
                              xaxis=dict(tickformat="+.0%", zeroline=True, zerolinecolor="#444"),
                              yaxis=dict(autorange="reversed", showgrid=False))
        chart(fig_cmp, height=360)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Passengers
# ═══════════════════════════════════════════════════════════════════════════════
def _set_search():
    st.session_state["search_q"] = st.session_state.get("search_try") or ""


with tab3:
    left, right = st.columns([1.75, 1])
    with left, card("search"):
        label("search", "Search the manifest", f"{n_total} passengers")
        query = st.text_input("Name", key="search_q", placeholder="Type a name, e.g. Astor, Strauss, Sage…",
                              label_visibility="collapsed")
        st.pills("Try", ["Astor", "Strauss", "Sage", "Goodwin", "Brown", "Andersson"], key="search_try",
                 on_change=_set_search, label_visibility="collapsed")
        results_df = search_passengers(raw_df, query)
        if not query:
            hint("Each passenger's prediction is out-of-fold: it comes from a model that never saw them "
                 "during training, so you can see where it really gets things right and wrong.")
        elif results_df.empty:
            hint("No passengers found. Try a partial surname (at least 2 characters).")
        else:
            out = []
            for idx, r in results_df.head(25).iterrows():
                p = oof_probs[idx]
                survived = r["Survived"] == 1
                correct = (p > 0.5) == survived
                age_text = f"age {r['Age']:.0f}" if pd.notna(r["Age"]) else "age unknown"
                sub = (f"{age_text} · {ordinal(int(r['Pclass']))} class · {r['Sex']} · £{r['Fare']:.0f} · "
                       f"{'survived' if survived else 'perished'}")
                value = (f"model {p:.0%} &nbsp;<span class='chip {'grey' if correct else 'red'}'>"
                         f"{'correct' if correct else 'wrong'}</span>")
                out.append(row(esc(r["Name"]), sub, value, "up" if survived else "down",
                               glyph="✓" if survived else "✕"))
            md('<div class="rows">' + "".join(out) + "</div>")
            if len(results_df) > 25:
                md(f'<p class="tiny">Showing 25 of {len(results_df)}. Refine your search to narrow down.</p>')

    with right, card("search_stats"):
        if query and not results_df.empty:
            correct = ((oof_probs[results_df.index] > 0.5) == (results_df["Survived"] == 1))
            label("chart", "This search")
            md('<div class="stats">'
               + stat(len(results_df), "passengers found")
               + stat(f"{results_df['Survived'].mean():.0%}", "survived")
               + stat(f"{correct.mean():.0%}", "predicted correctly")
               + stat(f"£{results_df['Fare'].median():.0f}", "median fare")
               + "</div>")
        else:
            all_correct = ((oof_probs > 0.5) == y_train)
            label("chart", "Whole manifest")
            md('<div class="stats">'
               + stat(n_total, "passengers")
               + stat(f"{y_train.mean():.0%}", "survived")
               + stat(f"{all_correct.mean():.0%}", "predicted correctly")
               + stat(int((~all_correct).sum()), "the model gets wrong")
               + "</div>")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data story
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    with card("sinking"):
        label("ship", "The sinking, minute by minute", "times are approximate")
        c_ship, c_info = st.columns([1.9, 1])
        with c_info:
            minute = st.select_slider("Time", options=list(range(0, history.END + 1, 5)), value=65,
                                      format_func=history.clock, key="sink_minute")
            s = history.state_at(minute)
            md(f'<div class="headline" style="margin-top:.4rem">{history.clock(minute)}</div>'
               f'<p class="hint" style="color:var(--text);font-size:.95rem">{s["event"][1]}</p>'
               f'<p class="tiny">latest event at {history.clock(s["event"][0])}</p>')
            md('<div class="stats">'
               + stat(f'{s["boats"]}<span style="font-size:1rem;color:var(--muted)"> / 20</span>',
                      "lifeboats launched")
               + stat(f'{s["seats"]:,}', f"seats launched, of {history.TOTAL_SEATS:,} for {history.ABOARD:,} aboard")
               + "</div>")
            dots = "".join(
                f'<span title="Boat {b} · {history.clock(t)} · {seats} seats" style="display:inline-block;'
                f'width:12px;height:12px;border-radius:50%;margin:0 6px 6px 0;'
                f'background:{GREEN if t <= minute else "#2a2a2a"}"></span>'
                for b, t, seats in history.LIFEBOATS)
            md(f'<div style="margin-top:1.2rem">{dots}</div>'
               '<p class="tiny">Each dot is a lifeboat, in launch order. Hover for details.</p>')
        with c_ship:
            ship_sinking(s["pitch"], s["sink"], s["boats"], height=400, key="ship_sinking")

    d = raw_df.assign(FamilySize=raw_df["SibSp"] + raw_df["Parch"] + 1,
                      Class=raw_df["Pclass"].map(ordinal),
                      Outcome=raw_df["Survived"].map({0: "Perished", 1: "Survived"}))
    rate = lambda mask: d.loc[mask, "Survived"].mean()  # noqa: E731

    left, right = st.columns([1.75, 1])
    with left, card("findings"):
        label("trending", "What the data says")
        hint(f"Five patterns from the {n_total} passengers in the training manifest.")
        findings = [
            (GREEN, f"{rate(d.Sex == 'female'):.0%} of women survived",
             f"vs {rate(d.Sex == 'male'):.0%} of men"),
            (BLUE, f"{rate(d.Pclass == 1):.0%} of 1st-class passengers survived",
             f"vs {rate(d.Pclass == 3):.0%} in 3rd class"),
            (AMBER, f"{rate(d.Age < 12):.0%} of children under 12 survived",
             f"vs {rate(d.Age >= 12):.0%} of everyone older"),
            (PINK, f"{rate(d.FamilySize.between(2, 4)):.0%} of small families (2–4) survived",
             f"vs {rate(d.FamilySize == 1):.0%} of solo travellers and {rate(d.FamilySize >= 5):.0%} "
             "of families of 5+"),
            (RED, f"Survivors paid a median £{d.loc[d.Survived == 1, 'Fare'].median():.0f}",
             f"vs £{d.loc[d.Survived == 0, 'Fare'].median():.0f} for those who perished"),
        ]
        md('<ul class="dots">' + "".join(f'<li style="--c:{c}">{a} <span>{b}</span></li>'
                                         for c, a, b in findings) + "</ul>")

    with right, card("numbers"):
        label("grid", "By the numbers")
        md('<div class="stats">'
           + stat(f"{n_total:,}", "passengers")
           + stat(int(d.Survived.sum()), f"survived ({d.Survived.mean():.0%})")
           + stat(f"{d.Age.mean():.0f}", "average age")
           + stat(f"£{d.Fare.mean():.0f}", "average fare")
           + "</div>")

    c1, c2 = st.columns(2)
    with c1, card("by_class"):
        label("chart", "Survival by class & gender")
        grp = d.groupby(["Class", "Sex"])["Survived"].mean().reset_index()
        fig1 = px.bar(grp, x="Class", y="Survived", color="Sex", barmode="group", text_auto=".0%",
                      color_discrete_map={"female": GREEN, "male": BLUE},
                      labels={"Survived": "survival rate", "Class": ""})
        fig1.update_layout(yaxis_tickformat=".0%", margin=dict(t=30, b=10, l=10, r=10))
        fig1.update_traces(textfont_color="#ededed", textposition="outside", cliponaxis=False)
        chart(fig1, height=300)

    with c2, card("by_age"):
        label("users", "Age of survivors vs non-survivors")
        fig2 = go.Figure()
        for lbl, outcome, color in [("Survived", 1, GREEN), ("Perished", 0, RED)]:
            fig2.add_trace(go.Histogram(x=d.loc[d.Survived == outcome, "Age"].dropna(), name=lbl,
                                        marker_color=color, opacity=0.85, xbins=dict(start=0, end=80, size=5)))
        fig2.update_layout(barmode="stack", xaxis_title="age", yaxis_title="passengers",
                           margin=dict(t=30, b=10, l=10, r=10))
        chart(fig2, height=300)

    c3, c4 = st.columns(2)
    with c3, card("by_fare"):
        label("activity", "Fare paid, by class and fate")
        fig3 = px.box(d, x="Class", y="Fare", color="Outcome",
                      color_discrete_map={"Perished": RED, "Survived": GREEN},
                      category_orders={"Class": ["1st", "2nd", "3rd"], "Outcome": ["Survived", "Perished"]},
                      labels={"Fare": "fare (£)", "Class": ""})
        fig3.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        chart(fig3, height=300)

    with c4, card("by_family"):
        label("users", "Family size vs survival rate")
        fs = d.groupby("FamilySize")["Survived"].agg(["mean", "size"]).reset_index()
        fig4 = px.line(fs, x="FamilySize", y="mean", markers=True, hover_data={"size": True},
                       labels={"FamilySize": "family size (incl. self)", "mean": "survival rate",
                               "size": "passengers"},
                       color_discrete_sequence=[AMBER])
        fig4.update_traces(line_width=2, marker_size=8)
        fig4.update_layout(yaxis_tickformat=".0%", margin=dict(t=30, b=10, l=10, r=10))
        chart(fig4, height=300)

    with card("missing"):
        label("info", "Missing data, and how it's handled")
        handling = {
            "Age": "Filled with the median age for the passenger's class & gender",
            "Cabin": "Not used by the model",
            "Embarked": "Filled with Southampton, the most common port",
            "Fare": "Filled with the median fare",
        }
        missing = raw_df.isna().sum()
        md('<div class="rows">' + "".join(
            row(col, handling.get(col, ""), f"{n:,} missing · {n / n_total:.0%}", "flat")
            for col, n in missing[missing > 0].items()) + "</div>")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    preds = (oof_probs > 0.5).astype(int)
    tp = int(((preds == 1) & (y_train == 1)).sum())
    fp = int(((preds == 1) & (y_train == 0)).sum())
    fn = int(((preds == 0) & (y_train == 1)).sum())
    tn = int(((preds == 0) & (y_train == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    naive = 1 - y_train.mean()

    left, right = st.columns([1.75, 1])
    with left, card("accuracy"):
        label("check", "How accurate is it?", "5-fold cross-validation")
        md(f'<div class="headline">{cv_mean:.1%} accurate '
           f'<span style="color:var(--muted);font-size:1rem">± {cv_std:.1%}</span></div>')
        hint(f"Tested only on passengers it never saw during training, it gets about "
             f"{cv_mean * 100:.0f} out of every 100 right.")
        md(prob_bar(cv_mean, marker=naive, left=f"│ guessing “perished” for everyone scores {naive:.0%}",
                    right="100%"))
        md('<div class="stats" style="grid-template-columns:repeat(3,1fr);margin-top:1.6rem">'
           + stat(f"{precision:.0%}", "precision: when it says “survived”, it's right this often")
           + stat(f"{recall:.0%}", "recall: share of real survivors it finds")
           + stat(f"{f1:.0%}", "F1: balance of the two")
           + "</div>")

    with right, card("confusion"):
        label("grid", "Where it goes wrong", "out-of-fold")
        md(f"""
        <div class="cm">
          <div></div><div>predicted perished</div><div>predicted survived</div>
          <div>perished</div>
          <div class="cell ok"><b>{tn}</b>correct</div>
          <div class="cell bad"><b>{fp}</b>false hope</div>
          <div>survived</div>
          <div class="cell bad"><b>{fn}</b>missed survivors</div>
          <div class="cell ok"><b>{tp}</b>correct</div>
        </div>
        """)
        lean = ("It misses more survivors than it wrongly rescues: it leans towards predicting “perished”, "
                "like the majority of passengers." if fn > fp else
                "It wrongly rescues more people than it misses: it leans towards predicting “survived”.")
        hint(lean)

    c1, c2 = st.columns(2)
    with c1, card("importance"):
        label("sliders", "What the model relies on")
        hint("Average swing in survival odds caused by each feature, in percentage points (mean |SHAP|).")
        imp = importance.sort_values(ascending=True)
        colors = [GREEN if i >= len(imp) - 3 else "#3a3a3a" for i in range(len(imp))]
        fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=colors,
                                   text=[f"{v * 100:.1f}" for v in imp.values], textposition="outside",
                                   cliponaxis=False))
        fig_imp.update_layout(xaxis=dict(showticklabels=False, showgrid=False),
                              yaxis=dict(showgrid=False), margin=dict(t=10, b=10, l=10, r=40))
        chart(fig_imp, height=420)

    with c2, card("fairness"):
        label("scale", "Fairness audit: reflect or amplify?")
        hint("Actual survival rate vs the share the model predicts will survive. Where the blue bar is "
             "more extreme than the amber one, the model exaggerates that group's historical fate.")
        audit = raw_df.assign(Predicted=preds)
        audit["Group"] = audit["Pclass"].map(ordinal) + " class " + audit["Sex"]
        fair_df = (audit.groupby("Group")[["Survived", "Predicted"]].mean()
                   .rename(columns={"Survived": "Actual"})
                   .reset_index()
                   .melt(id_vars="Group", var_name="Rate", value_name="Survival rate"))
        group_order = [f"{ordinal(c)} class {s}" for s in ("female", "male") for c in (1, 2, 3)]
        fig_fair = px.bar(fair_df, x="Survival rate", y="Group", color="Rate", orientation="h",
                          barmode="group", text_auto=".0%", category_orders={"Group": group_order},
                          color_discrete_map={"Actual": AMBER, "Predicted": BLUE},
                          labels={"Group": "", "Survival rate": ""})
        fig_fair.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1.12]), yaxis=dict(showgrid=False),
                               margin=dict(t=30, b=10, l=10, r=10))
        fig_fair.update_traces(textposition="outside", cliponaxis=False, textfont_color="#bdbdbd")
        chart(fig_fair, height=420)

    with card("model_card"):
        label("file", "Model card")
        model_card = [
            ("Model", "XGBoost classifier · 200 trees, depth 4"),
            ("Training data", f"Titanic train.csv · {n_total} passengers"),
            ("Features", "Sex, class, age, fare, siblings/spouses, parents/children, port, family size, "
                         "travelling alone, title, fare per person"),
            ("Evaluation", f"Stratified 5-fold CV · accuracy {cv_mean:.1%} ± {cv_std:.1%} · "
                           f"precision {precision:.0%} · recall {recall:.0%}"),
            ("Explanations", "SHAP TreeExplainer in probability space (interventional, 100-passenger background)"),
            ("Known limitations", "Class and gender dominate. That reflects history, but the audit above "
                                  "shows the model sharpens those gaps"),
            ("Intended use", "Education and portfolio exploration only, never real-world decisions"),
        ]
        md('<div class="rows">' + "".join(row(k, v, glyph="·") for k, v in model_card) + "</div>")


md('<div class="foot">Made by <a href="https://github.com/raunakpatil" target="_blank">@raunakpatil</a> · '
   'Streamlit · XGBoost · SHAP · Plotly · three.js</div>')
