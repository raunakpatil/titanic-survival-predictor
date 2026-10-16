"""Visual system for the app: global CSS, Plotly template and small HTML helpers.

Light page, dark rounded cards, thin oversized numbers and small uppercase
labels — modern and minimal, so the data does the talking.
"""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GREEN = "#4ade80"
RED = "#f87171"
BLUE = "#60a5fa"
AMBER = "#fbbf24"
PINK = "#e879f9"
MUTED = "#8a8a8a"
DIM = "#5c5c5c"
TEXT = "#ededed"

_ICON_PATHS = {
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "users": '<circle cx="9" cy="8" r="4"/><path d="M2 21a7 7 0 0 1 14 0"/><path d="M16 4a4 4 0 0 1 0 8"/><path d="M22 21a7 7 0 0 0-5-6.7"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 16v-4"/><path d="M12 16V8"/><path d="M17 16v-7"/>',
    "trending": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "scale": '<path d="M12 3v18M7 21h10M3 7h18M6 7l-3 7a3 3 0 0 0 6 0zM18 7l-3 7a3 3 0 0 0 6 0z"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "file": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
    "check": '<path d="m9 11 3 3 8-8"/><path d="M20 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "columns": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/>',
    "ship": '<path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1s1.2 1 2.5 1c2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M19.4 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.9 5.3 2.8 7.8"/><path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/><path d="M12 10v4M12 2v3"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
}


def icon(name, size=14):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --page: #efefef; --card: #161616; --card-2: #1e1e1e; --line: #262626;
  --text: #ededed; --muted: #8a8a8a; --dim: #5c5c5c;
  --green: #4ade80; --red: #f87171; --blue: #60a5fa; --amber: #fbbf24; --pink: #e879f9;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
}

/* ── Page ─────────────────────────────────────────── */
.stApp { background: var(--page); }
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea {
  font-family: 'Inter', system-ui, sans-serif;
}
.block-container { padding: 1.25rem 1.25rem 2rem !important; max-width: 1480px; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stHorizontalBlock"] { gap: 10px; }
.stApp [data-testid="stVerticalBlock"] { gap: 10px; }

/* ── Top bar ──────────────────────────────────────── */
.topbar { display: flex; align-items: center; justify-content: space-between; padding: .1rem .35rem .25rem; }
.brand { display: flex; align-items: baseline; gap: .6rem; color: #111; }
.brand b { font-size: 1.05rem; font-weight: 600; letter-spacing: -.01em; }
.brand span { font-family: var(--mono); font-size: .72rem; color: #777; }
.top-pills { display: flex; gap: .4rem; }
.top-pill {
  display: inline-flex; align-items: center; gap: .45rem; background: #fff; color: #222;
  border-radius: 999px; padding: .38rem .8rem; font-size: .8rem; text-decoration: none !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.06);
}
.stApp a.top-pill { color: #222; }
.top-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); }

/* ── Tabs as a centred pill switcher (role selectors work across Streamlit versions) ── */
.stTabs [role="tablist"] {
  justify-content: center; gap: 2px; background: #fff; width: fit-content; margin: -2.75rem auto 0;
  border: none; border-radius: 999px; padding: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
  position: relative; z-index: 5;
}
.stTabs [role="tab"] {
  height: auto; margin: 0; padding: .42rem 1.1rem; border-radius: 999px; background: transparent;
  color: #333; transition: background .15s;
}
.stTabs [role="tab"] p { font-size: .88rem; color: inherit; margin: 0; }
.stTabs [role="tab"]:hover { background: #f1f1f1; }
.stTabs [role="tab"][aria-selected="true"] { background: #111; color: #fff; font-weight: 500; }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [role="tabpanel"] { padding-top: 1.1rem; }

/* ── Cards ────────────────────────────────────────── */
[class*="st-key-card_"] {
  background: var(--card); border-radius: 20px; padding: 24px 26px; color: var(--text);
}
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] { gap: 1.25rem; }
.lbl {
  display: flex; align-items: center; gap: .5rem; color: #cfcfcf;
  font-size: .72rem; font-weight: 500; letter-spacing: .08em; text-transform: uppercase;
}
.lbl .lbl-r { margin-left: auto; font-family: var(--mono); text-transform: none; letter-spacing: 0; color: var(--dim); }
.stApp p.hint { color: var(--muted); font-size: .85rem; line-height: 1.5; margin: .35rem 0 0; }
.stApp p.tiny { color: var(--dim); font-size: .78rem; margin: .3rem 0 0; }

/* Big thin numbers */
.big { font-size: 3.6rem; font-weight: 300; letter-spacing: -.035em; line-height: 1; color: var(--text); }
.headline span { letter-spacing: 0; }
.big small { font-size: 1rem; letter-spacing: 0; color: var(--muted); font-weight: 400; margin-left: .5rem; }
.headline { font-size: 2.6rem; font-weight: 300; letter-spacing: -.03em; line-height: 1.1; color: var(--text); margin: .2rem 0 .35rem; }

/* Mono chips */
.chip {
  display: inline-block; font-family: var(--mono); font-size: .68rem; font-weight: 500;
  letter-spacing: .06em; text-transform: uppercase; border-radius: 6px; padding: .18rem .5rem;
  vertical-align: middle;
}
.chip.green { background: rgba(74,222,128,.12); color: var(--green); }
.chip.red   { background: rgba(248,113,113,.12); color: var(--red); }
.chip.grey  { background: #232323; color: var(--muted); }

/* Gradient probability bar */
.pbar { position: relative; height: 6px; background: #2a2a2a; border-radius: 999px; margin: 1.1rem 0 .55rem; }
.pbar-fill {
  height: 100%; border-radius: 999px;
  background-image: linear-gradient(90deg, #f472b6, #fb923c, #facc15, #4ade80);
}
.pbar-mark { position: absolute; top: -5px; width: 2px; height: 16px; background: #666; border-radius: 1px; }
.pbar-meta { display: flex; justify-content: space-between; color: var(--muted); font-size: .78rem; margin-bottom: .6rem; }

/* Row lists (factors, checklists, model card) */
.rows { margin-top: .75rem; }
.row {
  display: flex; align-items: center; gap: .9rem; padding: .7rem .6rem;
  border-bottom: 1px solid var(--line);
}
.row:last-child { border-bottom: none; }
.row.hl { background: #1d1d1d; border-radius: 12px; border-bottom-color: transparent; }
.row-ic {
  flex: none; width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
  font-size: .78rem; font-weight: 600;
}
.row-ic.up   { background: #173823; color: var(--green); }
.row-ic.down { background: #3a1717; color: var(--red); }
.row-ic.flat { background: #242424; color: var(--muted); }
.row-main { flex: 1; min-width: 0; }
.row-t { font-size: .92rem; font-weight: 500; color: var(--text); }
.row-s { font-size: .8rem; color: var(--muted); margin-top: 1px; }
.row-v { font-family: var(--mono); font-size: .82rem; white-space: nowrap; }
.row-v.up { color: var(--green); } .row-v.down { color: var(--red); } .row-v.flat { color: var(--dim); }

/* Stat grid */
.stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.4rem 1rem; margin-top: 1.1rem; }
.stats.four { grid-template-columns: repeat(4, 1fr); }
.stat-n { font-size: 2.3rem; font-weight: 300; letter-spacing: -.03em; line-height: 1; color: var(--text); }
.stat-l { font-size: .8rem; color: var(--muted); margin-top: .35rem; line-height: 1.35; }

/* Bulleted findings */
.dots { list-style: none; padding: 0; margin: .9rem 0 0; }
.dots li { display: flex; gap: .7rem; align-items: baseline; padding: .45rem 0; font-size: .92rem; color: var(--text); }
.dots li::before { content: ''; flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--c, var(--green)); transform: translateY(-2px); }
.dots li span { color: var(--muted); }

/* Confusion matrix */
.cm { display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; margin-top: 1rem; font-size: .78rem; color: var(--muted); }
.cm div { display: flex; align-items: center; }
.cm .cell { background: var(--card-2); border-radius: 12px; padding: .9rem 1rem; flex-direction: column; align-items: flex-start; }
.cm .cell b { font-size: 1.7rem; font-weight: 300; color: var(--text); }
.cm .cell.ok b { color: var(--green); } .cm .cell.bad b { color: var(--red); }

/* ── Widgets inside cards ─────────────────────────── */
[class*="st-key-card_"] label p, [class*="st-key-card_"] [data-testid="stWidgetLabel"] p {
  color: var(--muted) !important; font-size: .8rem !important;
}
[class*="st-key-card_"] [data-baseweb="select"] > div,
[class*="st-key-card_"] [data-baseweb="input"] { background: var(--card-2); border-color: #2a2a2a; border-radius: 10px; }
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stBaseButton-pillsActive"] {
  background: #ededed !important; border-color: #ededed !important;
}
[data-testid="stBaseButton-segmented_controlActive"] p, [data-testid="stBaseButton-pillsActive"] p { color: #111 !important; }
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-pills"] { border-color: #2c2c2c !important; }
[class*="st-key-card_"] .stTextInput { margin-top: .5rem; }
[data-testid="stSliderThumbValue"] { font-family: var(--mono); font-size: .75rem; }
[class*="st-key-card_"] [data-testid="stExpander"] details { border: none; background: transparent; }
[class*="st-key-card_"] [data-testid="stExpander"] summary p { color: var(--muted); font-size: .85rem; }
[data-testid="stElementToolbar"] { display: none; }

[data-testid="stSpinner"] p, .stApp > div [data-testid="stAlert"] p { color: #333; }
[class*="st-key-card_"] iframe { border-radius: 14px; }

.foot { text-align: center; color: #888; font-size: .78rem; padding: .75rem 0 0; }
.foot a { color: #444; }

@media (max-width: 900px) {
  .stats.four { grid-template-columns: repeat(2, 1fr); }
  .stTabs [role="tablist"] { margin-top: .5rem; flex-wrap: wrap; }
  .top-pills { display: none; }
}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates["minimal"] = go.layout.Template(layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", color=MUTED, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[GREEN, BLUE, AMBER, PINK, RED],
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#2a2a2a", ticks="",
                   title=dict(font=dict(color=DIM, size=11))),
        yaxis=dict(showgrid=True, gridcolor="#232323", zeroline=False, ticks="",
                   title=dict(font=dict(color=DIM, size=11))),
        legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)", title=None,
                    font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#222", bordercolor="#333", font=dict(color=TEXT, family="Inter")),
        bargap=0.25,
    ))
    pio.templates.default = "minimal"


def card(key):
    """A dark rounded card; use as a context manager."""
    return st.container(key=f"card_{key}")


def chart(fig, height=None):
    # Streamlit fills in theme backgrounds unless the figure sets them itself
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def md(markup):
    st.markdown(markup, unsafe_allow_html=True)


def label(icon_name, text, right=""):
    md(f'<div class="lbl">{icon(icon_name)}<span>{text}</span><span class="lbl-r">{right}</span></div>')


def hint(text):
    md(f'<p class="hint">{text}</p>')


def verdict_chip(prob):
    if prob > 0.5:
        return '<span class="chip green">Likely survived</span>'
    return '<span class="chip red">Likely perished</span>'


def prob_bar(prob, marker=None, left="", right=""):
    width = max(prob * 100, 1.5)
    mark = f'<div class="pbar-mark" style="left:{marker * 100:.1f}%"></div>' if marker is not None else ""
    return (f'<div class="pbar"><div class="pbar-fill" style="width:{width:.1f}%;'
            f'background-size:{10000 / width:.0f}% 100%"></div>{mark}</div>'
            f'<div class="pbar-meta"><span>{left}</span><span>{right}</span></div>')


def row(title, sub="", value="", tone="flat", glyph=None, highlight=False):
    glyph = glyph or {"up": "↑", "down": "↓", "flat": "•"}[tone]
    return (f'<div class="row{" hl" if highlight else ""}">'
            f'<div class="row-ic {tone}">{glyph}</div>'
            f'<div class="row-main"><div class="row-t">{title}</div>'
            f'{f"<div class=row-s>{sub}</div>" if sub else ""}</div>'
            f'<div class="row-v {tone}">{value}</div></div>')


def stat(number, text):
    return f'<div><div class="stat-n">{number}</div><div class="stat-l">{text}</div></div>'


def esc(text):
    return html.escape(str(text))
