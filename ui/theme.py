# ui/theme.py
"""Enterprise UI kit for OpsMedic — CSS, KPI tiles, Plotly gauges, chat chrome.

The chat styling is ported from the reference "Ai-Manager" React app
(Montserrat, light-grey sidebar rgb(246,246,246), uppercase collapsible
nav categories, profile card, breadcrumb header with icon buttons,
floating speaker hero with a bob animation, pill-chip input bar). Adapted
to Streamlit's DOM so the chat app matches that look without a rewrite.

Rendering only; no model or metrics logic lives here.
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import Any

import streamlit as st

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@lru_cache(maxsize=8)
def _asset_b64(name: str) -> str:
    """Return a data-URI for a bundled asset ('' if missing)."""
    path = os.path.join(_ASSETS, name)
    if not os.path.isfile(path):
        return ""
    ext = "png" if name.endswith(".png") else "jpeg"
    with open(path, "rb") as fh:
        return f"data:image/{ext};base64," + base64.b64encode(fh.read()).decode()

# palette (aligned to the reference + a Genpact-ish deep-blue accent)
PRIMARY = "#3B82F6"
DEEPBLUE = "#02028B"        # reference "Chat with AI" pill / accents
OK = "#10B981"
WARN = "#F59E0B"
ERR = "#EF4444"
VIOLET = "#8B5CF6"
INK = "#111827"
MUTED = "#7F7E7E"
PANEL = "rgb(246, 246, 246)"

# ---------------------------------------------------------------- scoping
# Streamlit wraps EVERY vertical block in stVerticalBlockBorderWrapper,
# so a bare :has(#anchor) also matches all of the anchor's ancestors —
# including the root main block, which leaked the composer's button rules
# onto every button on the page (chips rendered as dark circles).
#
# These selectors instead demand that the anchor sit in a MARKDOWN element
# that is the block's own first child. An ancestor cannot satisfy that:
# the root block's first child is the breadcrumb, and chat_area's is the
# hero. (An earlier :nth-child(-n+2) form still leaked, because chat_area
# is the root's 2nd child and holds the chips anchor deep inside — see
# tests/test_theme_selectors_offline.py, which runs these against a
# replica of Streamlit's DOM.)
_ANCHOR_CHAIN = '> div:first-child > div[data-testid="stMarkdown"]'
_COMPOSER = ('div[data-testid="stVerticalBlock"]'
             f':has({_ANCHOR_CHAIN} #opm-composer-anchor)')
_CHIPS = ('div[data-testid="stVerticalBlock"]'
          f':has({_ANCHOR_CHAIN} #opm-chips-anchor)')

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');

html, body, [class*="css"], .stApp {{ font-family: 'Montserrat', sans-serif; }}
.stApp {{ background: #FFFFFF; }}

/* ---------- Sidebar as the reference "aside" ---------- */
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid rgba(128,128,128,.13);
}}
section[data-testid="stSidebar"] * {{ color: {INK}; font-family:'Montserrat',sans-serif; }}
/* the sidebar button base rule is consolidated further down, with the
   truncation fix — one selector, one place */
section[data-testid="stSidebar"] .stButton>button:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"]>button:hover {{ color: {INK}; background: rgba(0,0,0,.03); }}
.opm-brand {{ font-size: 28px; font-weight: 800; color:{INK}; margin: 2px 0 10px 2px; }}
.opm-navlabel {{ font-size: 11px; font-weight: 700; color:{INK};
                 text-transform: uppercase; letter-spacing:.04em;
                 margin: 14px 0 4px 2px;
                 border-bottom: 1px solid rgba(128,128,128,.25); padding-bottom: 6px; }}

/* hide Streamlit's auto multipage nav (we render custom nav buttons) */
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {{ display: none; }}

/* Claude-style workspace nav buttons (Chat / Monitor / Fine-tune ...) */
.opm-nav .stButton>button {{
    border: 1px solid rgba(128,128,128,.22) !important; border-radius: 10px !important;
    background: #fff !important; color:{INK} !important; font-weight:600 !important;
    font-size: 13px !important; text-align:left !important; padding: 9px 12px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,.05) !important; margin-bottom: 6px !important;
}}
.opm-nav .stButton>button:hover {{ border-color:{PRIMARY} !important;
    background: rgba(59,130,246,.04) !important; }}
.opm-nav-active .stButton>button {{ border-color:{PRIMARY} !important;
    background: rgba(59,130,246,.08) !important; }}

/* Primary-action pill. Was the "+ New chat" button; that button is gone
   (the Chat nav entry starts a new conversation), so this rule is kept
   only for the active Workspace nav button, which reuses the styling. */
.opm-nav-active button {{
    border: 2px solid {DEEPBLUE} !important; color: {DEEPBLUE} !important;
    background: #fff !important; border-radius: 20px !important;
    font-weight: 700 !important; text-align: center !important;
    padding: 6px 14px !important;
}}
.opm-nav-active button:hover {{ background: rgba(2,2,139,.05) !important; }}

/* profile card at the sidebar bottom */
.opm-profile {{ background:#fff; border-radius:15px; padding:14px 12px;
                margin-top:12px; text-align:center; }}
.opm-profile h6 {{ font-size:14px; font-weight:700; margin:0; }}
.opm-profile p {{ font-size:11px; color:{MUTED}; margin:3px 0 0 0; }}

/* ---------- Main column ---------- */
.block-container {{ padding-top: 1.2rem; padding-bottom: 6rem; max-width: 1180px; }}

/* breadcrumb header bar (reference .header nav) */
.opm-topbar {{ display:flex; align-items:center; justify-content:space-between;
               flex-wrap: nowrap; gap: 12px; min-height: 34px;
               border-bottom: 2px solid rgba(128,128,128,.16);
               padding: 4px 2px 10px 2px; margin-bottom: 8px; }}
.opm-crumb {{ font-size: 13px; line-height: 1.4; color:{MUTED};
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
              min-width: 0; }}
.opm-crumb b {{ color:{INK}; font-weight:700; font-size: 13px; }}
.opm-topicons {{ flex: 0 0 auto; white-space: nowrap; }}
.opm-topicons i {{ margin: 0 6px; font-size:14px; color:{INK}; padding:7px;
                   border-radius:50%; cursor:pointer; }}
.opm-topicons i:hover {{ background: rgba(197,197,197,.4); }}

/* floating speaker hero — real reference image, transparent bg (reference .figure-img) */
.opm-figure {{ text-align:center; position:relative; margin-top: 3vh; }}
.opm-figure .device-wrap {{ position: relative; display:inline-block;
    animation: opmbob 2s ease-in-out infinite alternate; }}
.opm-figure img.device {{ width: 200px; height: auto; display:block; }}
/* live clock sits ON the painted dark display panel, bright like a real screen */
.opm-figure .opm-clock {{
    position:absolute; left:50%; top:40%; transform:translate(-50%,-50%);
    color:#f4f6fb; font-weight:700; font-size:24px; letter-spacing:2px;
    line-height:1.0; text-align:center;
    font-variant-numeric: tabular-nums; font-family:'Montserrat',sans-serif;
    text-shadow: 0 0 6px rgba(255,255,255,.25);
}}
.opm-figure h1 {{ font-size: 36px; font-weight:400; color:{MUTED}; margin-top: 6px; }}
.opm-figure h1 b {{ color:{INK}; font-weight:800; }}
@keyframes opmbob {{ 0%{{transform:translateY(-12px)}} 100%{{transform:translateY(12px)}} }}

/* ---------- suggested prompt chips ----------
   Scoped with _CHIPS (see above): an invisible anchor span in the
   container's first child. A wrapper <div> from st.markdown cannot be
   used, because Streamlit auto-closes it inside its own markdown
   container and it therefore contains none of the widgets that follow. */
div[data-testid="stElementContainer"]:has(#opm-chips-anchor),
div[data-testid="stElementContainer"]:has(#opm-composer-anchor) {{
    display:none !important;
}}
{_CHIPS} {{ max-width:900px; margin:0 auto !important; gap:.5rem !important; }}
{_CHIPS} button {{
    border:1px solid rgba(128,128,128,.28) !important; border-radius:14px !important;
    background:#fff !important; color:{INK} !important; font-size:13px !important;
    font-weight:500 !important; text-align:left !important; padding:10px 14px !important;
    box-shadow:0 1px 2px rgba(15,23,42,.05) !important; line-height:1.35 !important;
    width:100% !important; height:auto !important; min-height:0 !important;
    border-radius:14px !important; white-space:normal !important;
}}
{_CHIPS} button:hover {{ border-color:{PRIMARY} !important; color:{PRIMARY} !important; }}
{_CHIPS} button p {{ font-size:13px !important; text-align:left !important; }}

/* chat bubbles */
div[data-testid="stChatMessage"] {{ border-radius: 14px; }}

/* ================= COMPOSER (reference chat input bar) =================
   Layout, top to bottom, inside ONE plain st.container():
     hint line · text field · staged-context chips · control row
   Control row: pills on the left (Attach / Search / Writing Styles /
   Voice) and a dark circular send button flush right.

   The box border is drawn on the block itself rather than using
   st.container(border=True), so there is no default border to override.
   :has() needs Chrome 105+ / Safari 15.4+ / Firefox 121+. */
{_COMPOSER} {{
    border: 1px solid rgba(128,128,128,.30) !important;
    border-radius: 26px !important; background:#fff !important;
    padding: 12px 16px 10px 16px !important;
    max-width: 980px; margin: 10px auto 0 auto !important;
    box-shadow: rgba(0,0,0,.05) 0 8px 28px 0, rgba(0,0,0,.05) 0 0 0 1px !important;
    gap: 0.3rem !important;
}}
{_COMPOSER} div[data-testid="stHorizontalBlock"] {{
    gap: 0.4rem !important; align-items: center !important; flex-wrap: nowrap !important;
}}

/* hint line + staged-context chips (attachment, voice transcript) */
.opm-composer-hint {{ font-size:13.5px; font-weight:600; color:{INK};
    margin:0 0 2px 2px; line-height:1.45; }}
.opm-composer-hint span {{ color:{MUTED}; font-weight:500; }}
.opm-composer-chip {{ display:inline-block; font-size:11.5px; color:{MUTED};
    background:#F5F6F8; border-radius:9px; padding:4px 10px; margin:2px 0 0 2px; }}

/* text field: strip baseweb's grey fill and border so it reads as part
   of the box. The border lives on div[data-baseweb="base-input"], NOT on
   the stTextInput wrapper — targeting only the latter left the grey box
   visible. */
{_COMPOSER} div[data-testid="stTextInput"] > div,
{_COMPOSER} div[data-baseweb="input"],
{_COMPOSER} div[data-baseweb="base-input"],
{_COMPOSER} div[data-testid="stTextInputRootElement"] {{
    border:none !important; box-shadow:none !important;
    background:transparent !important; background-color:transparent !important;
}}
{_COMPOSER} div[data-testid="stTextInput"] input {{
    border:none !important; box-shadow:none !important; outline:none !important;
    background:transparent !important; font-size:15.5px !important;
    padding:4px 2px !important; font-family:'Montserrat',sans-serif !important;
    color:{INK} !important;
}}
{_COMPOSER} div[data-testid="stTextInput"] input::placeholder {{
    color:#9AA0A6 !important;
}}

/* left-hand pills. min-width:0 + ellipsis is what stops a long label
   (e.g. "Step-by-step") from overflowing its column and colliding with
   the next pill. */
{_COMPOSER} [data-testid="stPopover"] {{ min-width:0 !important; width:100% !important; }}
{_COMPOSER} [data-testid="stPopover"] button {{
    border:1px solid rgba(128,128,128,.30) !important; background:#fff !important;
    color:{INK} !important; border-radius:999px !important;
    font-size:12px !important; font-weight:600 !important;
    padding:6px 10px !important; box-shadow:none !important;
    width:100% !important; min-width:0 !important; height:34px !important;
    min-height:34px !important; overflow:hidden !important;
    text-overflow:ellipsis !important; white-space:nowrap !important;
    display:flex !important; align-items:center !important;
    justify-content:center !important;
}}
{_COMPOSER} [data-testid="stPopover"] button p {{
    font-size:12px !important; font-weight:600 !important; overflow:hidden !important;
    text-overflow:ellipsis !important; white-space:nowrap !important; margin:0 !important;
}}
{_COMPOSER} [data-testid="stPopover"] button:hover {{
    border-color:{PRIMARY} !important; color:{PRIMARY} !important;
}}

/* send button — the only bare st.button inside the composer. The :not()
   guard spares buttons rendered inside popover panels. */
{_COMPOSER} div[data-testid="stButton"] > button:not([data-testid="stPopover"] *),
{_COMPOSER} .stButton > button:not([data-testid="stPopover"] *) {{
    background:{INK} !important; color:#fff !important; border:none !important;
    border-radius:50% !important; width:38px !important; height:38px !important;
    min-height:38px !important; padding:0 !important; font-size:16px !important;
    box-shadow:none !important; margin-left:auto !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
}}
{_COMPOSER} div[data-testid="stButton"] > button:not([data-testid="stPopover"] *):hover,
{_COMPOSER} .stButton > button:not([data-testid="stPopover"] *):hover {{
    background:{PRIMARY} !important; color:#fff !important;
}}

/* ---- sidebar: layout, truncation, collapsible sections ----
   Streamlit renders a button's label in an inner <p> and centres it as a
   flex item. Putting nowrap/overflow/ellipsis on the BUTTON therefore
   clipped the <p> at both ends, which showed the MIDDLE of a long
   conversation title ("rinter on floor 3 shows jobs qu"). The rules must
   land on the <p>, and the button must pack its child to the start. */
section[data-testid="stSidebar"] {{ overflow-x: hidden !important; }}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{ gap: 0.15rem; }}
section[data-testid="stSidebar"] .stButton,
section[data-testid="stSidebar"] div[data-testid="stButton"] {{ margin: 0; }}
section[data-testid="stSidebar"] .stButton>button,
section[data-testid="stSidebar"] div[data-testid="stButton"]>button {{
    background: transparent; border: none; box-shadow: none;
    color: {MUTED}; font-size: 12.5px; font-weight: 500;
    line-height: 1.25; padding: 5px 8px; border-radius: 8px;
    display: flex !important; justify-content: flex-start !important;
    align-items: center !important; text-align: left !important;
    width: 100% !important; overflow: hidden !important;
}}
section[data-testid="stSidebar"] .stButton>button p,
section[data-testid="stSidebar"] .stButton>button div,
section[data-testid="stSidebar"] div[data-testid="stButton"]>button p,
section[data-testid="stSidebar"] div[data-testid="stButton"]>button div,
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {{
    width: 100% !important; text-align: left !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important; margin: 0 !important;
}}
/* the nav buttons keep their own centred-icon look, so re-assert left */
section[data-testid="stSidebar"] .opm-nav .stButton>button,
section[data-testid="stSidebar"] .opm-nav div[data-testid="stButton"]>button {{
    justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    margin: 8px 0 2px 2px; font-size: 10.5px; letter-spacing:.06em; color:{MUTED};
}}
section[data-testid="stSidebar"] hr {{ margin: 8px 0; }}

/* Knowledge-Base metrics were overflowing the sidebar width */
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    font-size: 20px !important; line-height: 1.15 !important;
}}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{
    font-size: 10.5px !important; white-space: normal !important;
    text-overflow: clip !important; color:{MUTED} !important;
}}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{
    padding: 0 !important; overflow: hidden !important;
}}

/* collapsible sections: st.expander styled as the old .opm-navlabel so
   Recents / Knowledge Base / Workspace read as section headers */
section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] details {{
    border: none !important; background: transparent !important;
    box-shadow: none !important; margin: 6px 0 0 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    padding: 4px 2px 6px 2px !important;
    border-bottom: 1px solid rgba(128,128,128,.25) !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
    color: {PRIMARY} !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: .04em !important;
    color: {INK} !important; margin: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
    padding: 4px 0 0 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    margin: 8px 0 2px 2px; font-size: 10.5px; letter-spacing:.06em; color:{MUTED};
}}
section[data-testid="stSidebar"] hr {{ margin: 8px 0; }}

/* ---- KPI tiles / gauges (monitor console) ---- */
.opm-tile {{ background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px;
    padding:16px 18px; box-shadow:0 1px 3px rgba(15,23,42,.06); height:100%; }}
.opm-tile .icon {{ font-size:20px; }}
.opm-tile .label {{ color:{MUTED}; font-size:12.5px; font-weight:600;
    text-transform:uppercase; letter-spacing:.04em; }}
.opm-tile .value {{ color:{INK}; font-size:26px; font-weight:700; line-height:1.1; margin-top:2px; }}
.opm-tile .sub {{ color:{MUTED}; font-size:12px; margin-top:4px; }}
.opm-badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:11.5px; font-weight:700; }}
.opm-eyebrow {{ color:{PRIMARY}; font-weight:700; font-size:12px;
    text-transform:uppercase; letter-spacing:.08em; }}
.opm-title {{ color:{INK}; font-size:26px; font-weight:800; margin:2px 0; }}
.opm-desc {{ color:{MUTED}; font-size:13.5px; margin-bottom:8px; }}

/* Sources cards */
.opm-src-wrap {{ display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 2px 0; }}
.opm-src {{ flex:1 1 210px; min-width:210px; max-width:280px;
    border:1px solid #E5E7EB; border-radius:12px; padding:10px 12px; background:#FBFCFE; }}
.opm-src .sid {{ font-size:11px; color:{PRIMARY}; font-weight:700; }}
.opm-src .stt {{ font-size:13px; color:{INK}; font-weight:600; margin:2px 0; line-height:1.25; }}
.opm-src .sbody {{ font-size:11.5px; color:{MUTED}; line-height:1.35; max-height:64px; overflow:hidden; }}
.opm-src .sscore {{ font-size:11px; color:{MUTED}; margin-top:4px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def topbar(section: str, page: str) -> None:
    """Reference-style breadcrumb header with right-side icon buttons."""
    st.markdown(
        f'''<div class="opm-topbar">
              <div class="opm-crumb">{section} / <b>{page}</b></div>
              <div class="opm-topicons">
                <i class="fa-regular fa-moon" title="Theme"></i>
                <i class="fa-regular fa-bell" title="Notifications"></i>
                <i class="fa-regular fa-comments" title="Chats"></i>
                <i class="fa-solid fa-headset" title="Support"></i>
              </div>
            </div>''',
        unsafe_allow_html=True,
    )


def hero(product_name: str) -> None:
    """Floating speaker hero (real transparent image) with a LIVE clock
    overlay showing the system's local time, and the greeting.

    The speaker image + greeting are drawn with markdown; the ticking
    clock is a tiny components.html iframe positioned over the display
    (st.markdown strips <script>, components.html does not)."""
    import streamlit.components.v1 as components

    img = _asset_b64("speaker.png")
    device = (
        f'<img class="device" src="{img}" alt="speaker"/>' if img
        else '<div style="height:200px"></div>'
    )
    st.markdown(
        f'''<div class="opm-figure">
              <div class="device-wrap">{device}
                <div class="opm-clock" id="opmClockAnchor"></div>
              </div>
              <h1>Hello, what's on <b>your mind?</b></h1>
            </div>''',
        unsafe_allow_html=True,
    )
    # live clock: reach up to the parent doc's anchor and tick it every second
    components.html(
        """
        <script>
          function tick() {
            var doc = window.parent.document;
            var el = doc.getElementById("opmClockAnchor");
            if (!el) return;
            var d = new Date();
            var hh = String(d.getHours()).padStart(2,"0");
            var mm = String(d.getMinutes()).padStart(2,"0");
            el.innerHTML = hh + "<br>" + mm;
          }
          tick(); setInterval(tick, 1000);
        </script>
        """,
        height=0,
    )


def chips_anchor() -> None:
    """Emit the CSS anchor that styles suggestion-chip buttons.

    Call as the FIRST element inside a plain st.container() holding the
    chips: the anchored rules match the block whose first child contains
    this span, which is what keeps them off the rest of the page.
    (The composer's own hint line lives in ui/composer.py, which is why
    the old free-floating input_hint() is gone.)
    """
    st.markdown('<span id="opm-chips-anchor"></span>', unsafe_allow_html=True)


def profile_card(name: str = "Naveen", role: str = "Admin") -> None:
    """Sidebar profile card with the reference avatar."""
    avatar = _asset_b64("boy.png")
    av = (f'<img src="{avatar}" style="width:46px;border-radius:50%"/>' if avatar
          else "")
    st.markdown(
        f'''<div class="opm-profile">
              <h6>How can I help?</h6>
              <p>Report an incident by text or voice</p>
              <div style="display:flex;align-items:center;gap:10px;
                          justify-content:center;margin-top:10px">
                {av}
                <div style="text-align:left">
                  <div style="font-weight:700;font-size:13px">{name}</div>
                  <div style="color:#7F7E7E;font-size:10px">{role}</div>
                </div>
              </div>
            </div>''',
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, desc: str = "") -> None:
    st.markdown(f'<div class="opm-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="opm-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="opm-desc">{desc}</div>', unsafe_allow_html=True)


def status_badge(text: str, state: str = "ok") -> str:
    colors = {"ok": (OK, "#DCFCE7"), "warn": (WARN, "#FEF3C7"),
              "err": (ERR, "#FEE2E2"), "info": (PRIMARY, "#DBEAFE")}
    fg, bg = colors.get(state, colors["info"])
    return (f'<span class="opm-badge" style="color:{fg};background:{bg}">'
            f'{text}</span>')


def tile(icon: str, label: str, value: str, sub: str = "",
         badge_html: str = "") -> None:
    st.markdown(
        f'''<div class="opm-tile"><div class="icon">{icon}</div>
              <div class="label">{label}</div><div class="value">{value}</div>
              <div class="sub">{sub} {badge_html}</div></div>''',
        unsafe_allow_html=True,
    )


def donut_gauge(value: float, label: str, color: str = PRIMARY, vmax: float = 1.0):
    import plotly.graph_objects as go

    pct = 0.0 if vmax == 0 else max(0.0, min(1.0, value / vmax))
    fig = go.Figure(go.Pie(values=[pct, 1 - pct], hole=0.72,
                           marker_colors=[color, "#EEF2F7"], textinfo="none",
                           sort=False, direction="clockwise", showlegend=False))
    fig.add_annotation(
        text=f"<b>{value:.2f}</b><br><span style='font-size:11px;color:{MUTED}'>"
             f"{label}</span>",
        showarrow=False, font=dict(size=18, color=INK))
    fig.update_layout(height=170, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def style_plotly(fig) -> None:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, size=12),
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=True, gridcolor="#EEF2F7")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7")


def latency_color(p95_ms: float) -> str:
    if p95_ms <= 1500:
        return OK
    if p95_ms <= 8000:
        return WARN
    return ERR


def rate_state(rate: float, good_high: bool = False) -> str:
    if good_high:
        return "ok" if rate >= 0.5 else "warn" if rate >= 0.2 else "err"
    return "ok" if rate <= 0.02 else "warn" if rate <= 0.1 else "err"


def source_cards(similar: list[dict[str, Any]]) -> None:
    if not similar:
        return
    st.markdown("**Sources**", unsafe_allow_html=True)
    cards = []
    for t in similar:
        body = str(t.get("description", ""))[:180].replace("<", "&lt;")
        title = str(t.get("title", "")).replace("<", "&lt;")
        cards.append(
            f'''<div class="opm-src">
                  <div class="sid">[{t['ticket_id']}] · {t.get('category','')}</div>
                  <div class="stt">{title}</div>
                  <div class="sbody">{body}…</div>
                  <div class="sscore">similarity {t.get('score', 0):.2f}</div>
                </div>''')
    st.markdown(f'<div class="opm-src-wrap">{"".join(cards)}</div>',
                unsafe_allow_html=True)
