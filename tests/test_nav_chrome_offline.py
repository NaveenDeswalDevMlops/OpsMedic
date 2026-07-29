# tests/test_nav_chrome_offline.py
"""Offline tests for the shared navigation chrome.

Covers the three sidebar/header defects fixed in this build:

  1. the breadcrumb topbar was rendered only by app.py, so every
     pages/*.py shipped without a header -> render_page_chrome() now
     renders both halves and is asserted to emit .opm-topbar;
  2. sidebar sections were a flat, crowded stack -> section_header()
     wraps each in a collapsible st.expander;
  3. long conversation titles showed their MIDDLE, because the
     nowrap/ellipsis rules sat on the flex button instead of the inner
     <p> -> asserted against the DOM replica in
     tests/test_theme_selectors_offline.py and by CSS assertions here.

Streamlit is replaced with a recording stub, so this exercises the real
control flow of ui/nav.py without a browser or a Streamlit runtime.

Run:  python tests/test_nav_chrome_offline.py
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------- recording Streamlit stub
CALLS: list[tuple[str, tuple, dict]] = []


class _Ctx:
    """Stands in for st.sidebar and st.expander (both context managers)."""

    def __init__(self, name: str = "ctx") -> None:
        self.name = name

    def __enter__(self):
        CALLS.append((f"enter:{self.name}", (), {}))
        return self

    def __exit__(self, *exc) -> bool:
        CALLS.append((f"exit:{self.name}", (), {}))
        return False


def _install_stub() -> types.ModuleType:
    st = types.ModuleType("streamlit")

    def markdown(body, *a, **k):
        CALLS.append(("markdown", (body,), k))

    def button(label, *a, **k):
        CALLS.append(("button", (label,), k))
        return False

    def page_link(target, *a, **k):
        CALLS.append(("page_link", (target,), k))

    def expander(label, expanded: bool = True, *a, **k):
        CALLS.append(("expander", (label,), {"expanded": expanded}))
        return _Ctx(f"expander:{label}")

    def caption(body, *a, **k):
        CALLS.append(("caption", (body,), k))

    st.markdown = markdown
    st.button = button
    st.page_link = page_link
    st.expander = expander
    st.caption = caption
    st.sidebar = _Ctx("sidebar")
    sys.modules["streamlit"] = st
    return st


# Install BEFORE importing the modules under test so they bind the stub.
_install_stub()

from ui import nav, theme  # noqa: E402


def _rules(css: str) -> list[tuple[str, str]]:
    """Crude CSS rule splitter -> [(selector, body), ...].

    String slicing on selectors proved brittle (it silently picked up a
    duplicate rule), so the CSS assertions below query parsed rules.
    """
    import re

    out = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        out.append((" ".join(match.group(1).split()), match.group(2)))
    return out


def _rule_bodies(css: str, *needles: str) -> list[str]:
    """Bodies of every rule whose selector contains all `needles`."""
    return [body for sel, body in _rules(css)
            if all(n in sel for n in needles)]


def _reset() -> None:
    CALLS.clear()


def _markdowns() -> str:
    return "\n".join(str(a[0]) for name, a, _ in CALLS if name == "markdown")


def _expander_labels() -> list[str]:
    return [a[0] for name, a, _ in CALLS if name == "expander"]


# ------------------------------------------------- defect 1: missing topbar
def test_page_chrome_renders_the_topbar():
    _reset()
    nav.render_page_chrome(active_label="📡  Monitor")
    html = _markdowns()
    assert 'class="opm-topbar"' in html, "no topbar emitted by page chrome"
    assert 'class="opm-crumb"' in html
    assert "Monitor" in html


def test_page_chrome_renders_sidebar_and_topbar_together():
    _reset()
    nav.render_page_chrome(active_label="📊  Metrics")
    names = [n for n, _, _ in CALLS]
    assert "enter:sidebar" in names, "sidebar not rendered"
    assert "exit:sidebar" in names
    # the topbar must be emitted AFTER the sidebar block closes, i.e. into
    # the main area rather than inside the sidebar
    html_calls = [i for i, (n, a, _) in enumerate(CALLS)
                  if n == "markdown" and "opm-topbar" in str(a[0])]
    assert html_calls, "no topbar markdown"
    assert html_calls[0] > names.index("exit:sidebar"), \
        "topbar rendered inside the sidebar"


def test_every_crumb_label_is_mapped():
    for label, _ in nav._PAGES:
        crumb = nav.crumb_for(label)
        assert crumb and "  " not in crumb, label
    assert nav.crumb_for("💬  Chat") == "Ask-AI"
    assert nav.crumb_for("unknown label") == "unknown label"


def test_explicit_page_name_overrides_the_crumb_map():
    _reset()
    nav.render_page_chrome(active_label="📡  Monitor", page="System Health")
    assert "System Health" in _markdowns()


def test_topbar_markup_is_nowrap_safe():
    """The header wrapped and clipped at high browser zoom."""
    css = theme.CSS
    bars = _rule_bodies(css, ".opm-topbar")
    assert bars, "no .opm-topbar rule"
    assert any("flex-wrap: nowrap" in b and "min-height" in b for b in bars)
    crumbs = _rule_bodies(css, ".opm-crumb")
    assert crumbs, "no .opm-crumb rule"
    assert any("white-space: nowrap" in b and "text-overflow: ellipsis" in b
               for b in crumbs)


# ------------------------------------- defect 2: collapsible sidebar sections
def test_workspace_nav_is_collapsible_by_default():
    _reset()
    nav.render_workspace_nav(active_label="💬  Chat")
    assert "Workspace" in _expander_labels()


def test_workspace_nav_can_stay_flat():
    _reset()
    nav.render_workspace_nav(active_label="💬  Chat", collapsible=False)
    assert _expander_labels() == []


def test_section_header_returns_a_context_manager():
    _reset()
    with nav.section_header("Recents", expanded=False):
        pass
    assert ("expander", ("Recents",), {"expanded": False}) in CALLS


def test_sidebar_nav_wraps_navigate_section():
    _reset()
    nav.render_sidebar_nav(active_label="🎯  Fine-tune")
    labels = _expander_labels()
    assert "Workspace" in labels and "Navigate" in labels


def test_active_page_is_a_button_and_others_are_links():
    _reset()
    nav.render_workspace_nav(active_label="📡  Monitor", collapsible=False)
    buttons = [a[0] for n, a, _ in CALLS if n == "button"]
    links = [a[0] for n, a, _ in CALLS if n == "page_link"]
    assert buttons == ["📡  Monitor"]
    assert "app.py" in links and "pages/3_Monitor.py" not in links


def test_expander_summary_is_styled_as_a_section_label():
    css = theme.CSS
    bodies = _rule_bodies(css, "stSidebar", "stExpander", "summary p")
    assert bodies, "no styled expander summary rule"
    assert any("text-transform: uppercase" in b and "font-weight: 700" in b
               for b in bodies)


# --------------------------------- defect 3: conversation title truncation
def test_sidebar_label_rules_target_the_inner_paragraph():
    """Rules on the flex BUTTON clipped both ends; they must hit the <p>."""
    css = theme.CSS
    # the ellipsis/nowrap must be declared on a rule that selects the <p>
    para = [b for b in _rule_bodies(css, "stSidebar", "button p")
            if "text-overflow: ellipsis" in b and "white-space: nowrap" in b]
    assert para, "truncation rules do not target the label paragraph"
    # and the button itself must pack its label to the start, not centre it
    btn = [b for b in _rule_bodies(css, "stSidebar", "button")
           if "justify-content: flex-start" in b and "text-align: left" in b]
    assert btn, "sidebar button still centres its label"
    # both selector forms Streamlit may emit are covered
    assert 'section[data-testid="stSidebar"] .stButton>button p' in css
    assert 'section[data-testid="stSidebar"] div[data-testid="stButton"]>button p' in css


def test_no_duplicate_sidebar_button_base_rule():
    """Two competing base rules is what made the earlier assertion lie."""
    css = theme.CSS
    bases = [b for sel, b in _rules(css)
             if sel.startswith('section[data-testid="stSidebar"]')
             and sel.endswith("button")
             and "background: transparent" in b]
    assert len(bases) == 1, f"{len(bases)} sidebar button base rules"


def test_sidebar_metrics_are_size_capped():
    css = theme.CSS
    assert '[data-testid="stMetricValue"]' in css
    assert "overflow-x: hidden" in css


# ------------------------------------------------------- stdlib test runner
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
