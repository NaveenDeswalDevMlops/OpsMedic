# ui/nav.py
"""Shared navigation chrome for OpsMedic (chat + all monitor pages).

Two entry points:

  render_page_chrome(active_label, page)  - for pages/*.py. Renders the
      sidebar (brand, collapsible Workspace nav, profile) AND the
      main-area breadcrumb topbar, so no page can accidentally ship
      without the header. app.py builds a richer sidebar itself and calls
      section_header()/theme.topbar() directly.

  section_header(label, expanded)         - the collapsible sidebar
      section used for Workspace / Recents / Knowledge Base. Returns the
      expander context manager; theme.py styles the summary row like the
      old flat .opm-navlabel, so collapsing costs no visual clarity.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from src import config
from ui import theme

# label -> page file path for st.page_link
_PAGES = [
    ("💬  Chat", "app.py"),
    ("📡  Monitor", "pages/3_Monitor.py"),
    ("🎯  Fine-tune", "pages/2_Finetune_Comparison.py"),
    ("📊  Metrics", "pages/1_LLMOps_Dashboard.py"),
]

#: breadcrumb page name per nav label, so the topbar text is defined once
_CRUMBS = {
    "💬  Chat": "Ask-AI",
    "📡  Monitor": "Monitor",
    "🎯  Fine-tune": "Fine-tune",
    "📊  Metrics": "Metrics",
}


def render_brand() -> None:
    st.markdown(f'<div class="opm-brand">🩺 {config.PRODUCT_NAME}</div>',
                unsafe_allow_html=True)


def section_header(label: str, expanded: bool = True) -> Any:
    """A collapsible sidebar section. Use as a context manager:

        with nav.section_header("Recents"):
            ...

    Styled by theme.py to look like the previous static section label.
    """
    return st.expander(label, expanded=expanded)


def render_workspace_nav(active_label: str, collapsible: bool = True,
                         expanded: bool = True) -> None:
    """Workspace buttons. `active_label` is the current page's label so it
    renders highlighted; the others are page links that navigate away."""
    def _body() -> None:
        for label, target in _PAGES:
            is_active = label == active_label
            wrap = "opm-nav opm-nav-active" if is_active else "opm-nav"
            st.markdown(f'<div class="{wrap}">', unsafe_allow_html=True)
            if is_active:
                # current page: a non-navigating button (visual highlight only)
                st.button(label, use_container_width=True,
                          key=f"navbtn_{label}", disabled=False)
            else:
                st.page_link(target, label=label)
            st.markdown('</div>', unsafe_allow_html=True)

    if collapsible:
        with section_header("Workspace", expanded=expanded):
            _body()
    else:
        _body()


def crumb_for(active_label: str) -> str:
    """Breadcrumb page name for a nav label."""
    return _CRUMBS.get(active_label, active_label.strip())


def render_sidebar_nav(active_label: str, show_profile: bool = True) -> None:
    """Sidebar header used on the monitor/finetune/metrics pages.

    The chat app builds its own richer sidebar (with recents); the other
    pages use this so they always have the brand + a Back-to-Chat link.
    """
    with st.sidebar:
        render_brand()
        render_workspace_nav(active_label)
        with section_header("Navigate", expanded=True):
            st.page_link("app.py", label="←  Back to Chat")
        if show_profile:
            theme.profile_card()


def render_page_chrome(active_label: str, page: str | None = None,
                       section: str = "Overview",
                       show_profile: bool = True) -> None:
    """Sidebar + breadcrumb topbar in one call.

    Every pages/*.py calls this instead of render_sidebar_nav so the
    topbar can never be omitted on a page again (previously only app.py
    rendered it, which is why the header vanished off the dashboards).
    """
    render_sidebar_nav(active_label, show_profile=show_profile)
    theme.topbar(section, page or crumb_for(active_label))
