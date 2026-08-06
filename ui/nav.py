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
#
# Fine-tune and Metrics used to be separate entries here, pointing at
# pages/2_*.py and pages/1_*.py. They are now tabs on the Monitor page
# (ui/model_card_panel.py and ui/metrics_dashboard.py), so the sidebar
# holds only the two genuine destinations: the chat, and everything
# operational. Streamlit's own page nav is hidden in theme.py, so
# deleting those page files removed them from the sidebar entirely.
_PAGES = [
    ("💬  Chat", "app.py"),
    ("📡  Monitor", "pages/3_Monitor.py"),
]

#: breadcrumb page name per nav label, so the topbar text is defined once
_CRUMBS = {
    "💬  Chat": "Ask-AI",
    "📡  Monitor": "Monitor",
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
                         expanded: bool = True,
                         on_active_click: Any = None,
                         active_help: str | None = None) -> None:
    """Workspace buttons. `active_label` is the current page's label so it
    renders highlighted; the others are page links that navigate away.

    `on_active_click` is a zero-argument callable invoked when the user
    clicks the button for the page they are already on. That is what
    powers "Chat" doubling as the new-chat action: there is no separate
    "+ New chat" button any more, because clicking the destination you
    are already at has no other sensible meaning.
    """
    def _body() -> None:
        for label, target in _PAGES:
            is_active = label == active_label
            wrap = "opm-nav opm-nav-active" if is_active else "opm-nav"
            st.markdown(f'<div class="{wrap}">', unsafe_allow_html=True)
            if is_active:
                # current page: clicking it triggers the page's own
                # "restart" action if one was supplied, otherwise it is
                # an inert visual highlight.
                clicked = st.button(label, use_container_width=True,
                                    key=f"navbtn_{label}",
                                    help=active_help)
                if clicked and callable(on_active_click):
                    on_active_click()
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
    """Sidebar header used on the Monitor page.

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
