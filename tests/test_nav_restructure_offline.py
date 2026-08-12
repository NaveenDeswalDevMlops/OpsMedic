# tests/test_nav_restructure_offline.py
"""Guards for the navigation restructure.

Three changes this locks in:
  1. Fine-tune and Metrics are no longer sidebar destinations - they are
     tabs on the Monitor page. The sidebar holds exactly two entries.
  2. Monitor's tab order puts Golden Signals immediately after Overview.
  3. The "+ New chat" button is gone; the Chat nav entry carries that
     action via render_workspace_nav(on_active_click=...).

Pure source inspection plus one import: no Streamlit runtime needed.
"""
from __future__ import annotations

import ast
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITOR = os.path.join(REPO, "pages", "3_Monitor.py")
APP = os.path.join(REPO, "app.py")
NAV = os.path.join(REPO, "ui", "nav.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _nav_assign(name: str):
    """Evaluate a literal module-level assignment in ui/nav.py.

    Parsed rather than imported: ui/nav.py imports Streamlit at module
    scope, and this suite must stay runnable without it.
    """
    tree = ast.parse(_read(NAV))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in ui/nav.py")


def _nav_func_params(name: str) -> list[str]:
    """Argument names of a top-level function in ui/nav.py."""
    tree = ast.parse(_read(NAV))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            a = node.args
            return ([p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
                    + [p.arg for p in a.kwonlyargs])
    raise AssertionError(f"{name}() not found in ui/nav.py")


def _monitor_tab_labels() -> list[str]:
    """Extract the st.tabs([...]) label list from the Monitor page."""
    tree = ast.parse(_read(MONITOR))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tabs"
                and node.args
                and isinstance(node.args[0], (ast.List, ast.Tuple))):
            return [
                el.value for el in node.args[0].elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            ]
    raise AssertionError("no st.tabs([...]) call found in the Monitor page")


# ------------------------------------------------- 1. sidebar destinations
def test_sidebar_has_exactly_chat_and_monitor():
    _PAGES = _nav_assign("_PAGES")
    labels = [lbl for lbl, _ in _PAGES]
    assert len(_PAGES) == 2, f"expected 2 sidebar entries, got {labels}"
    assert any("Chat" in lbl for lbl in labels)
    assert any("Monitor" in lbl for lbl in labels)


def test_finetune_and_metrics_are_not_sidebar_entries():
    _PAGES = _nav_assign("_PAGES")
    labels = " ".join(lbl for lbl, _ in _PAGES)
    assert "Fine-tune" not in labels
    assert "Metrics" not in labels


def test_every_sidebar_target_file_actually_exists():
    """A stale page_link target raises at runtime, so check the paths."""
    for label, target in _nav_assign("_PAGES"):
        assert os.path.exists(os.path.join(REPO, target)), \
            f"{label} points at missing file {target}"


def test_crumbs_cover_every_sidebar_label():
    _CRUMBS = _nav_assign("_CRUMBS")
    for label, _ in _nav_assign("_PAGES"):
        assert label in _CRUMBS, f"no breadcrumb for {label}"


def test_deleted_page_files_are_really_gone():
    for gone in ("pages/1_LLMOps_Dashboard.py",
                 "pages/2_Finetune_Comparison.py",
                 "pages/2_Model_Card.py"):
        assert not os.path.exists(os.path.join(REPO, gone)), \
            f"{gone} still present - it would reappear in the sidebar"


def test_monitor_is_the_only_remaining_page():
    entries = sorted(
        f for f in os.listdir(os.path.join(REPO, "pages"))
        if f.endswith(".py") and not f.startswith("__")
    )
    assert entries == ["3_Monitor.py"], entries


# ------------------------------------------------- 2. tab order
def test_golden_signals_comes_immediately_after_overview():
    labels = _monitor_tab_labels()
    assert labels[0] == "Overview", labels
    assert labels[1] == "Golden Signals", labels


def test_metrics_and_finetune_are_monitor_tabs():
    labels = _monitor_tab_labels()
    assert "Metrics" in labels
    assert "Fine-tune" in labels


def test_tab_count_matches_the_number_of_bodies():
    """Every declared tab must have a `with page_tabs[i]:` block."""
    src = _read(MONITOR)
    labels = _monitor_tab_labels()
    indices = sorted(int(m) for m in re.findall(r"with page_tabs\[(\d+)\]:", src))
    assert indices == list(range(len(labels))), (
        f"{len(labels)} tabs declared but bodies are {indices}"
    )


def test_tab_bodies_are_each_defined_once():
    src = _read(MONITOR)
    indices = [int(m) for m in re.findall(r"with page_tabs\[(\d+)\]:", src)]
    assert len(indices) == len(set(indices)), f"duplicate tab bodies: {indices}"


def test_monitor_imports_the_two_moved_panels():
    src = _read(MONITOR)
    assert "metrics_dashboard" in src
    assert "model_card_panel" in src


# ------------------------------------------------- 3. new-chat folding
def test_new_chat_button_is_gone_from_app():
    src = _read(APP)
    assert 'key="newchat"' not in src
    assert "＋  New chat" not in src


def test_workspace_nav_accepts_an_active_click_callback():
    params = _nav_func_params("render_workspace_nav")
    assert "on_active_click" in params
    assert "active_help" in params


def test_app_passes_a_new_chat_callback_to_the_nav():
    src = _read(APP)
    assert "on_active_click" in src
    assert "start_new_chat" in src


def test_start_new_chat_still_clears_conversation_state():
    """The action itself must survive the move out of the button."""
    src = _read(APP)
    body = src[src.index("def start_new_chat"):]
    body = body[:body.index("def load_chat")]
    for key in ("conversation_id", "messages", "pending_attachment"):
        assert key in body, f"start_new_chat no longer resets {key}"
