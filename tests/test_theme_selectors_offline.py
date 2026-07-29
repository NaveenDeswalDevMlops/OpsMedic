# tests/test_theme_selectors_offline.py
"""Regression tests for the anchored composer CSS selectors.

Why this file exists: the first version of the composer CSS used
`:has(#opm-composer-anchor)` unscoped. Streamlit wraps EVERY vertical
block in a wrapper element, so that selector matched all of the anchor's
ANCESTORS too — including the root block — and the send-button rule
therefore restyled every button on the page (the suggestion chips
rendered as dark circles).

Reasoning about selectors is what produced that bug, so these tests run
the real selectors from ui/theme.py against a replica of the DOM
Streamlit emits for app.py, using soupsieve (the CSS engine behind
BeautifulSoup, which implements :has(), :nth-child() and complex
:not()). If a future edit re-broadens a scope, this fails loudly.

Skips cleanly when bs4/soupsieve is unavailable.

Run:  python tests/test_theme_selectors_offline.py
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "streamlit" not in sys.modules:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.modules["streamlit"] = types.ModuleType("streamlit")

try:
    import soupsieve as sv
    from bs4 import BeautifulSoup
    _HAVE_SOUP = True
except ImportError:  # pragma: no cover - optional dev dependency
    _HAVE_SOUP = False

from ui import theme  # noqa: E402


def _element(inner: str) -> str:
    """Streamlit wraps each widget/element in an stElementContainer."""
    return f'<div data-testid="stElementContainer">{inner}</div>'


def _markdown(inner: str) -> str:
    return _element(
        '<div data-testid="stMarkdown">'
        f'<div data-testid="stMarkdownContainer">{inner}</div></div>'
    )


def _button(label: str) -> str:
    return _element(
        f'<div class="stButton" data-testid="stButton">'
        f'<button data-testid="stBaseButton-secondary"><p>{label}</p></button>'
        f'</div>'
    )


def _popover(label: str, panel_button: str = "") -> str:
    """Popover trigger plus (worst case) its panel rendered inline.

    Streamlit may portal the panel elsewhere; rendering it inline here is
    the harsher assumption, so the send-button :not() guard gets tested.
    """
    panel = (f'<div data-testid="stPopoverBody">{_button(panel_button)}</div>'
             if panel_button else "")
    return _element(
        f'<div data-testid="stPopover"><button><p>{label}</p></button>'
        f'{panel}</div>'
    )


def _column(inner: str) -> str:
    return (f'<div data-testid="stColumn">'
            f'<div data-testid="stVerticalBlockBorderWrapper">'
            f'<div data-testid="stVerticalBlock">{inner}</div></div></div>')


def _block(inner: str) -> str:
    """A container: border wrapper -> vertical block."""
    return ('<div data-testid="stVerticalBlockBorderWrapper">'
            f'<div data-testid="stVerticalBlock">{inner}</div></div>')


def build_dom(with_chips: bool = True, extra_before: int = 0) -> "BeautifulSoup":
    """Replica of app.py's main area: breadcrumb, chat_area (hero + chips),
    composer, settings caption — plus a sidebar with its own buttons.

    `with_chips` drops the empty-state chips (i.e. a started conversation).
    `extra_before` inserts filler elements ahead of the composer so the
    scopes are proven independent of sibling position.
    """
    chips = _block(
        _markdown('<span id="opm-chips-anchor"></span>')
        + '<div data-testid="stHorizontalBlock">'
        + _column(_button("Several users can't sign in"))
        + _column(_button("Shared printer shows jobs queued"))
        + "</div>"
    ) if with_chips else ""
    chat_area = _block(
        _markdown('<div class="opm-figure">hero</div>')
        + _markdown("<div>Try one of these</div>")
        + chips
        + '<div data-testid="stChatMessage">'
        + _markdown("<p>previous answer</p>")
        + _button("👍") + _button("👎")
        + "</div>"
    )
    composer = _block(
        _markdown('<span id="opm-composer-anchor"></span>')
        + _markdown('<div class="opm-composer-hint">Ask me anything</div>')
        + _element('<div data-testid="stTextInput">'
                   '<div data-baseweb="input">'
                   '<div data-baseweb="base-input"><input/></div></div></div>')
        + '<div data-testid="stHorizontalBlock">'
        + _column(_popover("📎 Attach", panel_button="Remove attachment"))
        + _column(_popover("🔍 Search"))
        + _column(_popover("✍ Writing Styles"))
        + _column(_popover("🎙 Voice"))
        + _column("")
        + _column(_button("⬆"))
        + "</div>"
    )
    sidebar = ('<section data-testid="stSidebar">'
               + _block(_button("＋ New chat") + _button("VPN keeps dropping"))
               + "</section>")
    filler = _markdown("<div>filler</div>") * extra_before
    html = (f"<body>{sidebar}"
            f'<section data-testid="stMain">'
            f'<div data-testid="stVerticalBlockBorderWrapper">'
            f'<div data-testid="stVerticalBlock">'
            f'{_markdown("<div class=\'opm-topbar\'>Overview / Ask-AI</div>")}'
            f"{filler}{chat_area}{composer}"
            f'{_element("<div>Style: Step-by-step</div>")}'
            f"</div></div></section></body>")
    return BeautifulSoup(html, "html.parser")


def _select(dom, selector: str) -> list:
    return sv.select(selector, dom)


def _skip(name: str) -> None:
    print(f"SKIP  {name}: bs4/soupsieve not installed")


# ------------------------------------------------------- scope resolution
def test_composer_scope_matches_exactly_one_block():
    if not _HAVE_SOUP:
        return _skip("test_composer_scope_matches_exactly_one_block")
    dom = build_dom()
    hits = _select(dom, theme._COMPOSER)
    assert len(hits) == 1, f"composer scope matched {len(hits)} blocks"
    # and it is the block that actually holds the anchor
    assert hits[0].select_one("#opm-composer-anchor") is not None


def test_chips_scope_matches_exactly_one_block():
    if not _HAVE_SOUP:
        return _skip("test_chips_scope_matches_exactly_one_block")
    dom = build_dom()
    hits = _select(dom, theme._CHIPS)
    assert len(hits) == 1, f"chips scope matched {len(hits)} blocks"
    assert hits[0].select_one("#opm-chips-anchor") is not None


def test_naive_selector_would_have_leaked():
    """Documents the original bug so nobody 'simplifies' it back."""
    if not _HAVE_SOUP:
        return _skip("test_naive_selector_would_have_leaked")
    dom = build_dom()
    naive = ('div[data-testid="stVerticalBlockBorderWrapper"]'
             ':has(#opm-composer-anchor)')
    leaked = _select(dom, naive)
    assert len(leaked) > 1, (
        "expected the naive selector to match ancestors as well; the DOM "
        "replica may no longer reflect Streamlit's nesting"
    )
    assert len(_select(dom, theme._COMPOSER)) == 1


# ------------------------------------------------------- send button rule
SEND_RULE = ('{scope} div[data-testid="stButton"] > '
             'button:not([data-testid="stPopover"] *)')


def test_send_rule_hits_only_the_send_button():
    if not _HAVE_SOUP:
        return _skip("test_send_rule_hits_only_the_send_button")
    dom = build_dom()
    hits = _select(dom, SEND_RULE.format(scope=theme._COMPOSER))
    labels = [b.get_text(strip=True) for b in hits]
    assert labels == ["⬆"], f"send rule matched {labels}"


def test_send_rule_spares_the_suggestion_chips():
    """The exact defect seen in the screenshot: chips as dark circles."""
    if not _HAVE_SOUP:
        return _skip("test_send_rule_spares_the_suggestion_chips")
    dom = build_dom()
    hits = set(_select(dom, SEND_RULE.format(scope=theme._COMPOSER)))
    chip_block = _select(dom, theme._CHIPS)[0]
    for chip in chip_block.select("button"):
        assert chip not in hits, f"chip restyled as send: {chip.get_text()}"


def test_send_rule_spares_popover_panel_buttons():
    if not _HAVE_SOUP:
        return _skip("test_send_rule_spares_popover_panel_buttons")
    dom = build_dom()
    hits = _select(dom, SEND_RULE.format(scope=theme._COMPOSER))
    assert all("Remove attachment" not in b.get_text() for b in hits)


def test_send_rule_spares_sidebar_and_feedback_buttons():
    if not _HAVE_SOUP:
        return _skip("test_send_rule_spares_sidebar_and_feedback_buttons")
    dom = build_dom()
    hits = {b.get_text(strip=True)
            for b in _select(dom, SEND_RULE.format(scope=theme._COMPOSER))}
    for label in ("＋ New chat", "VPN keeps dropping", "👍", "👎"):
        assert label not in hits, f"send rule leaked onto {label!r}"


# ------------------------------------------------------- other composer rules
def test_pill_rule_hits_every_popover_trigger():
    if not _HAVE_SOUP:
        return _skip("test_pill_rule_hits_every_popover_trigger")
    dom = build_dom()
    hits = _select(dom,
                   f'{theme._COMPOSER} [data-testid="stPopover"] button')
    labels = [b.get_text(strip=True) for b in hits]
    for expected in ("📎 Attach", "🔍 Search", "✍ Writing Styles", "🎙 Voice"):
        assert expected in labels, f"{expected} pill not styled"


def test_text_input_rule_reaches_the_baseweb_wrapper():
    """The grey box came from data-baseweb, not the stTextInput wrapper."""
    if not _HAVE_SOUP:
        return _skip("test_text_input_rule_reaches_the_baseweb_wrapper")
    dom = build_dom()
    assert _select(dom, f'{theme._COMPOSER} div[data-baseweb="base-input"]')
    assert _select(dom, f'{theme._COMPOSER} div[data-baseweb="input"]')
    assert _select(dom, f'{theme._COMPOSER} div[data-testid="stTextInput"] input')


def test_anchor_containers_are_hidden():
    if not _HAVE_SOUP:
        return _skip("test_anchor_containers_are_hidden")
    dom = build_dom()
    hidden = _select(
        dom,
        'div[data-testid="stElementContainer"]:has(#opm-chips-anchor),'
        'div[data-testid="stElementContainer"]:has(#opm-composer-anchor)',
    )
    assert len(hidden) == 2


def test_css_declares_every_scoped_rule_with_the_safe_scopes():
    """No rule may use the old unscoped/last-of-type forms."""
    css = theme.CSS
    assert ":last-of-type" not in css
    assert 'BorderWrapper"]:has(#opm-composer-anchor)' not in css
    assert 'BorderWrapper"]:has(#opm-chips-anchor)' not in css
    assert theme._COMPOSER in css and theme._CHIPS in css
    assert css.count("{") == css.count("}")


def test_scopes_are_independent_of_sibling_position():
    """The -n+2 form broke as soon as sibling order changed; this pins it."""
    if not _HAVE_SOUP:
        return _skip("test_scopes_are_independent_of_sibling_position")
    for extra in (0, 1, 2, 5):
        dom = build_dom(extra_before=extra)
        assert len(_select(dom, theme._COMPOSER)) == 1, f"extra={extra}"
        assert len(_select(dom, theme._CHIPS)) == 1, f"extra={extra}"
        sends = [b.get_text(strip=True)
                 for b in _select(dom, SEND_RULE.format(scope=theme._COMPOSER))]
        assert sends == ["⬆"], f"extra={extra} -> {sends}"


def test_started_conversation_has_no_chips_but_composer_still_scoped():
    if not _HAVE_SOUP:
        return _skip("test_started_conversation_has_no_chips_but_composer_ok")
    dom = build_dom(with_chips=False)
    assert _select(dom, theme._CHIPS) == []
    assert len(_select(dom, theme._COMPOSER)) == 1
    sends = [b.get_text(strip=True)
             for b in _select(dom, SEND_RULE.format(scope=theme._COMPOSER))]
    assert sends == ["⬆"]


def test_sidebar_label_rule_matches_the_paragraph_not_the_button():
    """The middle-of-title bug: rules must land on the inner <p>."""
    if not _HAVE_SOUP:
        return _skip("test_sidebar_label_rule_matches_the_paragraph_not_the_button")
    dom = build_dom()
    paras = _select(
        dom, 'section[data-testid="stSidebar"] .stButton>button p')
    texts = [p.get_text(strip=True) for p in paras]
    assert "VPN keeps dropping" in texts, texts
    # the composer's send rule must not reach into the sidebar
    sends = _select(dom, SEND_RULE.format(scope=theme._COMPOSER))
    for p in paras:
        assert p.find_parent("button") not in sends


def test_sidebar_rule_does_not_touch_composer_pills():
    if not _HAVE_SOUP:
        return _skip("test_sidebar_rule_does_not_touch_composer_pills")
    dom = build_dom()
    sidebar_hits = _select(dom, 'section[data-testid="stSidebar"] .stButton>button')
    labels = {b.get_text(strip=True) for b in sidebar_hits}
    for pill in ("📎 Attach", "🔍 Search", "✍ Writing Styles", "🎙 Voice", "⬆"):
        assert pill not in labels, f"sidebar rule leaked onto {pill}"


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
