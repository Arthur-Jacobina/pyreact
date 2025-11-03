from __future__ import annotations

import html
import re
from typing import Dict


_SGR_RE = re.compile(r"\x1b\[(?P<codes>[0-9;]*)m")


def _style_from_codes(codes: str, state: Dict[str, object]) -> Dict[str, object]:
    # Mutates and returns state
    if not codes:
        codes_list = [0]
    else:
        codes_list = [int(c or 0) for c in codes.split(";")]

    for code in codes_list:
        if code == 0:  # reset
            state.clear()
            continue

        # intensity / decorations
        if code == 1:
            state["bold"] = True
        elif code == 2:
            state["dim"] = True
        elif code == 22:
            state.pop("bold", None)
            state.pop("dim", None)
        elif code == 3:
            state["italic"] = True
        elif code == 23:
            state.pop("italic", None)
        elif code == 4:
            state["underline"] = True
        elif code == 24:
            state.pop("underline", None)

        # foreground
        elif code == 39:
            state.pop("fg", None)
        elif 30 <= code <= 37:
            state["fg"] = code
        elif 90 <= code <= 97:
            state["fg"] = code

        # background
        elif code == 49:
            state.pop("bg", None)
        elif 40 <= code <= 47:
            state["bg"] = code
        elif 100 <= code <= 107:
            state["bg"] = code

    return state


_COLOR_MAP = {
    30: "#000000",
    31: "#dc2626",  # red-600
    32: "#16a34a",  # green-600
    33: "#ca8a04",  # yellow-600
    34: "#2563eb",  # blue-600
    35: "#7c3aed",  # violet-600
    36: "#0891b2",  # cyan-600
    37: "#e5e7eb",  # gray-200
    90: "#6b7280",  # gray-500
    91: "#ef4444",  # red-500
    92: "#22c55e",  # green-500
    93: "#eab308",  # yellow-500
    94: "#3b82f6",  # blue-500
    95: "#a78bfa",  # violet-400
    96: "#06b6d4",  # cyan-500
    97: "#f3f4f6",  # gray-100
}

_BGCOLOR_FROM_FG = {
    40: 30,
    41: 31,
    42: 32,
    43: 33,
    44: 34,
    45: 35,
    46: 36,
    47: 37,
    100: 90,
    101: 91,
    102: 92,
    103: 93,
    104: 94,
    105: 95,
    106: 96,
    107: 97,
}


def _css_from_state(state: Dict[str, object]) -> str:
    css: list[str] = []
    fg = state.get("fg")
    bg = state.get("bg")
    if isinstance(fg, int):
        col = _COLOR_MAP.get(fg)
        if col:
            css.append(f"color:{col}")
    if isinstance(bg, int):
        fg_equiv = _BGCOLOR_FROM_FG.get(bg)
        col = _COLOR_MAP.get(fg_equiv) if fg_equiv is not None else None
        if col:
            css.append(f"background-color:{col}")
    if state.get("bold"):
        css.append("font-weight:600")
    if state.get("dim"):
        css.append("opacity:0.8")
    if state.get("italic"):
        css.append("font-style:italic")
    if state.get("underline"):
        css.append("text-decoration:underline")
    return ";".join(css)


def ansi_to_html(s: str) -> str:
    """Convert ANSI-colored text to escaped HTML with <span> styles.
    Keeps newlines; safe to inject as innerHTML.
    """
    out: list[str] = []
    pos = 0
    state: Dict[str, object] = {}
    open_style = ""

    def open_span(new_style: str):
        nonlocal open_style
        if open_style:
            out.append("</span>")
            open_style = ""
        if new_style:
            out.append(f'<span style="{new_style}">')
            open_style = new_style

    for m in _SGR_RE.finditer(s):
        if m.start() > pos:
            out.append(html.escape(s[pos : m.start()]))

        _style_from_codes(m.group("codes"), state)
        open_span(_css_from_state(state))
        pos = m.end()

    # tail
    if pos < len(s):
        out.append(html.escape(s[pos:]))
    if open_style:
        out.append("</span>")
    return "".join(out)
