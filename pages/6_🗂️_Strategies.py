"""Browse the strategy gallery — name, docstring, and source code per strategy."""
from __future__ import annotations

import ast
from pathlib import Path

import streamlit as st

from app_shared import setup_page
from trading_agent.config import PROJECT_ROOT


setup_page("Strategies", icon="🗂️")

st.title("🗂️ Strategies")
st.caption(
    "Every Python file under `strategies/` that subclasses the `Strategy` base "
    "class is automatically discovered and usable from the Backtest, Run Agent, "
    "and Paper Trade pages."
)


def _strategy_files() -> list[Path]:
    sd = PROJECT_ROOT / "strategies"
    if not sd.exists():
        return []
    return sorted(
        p for p in sd.glob("*.py")
        if not p.name.startswith("_") and p.stem != "__init__"
    )


def _extract_docstring(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                return doc
    return None


files = _strategy_files()
if not files:
    st.warning("No strategies found under `strategies/`.")
    st.stop()

st.markdown(f"### {len(files)} strategy module(s)")

for f in files:
    source = f.read_text(encoding="utf-8")
    docstring = _extract_docstring(source) or "(no docstring)"

    with st.expander(f"**{f.stem}** — {docstring.splitlines()[0]}", expanded=False):
        st.markdown(f"**Docstring**")
        st.write(docstring)

        st.markdown(f"**Source** (`strategies/{f.name}`)")
        st.code(source, language="python")

st.divider()
st.markdown("### Adding your own")
st.write(
    "Drop a Python file into `strategies/` with a class that subclasses "
    "`trading_agent.core.strategy.Strategy` and implements `on_bar(bar, portfolio) -> list[Order]`. "
    "Restart the app and it'll appear in the dropdowns."
)
