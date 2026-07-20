"""Assembles the HTML viewer page from the template, stylesheet, and JS modules."""
import json
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Load order only matters for main.js, which must run last (it invokes the
# render functions defined by the other modules and expects them to exist).
_JS_FILES = [
    "utils.js",
    "diff.js",
    "evolution.js",
    "validation_chart.js",
    "similarity_heatmap.js",
    "scores.js",
    "pool.js",
    "reflector.js",
    "reasoning.js",
    "iteration.js",
    "info_sidebar.js",
    "sidebar.js",
    "resizer.js",
    "main.js",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_html(data: dict) -> str:
    template = _read(_ASSETS_DIR / "template.html")
    styles = _read(_ASSETS_DIR / "styles.css")
    scripts = "\n\n".join(_read(_ASSETS_DIR / "js" / f) for f in _JS_FILES)

    safe_json = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )

    html = (
        template
        .replace("__NAME__", data["name"])
        .replace("__STYLES__", styles)
        .replace("__SCRIPTS__", scripts)
    )
    return html.replace("__DATA__", safe_json)
