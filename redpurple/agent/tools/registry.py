from collections.abc import Callable
from pathlib import Path
from typing import Any

_tools: dict[str, Callable[..., Any]] = {}
_schemas: dict[str, str] = {}


def register_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    name = func.__name__
    _tools[name] = func

    # Load schema from adjacent XML file: tools/<folder>/<file>_schema.xml
    parts = func.__module__.split(".")
    if len(parts) >= 2:
        folder = parts[-2]
        file_stem = parts[-1]
        schema_path = Path(__file__).parent / folder / f"{file_stem}_schema.xml"
        if schema_path.exists():
            _schemas[name] = schema_path.read_text(encoding="utf-8")

    return func


def get_tool(name: str) -> Callable[..., Any] | None:
    return _tools.get(name)


def get_tools_prompt() -> str:
    return "\n\n".join(_schemas.values())
