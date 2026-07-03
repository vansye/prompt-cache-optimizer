"""Configuration file loader for popt (.poptimerc).

Search order (last found wins for merge):
  1. Environment variable ``POPT_CONFIG`` (explicit path)
  2. ``./.poptimerc`` (project-local)
  3. Walk up directories from cwd toward filesystem root
  4. ``~/.poptimerc`` (user home)

Format: TOML (preferred) or JSON (fallback).  Supports Python 3.10+
via ``tomllib`` (stdlib 3.11+) or ``tomli`` (backport, optional).
"""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Config Model ────────────────────────────────────────────────────────


@dataclass
class PopConfig:
    """Merged configuration from all discovered .poptimerc files.

    Attributes:
        model: Default model name (e.g. "deepseek-v4-flash").
        provider: Default provider name.
        upstream: Default upstream URL.
        host: Proxy bind host.
        port: Proxy bind port.
        extra: Any unrecognised keys from the config file
               (for forward-compatibility).
    """
    model: str = ""
    provider: str = ""
    upstream: str = ""
    host: str = "127.0.0.1"
    port: int = 9999
    extra: dict = field(default_factory=dict)


# ── File Discovery ─────────────────────────────────────────────────────


def _search_parents(start: Path) -> Iterator[Path]:
    """Yield ancestor directories from ``start`` up to the filesystem root."""
    current = start.resolve()
    while True:
        yield current
        parent = current.parent
        if parent == current:
            break
        current = parent


def find_config_files(cwd: str | None = None) -> list[str]:
    """Find all .poptimerc files in the search path.

    Returns paths in order of increasing priority (so later entries
    override earlier ones when merged).
    """
    seen: list[str] = []

    # 1. Explicit env var
    env_path = os.environ.get("POPT_CONFIG", "")
    if env_path:
        p = Path(env_path)
        if p.exists():
            seen.append(str(p.resolve()))

    # 2. Walk up from cwd
    start = Path(cwd or os.getcwd())
    for d in _search_parents(start):
        candidate = d / ".poptimerc"
        if candidate.exists() and str(candidate.resolve()) not in seen:
            seen.append(str(candidate.resolve()))

    # 3. User home
    home = Path.home() / ".poptimerc"
    if home.exists() and str(home.resolve()) not in seen:
        seen.append(str(home.resolve()))

    return seen


# ── Parsing ─────────────────────────────────────────────────────────────


def _try_load_toml(path: str) -> dict | None:
    """Try parsing a file as TOML.

    Uses ``tomllib`` (Python 3.11+) or ``tomli`` (backport).
    Returns None if neither is available or parsing fails.
    """
    for mod_name in ("tomllib", "tomli"):
        try:
            mod = __import__(mod_name)
            with open(path, "rb") as f:
                return mod.load(f)
        except (ImportError, ModuleNotFoundError):
            continue
        except Exception:
            return None
    return None


def _try_load_json(path: str) -> dict | None:
    """Try parsing a file as JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_file(path: str) -> dict | None:
    """Load a single config file, guessing format from extension.

    ``.toml`` → TOML,  ``.json`` → JSON,  no extension → try TOML then JSON.
    """
    ext = Path(path).suffix.lower()
    if ext == ".toml":
        return _try_load_toml(path)
    elif ext == ".json":
        return _try_load_json(path)
    else:
        # No extension: try TOML first (preferred), then JSON
        return _try_load_toml(path) or _try_load_json(path)


# ── Merge ───────────────────────────────────────────────────────────────


def _merge_into(base: dict, overlay: dict) -> dict:
    """Merge ``overlay`` into ``base`` (shallow merge, overlay wins)."""
    result = dict(base)
    for k, v in overlay.items():
        result[k] = v
    return result


def load_config(cwd: str | None = None) -> PopConfig:
    """Discover and merge all .poptimerc files into a single PopConfig.

    Args:
        cwd: Working directory to start searching from.
             Defaults to ``os.getcwd()``.

    Returns:
        A ``PopConfig`` with all values merged.  Missing values
        remain at their defaults (empty strings / sensible fallbacks).
    """
    files = find_config_files(cwd)
    merged: dict = {}

    for path in files:
        data = load_file(path)
        if data:
            merged = _merge_into(merged, data)

    if not merged:
        return PopConfig()

    # Extract known keys from the [project] section or top-level
    project = merged.get("project", merged)
    proxy_sec = merged.get("proxy", {})

    return PopConfig(
        model=str(project.get("model", "")),
        provider=str(project.get("provider", "")),
        upstream=str(project.get("upstream", "")),
        host=str(proxy_sec.get("host", "127.0.0.1")),
        port=int(proxy_sec.get("port", 9999)),
        extra={k: v for k, v in merged.items()
               if k not in ("project", "proxy")},
    )


# ── Convenience ─────────────────────────────────────────────────────────


def get_config_value(key: str, default: str = "") -> str:
    """Quick helper: load config and read one string value.

    Useful for simple lookups without creating a full PopConfig.
    """
    cfg = load_config()
    return str(getattr(cfg, key, default) or default)
