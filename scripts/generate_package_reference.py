#!/usr/bin/env python3
"""
Generate a reference document with the output of Python's help() for any package and
its submodules, plus classes and functions, recursively.

Usage (from repo root):
    uv run python scripts/generate_package_reference.py \
        --module <package_name> \
        --output <OUTPUT_FILE>.md

Examples:
    uv run python scripts/generate_package_reference.py --module arbor --output ARBOR_REFERENCE.md
    uv run python scripts/generate_package_reference.py --module numpy --output NUMPY_REFERENCE.md
    uv run python scripts/generate_package_reference.py --module pandas --output PANDAS_REFERENCE.md

Notes:
- This can produce a VERY large file. Expect minutes and many MBs for large packages.
- We use pydoc.render_doc() to capture the same text help() would print.
- We attempt to import package submodules via pkgutil.walk_packages; failures are skipped.
- We also render help() for classes and functions found in each module.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import io
import pkgutil
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

import pydoc


def iter_submodules(pkg_name: str) -> Iterable[str]:
    """Yield fully-qualified submodule names for a package.

    If the package has no __path__ (i.e., is a single module), yields nothing.
    """
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:
        return
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return
    for modinfo in pkgutil.walk_packages(pkg_path, prefix=pkg_name + "."):
        yield modinfo.name


def safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _remove_overstrikes(s: str) -> str:
    """Remove backspace overstrike formatting (e.g., 'N\bN' patterns)."""
    out: list[str] = []
    for ch in s:
        if ch == "\b":
            if out:
                out.pop()
        else:
            out.append(ch)
    return "".join(out)


def render_help(obj) -> str:
    """Render help() output for obj as a plain, readable string."""
    try:
        raw = pydoc.render_doc(obj)
        try:
            # pydoc.plain strips overstrikes and control sequences
            text = pydoc.plain(raw)
        except Exception:
            text = _remove_overstrikes(raw)
    except Exception as e:
        text = f"<help() failed: {e}>\n"
    return text


def collect_module_members(mod) -> Tuple[list, list]:
    """Return (classes, functions) from a module object."""
    classes, functions = [], []
    try:
        for name, val in inspect.getmembers(mod):
            if inspect.isclass(val):
                classes.append(val)
            elif (
                inspect.isfunction(val)
                or inspect.isbuiltin(val)
                or inspect.ismethoddescriptor(val)
            ):
                functions.append(val)
    except Exception:
        pass
    return classes, functions


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--module", required=True, help="Top-level module/package to document"
    )
    ap.add_argument("--output", required=True, help="Output Markdown file path")
    ap.add_argument(
        "--include-privates",
        action="store_true",
        help="Attempt to import private submodules as well",
    )
    ap.add_argument(
        "--extra-submodules",
        nargs="*",
        default=[],
        help="Additional submodules to try importing explicitly",
    )
    args = ap.parse_args()

    top_name = args.module
    out_path = Path(args.output)

    visited_modules: Set[str] = set()
    visited_objects: Set[int] = set()

    # Gather module list: top module + discovered submodules.
    modules: list[str] = []

    # Always include top module.
    modules.append(top_name)

    # Discover submodules.
    for mname in iter_submodules(top_name):
        # Optionally skip private names (segments with ._)
        if not args.include_privates and any(
            part.startswith("_") for part in mname.split(".")
        ):
            continue
        modules.append(mname)

    # Try additional submodules explicitly if provided
    for sub in args.extra_submodules:
        mname = f"{top_name}.{sub}"
        if safe_import(mname) is not None:
            modules.append(mname)

    # De-duplicate while preserving order.
    seen = set()
    modules = [m for m in modules if not (m in seen or seen.add(m))]

    lines: list[str] = []
    lines.append(f"{top_name} Reference (help() output)\n")
    lines.append(
        "Generated via pydoc.render_doc for modules, classes, and functions.\n"
    )

    for mname in modules:
        if mname in visited_modules:
            continue
        mod = safe_import(mname)
        if mod is None:
            continue
        visited_modules.add(mname)

        # Module header
        lines.append("\n" + "=" * 80 + "\n")
        lines.append(f"Module: {mname}\n")
        lines.append("-" * 80 + "\n")
        lines.append(render_help(mod))

        # Classes and functions in this module
        classes, functions = collect_module_members(mod)

        # Classes
        for cls in classes:
            if id(cls) in visited_objects:
                continue
            visited_objects.add(id(cls))
            qn = getattr(cls, "__qualname__", getattr(cls, "__name__", str(cls)))
            lines.append("\n" + "~" * 80 + "\n")
            lines.append(f"Class: {mname}.{qn}\n")
            lines.append("~" * 80 + "\n")
            lines.append(render_help(cls))

        # Functions
        for fn in functions:
            if id(fn) in visited_objects:
                continue
            visited_objects.add(id(fn))
            qn = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
            lines.append("\n" + "~" * 80 + "\n")
            lines.append(f"Function: {mname}.{qn}\n")
            lines.append("~" * 80 + "\n")
            lines.append(render_help(fn))

    # Write output
    out_path.write_text("".join(lines))
    print(f"Wrote {out_path} with {len(lines)} lines across {len(modules)} modules.")


if __name__ == "__main__":
    sys.exit(main())
