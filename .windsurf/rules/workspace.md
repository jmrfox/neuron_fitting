---
trigger: always_on
---

uv:
I use the uv package manager. When running Python in the terminal, do "uv run script.py" to run a script with the venv.
Tests should be run using "uv run pytest".

package reference documents:
I have a script at scripts/generate_package_reference.py that can be used to generate a reference document with the output of Python's help() for any package and its submodules, plus classes and functions, recursively. Use this to generate reference documents for packages that are not well documented.

