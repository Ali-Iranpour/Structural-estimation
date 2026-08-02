#!/usr/bin/env python3
"""
Strip outputs and execution counts from a Jupyter notebook.

Used as a git *clean* filter: your working copy keeps its outputs, but what
gets committed is the stripped version. This keeps the repository small --
the full transfer_CRRA_wage.ipynb is ~58 MB with outputs and well under 1 MB
without them, and every save would otherwise add another copy to history.

Reads a notebook on stdin, writes the stripped notebook to stdout.
Anything that is not valid JSON is passed through unchanged, so the filter
can never corrupt a file it does not understand (e.g. an LFS pointer).

Install with tools/setup-git-filters.sh -- git filter config lives in
.git/config, which is not committed, so every clone must run it once.
"""
import json
import sys


def strip(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            # widget/scroll state that changes on every run
            meta = cell.get("metadata", {})
            for key in ("collapsed", "scrolled", "execution"):
                meta.pop(key, None)
    meta = nb.get("metadata", {})
    # kernel session id changes every run and creates spurious diffs
    meta.pop("widgets", None)
    if "language_info" in meta:
        meta["language_info"].pop("version", None)
    return nb


def main():
    raw = sys.stdin.read()
    try:
        nb = json.loads(raw)
    except ValueError:
        # not JSON (LFS pointer, partial file, ...) -- pass through untouched
        sys.stdout.write(raw)
        return
    json.dump(strip(nb), sys.stdout, indent=1, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
