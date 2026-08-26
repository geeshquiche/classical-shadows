#!/usr/bin/env python3
"""Every report figure must be drawn at or under the text-column width.

A figure wider than the column is scaled down by \\includegraphics, and every label in it shrinks by the
same factor -- the cause of the unreadable-axis problem.  A wide legend or a stray annotation silently
expands the canvas under savefig(bbox="tight"), so this is checked after every regeneration.
"""
import re, sys, pathlib
TEXTWIDTH = 6.7
bad = []
for f in sorted(pathlib.Path("out").glob("*.pdf")):
    d = f.read_bytes()[:4000]
    m = re.search(rb'MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', d)
    if not m:
        continue
    w = (float(m.group(3)) - float(m.group(1))) / 72.0
    h = (float(m.group(4)) - float(m.group(2))) / 72.0
    flag = ""
    if w > TEXTWIDTH + 0.02:
        flag = f"  <-- TOO WIDE, would shrink labels to {TEXTWIDTH/w:.0%}"
        bad.append(f.name)
    print(f"  {f.name:36s} {w:5.2f} x {h:4.2f} in{flag}")
print(f"\n{'FAIL: ' + ', '.join(bad) if bad else 'OK: every figure fits the text column'}")
sys.exit(1 if bad else 0)
