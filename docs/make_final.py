#!/usr/bin/env python3
"""Produce report_final.tex: the document as it will actually be submitted.

report_main.tex carries proposed deletions marked in blue (\cutw{...} and cutblock).  Those still
typeset, so the working PDF shows content -- including whole figures -- that will not be in the
submission.  This applies the deletions for real, so what is reviewed is what is handed in.
"""
import pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from count_words import strip_cut

src = pathlib.Path(__file__).with_name("report_main.tex").read_text()
out = strip_cut(src)
# with nothing left to mark, render the cut macros inert so no blue survives
# nothing is marked any more, so make the cut macros inert (no stray blue can survive)
out = out.replace(r"\newcommand{\cutw}[1]{{\color{blue}#1}}", r"\newcommand{\cutw}[1]{#1}")
out = out.replace(r"\newenvironment{cutblock}{\par\begingroup\color{blue}}{\endgroup\par}",
                  r"\newenvironment{cutblock}{\par}{\par}")
# collapse blank-line runs left by removed blocks (they become spurious \par breaks)
out = re.sub(r"\n{3,}", "\n\n", out)
pathlib.Path(__file__).with_name("report_final.tex").write_text(out)
print(f"  report_final.tex written ({len(src)} -> {len(out)} chars)")
