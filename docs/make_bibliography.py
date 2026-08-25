"""Generate bibliography.tex (a self-contained thebibliography) from refs.bib.

Kept in the report folder so the build has no external dependency: the LaTeX service used for
remote compiles does not run BibTeX reliably, so the bibliography is written out explicitly.
"""
import re
from pathlib import Path

HERE = Path(__file__).parent


def _fields(body):
    """Split an entry body into name -> value, respecting nested braces."""
    out, i, n = {}, 0, len(body)
    while i < n:
        m = re.compile(r"(\w+)\s*=\s*").search(body, i)
        if not m:
            break
        j = m.end()
        if j < n and body[j] == "{":
            depth, k = 0, j
            while k < n:
                depth += (body[k] == "{") - (body[k] == "}")
                k += 1
                if depth == 0:
                    break
            val, i = body[j + 1:k - 1], k
        else:                                   # bare or quoted value
            k = body.find(",", j)
            k = n if k < 0 else k
            val, i = body[j:k].strip().strip('"'), k
        out[m.group(1).lower()] = " ".join(val.split())
    return out


def parse_bib(text):
    text = re.sub(r"^\s*%.*$", "", text, flags=re.M)
    entries, i = [], 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,]+),", re.S).search(text, i)
        if not m:
            return entries
        depth, k = 0, m.end() - len(m.group(0)) + m.group(0).index("{")
        while k < len(text):
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
            if depth == 0:
                break
        entries.append((m.group(2).strip(), _fields(text[m.end():k - 1])))
        i = k


def unwrap(t):
    """Strip one redundant outer brace pair only.

    Braces are left alone otherwise: they are harmless to LaTeX ({QuTiP} renders as QuTiP) and
    removing them corrupts accent constructs such as {\\c c} and {\\'e}.
    """
    t = t.strip()
    if t.startswith("{") and t.endswith("}"):
        depth = 0
        for i, ch in enumerate(t):
            depth += (ch == "{") - (ch == "}")
            if depth == 0 and i < len(t) - 1:
                return t                      # the outer braces are not a single group
        return t[1:-1].strip()
    return t


def fmt(key, f):
    parts = []
    if "author" in f:
        parts.append(unwrap(f["author"]))
    if "title" in f:
        parts.append(f"\\emph{{{unwrap(f['title'])}}}")
    for k in ("journal", "booktitle", "howpublished", "school", "publisher"):
        if k in f:
            parts.append(f[k] if k == "howpublished" else unwrap(f[k]))
            break
    if "volume" in f:
        parts.append(f"\\textbf{{{unwrap(f['volume'])}}}")
    if "pages" in f:
        parts.append(unwrap(f["pages"]))
    if "year" in f:
        parts.append(f"({unwrap(f['year'])})")
    if "note" in f:
        parts.append(unwrap(f["note"]))
    return f"\\bibitem{{{key}}} " + ", ".join(parts) + "."


def main():
    entries = parse_bib((HERE / "refs.bib").read_text())
    lines = ["% generated from refs.bib by make_bibliography.py -- do not edit by hand",
             "\\begin{thebibliography}{99}"]
    lines += [fmt(k, f) for k, f in entries]
    lines.append("\\end{thebibliography}")
    (HERE / "bibliography.tex").write_text("\n".join(lines) + "\n")
    print(f"bibliography: {len(entries)} entries")


if __name__ == "__main__":
    main()
