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


def citation_order():
    """Keys in order of first citation, submission first then anything only the working copy cites.

    A numbered reference list should count upward as the reader meets the references; ordering by
    refs.bib file order instead makes the introduction open on [28, 29, 30].
    """
    import re, sys
    sys.path.insert(0, str(HERE))
    from count_words import strip_cut
    seen = []
    for src, transform in (("report_main.tex", strip_cut), ("report_main.tex", lambda x: x)):
        text = transform((HERE / src).read_text())
        for m in re.finditer(r"\\cite\{([^}]+)\}", text):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k and k not in seen:
                    seen.append(k)
    return seen


def main():
    entries = parse_bib((HERE / "refs.bib").read_text())
    by_key = dict(entries)
    order = citation_order()
    ordered = [(k, by_key[k]) for k in order if k in by_key]
    missing = [(k, f) for k, f in entries if k not in order]      # in refs.bib but never cited
    lines = ["% generated from refs.bib by make_bibliography.py -- do not edit by hand",
             "% entries are numbered in order of first citation",
             "\\begin{thebibliography}{99}"]
    lines += [fmt(k, f) for k, f in ordered + missing]
    lines.append("\\end{thebibliography}")
    (HERE / "bibliography.tex").write_text("\n".join(lines) + "\n")
    print(f"bibliography: {len(ordered)} cited entries in citation order"
          + (f", {len(missing)} uncited appended" if missing else ""))


if __name__ == "__main__":
    main()
