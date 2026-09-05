#!/bin/zsh
# Build the report.  Run from report_build/:  zsh build.sh
#
# Two documents come out of this:
#   report_main.pdf              working copy - proposed deletions still typeset, marked blue
#   MLBD_2025_10_01951547.pdf    THE SUBMISSION - deletions actually applied (built from report_final.tex)
#
# Reviewing report_main.pdf is misleading: text and whole figures marked for deletion still appear
# there, so it shows more figures and thousands more words than will be handed in.  Review the
# submission file.
cd "$(dirname "$0")"
export TEXMFVAR="${TEXMFVAR:-${TMPDIR:-/tmp}/texmf-var}"
PDFLATEX=/Library/TeX/texbin/pdflatex
PY="${PY:-python3}"

"$PY" count_words.py --write | tail -2
if [ refs.bib -nt bibliography.tex ]; then
  echo "refs.bib changed: regenerating bibliography.tex"
  "$PY" make_bibliography.py || exit 1
fi

build () {  # $1 = jobname without .tex
  for i in 1 2; do
    "$PDFLATEX" -interaction=nonstopmode -halt-on-error "$1.tex" > "build_$1_$i.log" 2>&1 || {
      echo "BUILD FAILED ($1, pass $i):"; grep -m5 -A3 "^!" "build_$1_$i.log"; exit 1; }
  done
  rm -f "build_$1_1.log" "build_$1_2.log"
}

build report_main
"$PY" make_final.py
build report_final
cp report_final.pdf MLBD_2025_10_01951547.pdf

pages () { grep -o "Output written on $1.pdf ([0-9]* pages" "$1.log" | grep -o "[0-9]*"; }
figs ()  { grep -c "^\\\\includegraphics\|includegraphics" "$1.tex"; }
echo "working copy : report_main.pdf   $(pages report_main) pages"
echo "SUBMISSION   : MLBD_2025_10_01951547.pdf   $(pages report_final) pages, $(grep -c 'includegraphics' report_final.tex) figures"
u=$(grep -c "Reference.*undefined" report_final.log 2>/dev/null); u=${u:-0}
[ "$u" -gt 0 ] && echo "WARNING: $u undefined reference(s) in the submission" || echo "references: all resolved"
