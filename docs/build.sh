#!/bin/zsh
# Build the report. Run from report_build/:  zsh build.sh
# bibliography.tex is generated from refs.bib by make_bibliography.py when that file is newer.
cd "$(dirname "$0")"
export TEXMFVAR=/Users/vzs/MLBD/tools/texmf-var
PDFLATEX=/Library/TeX/texbin/pdflatex
PY=/Users/vzs/miniconda3/envs/MLBDproj/bin/python
"$PY" count_words.py --write | tail -2
if [ refs.bib -nt bibliography.tex ]; then
  echo "refs.bib changed: regenerating bibliography.tex"
  /Users/vzs/miniconda3/envs/MLBDproj/bin/python make_bibliography.py || exit 1
fi
for i in 1 2; do
  "$PDFLATEX" -interaction=nonstopmode -halt-on-error report_main.tex > build_pass$i.log 2>&1 || {
    echo "BUILD FAILED (pass $i):"; grep -m5 -A3 "^!" build_pass$i.log; exit 1; }
done
rm -f build_pass1.log build_pass2.log
cp report_main.pdf MLBD_2025_10_01951547.pdf
echo "BUILD OK: $(stat -f%z report_main.pdf) bytes  $(date '+%b %d %H:%M')"
echo "submission copy: MLBD_2025_10_01951547.pdf"
grep -c "Reference.*undefined" report_main.log > /dev/null && u=$(grep -c "Reference.*undefined" report_main.log) || u=0
[ "$u" -gt 0 ] && echo "WARNING: $u undefined reference(s)" || echo "references: all resolved"
