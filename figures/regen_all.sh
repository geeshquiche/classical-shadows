#!/bin/zsh
PY=/Users/vzs/miniconda3/envs/MLBDproj/bin/python
cd "$(dirname "$0")"
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mplcfg
for f in element_grid_final.py csv_figs_vector.py fig_extra.py ladder_fig.py sweep_figs.py partial_fig.py; do
  echo "--- $f"; "$PY" "$f" 2>&1 | grep -v Warning | tail -1
done
cp out/*.pdf out/*.png /Users/vzs/MLBD/ProjQDofCS/report_build/
echo "FIGURES REGENERATED $(date)"
