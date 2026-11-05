#!/bin/zsh
# GFN2 data for the 7-rotaxane test: MD on each complex, frames cut into complex/rod/wheel,
# gas labels, then GFN2/ALPB relabels in 6 solvents. dethread4 is held out as validation.
# Every stage skips finished outputs, so rerunning resumes. Launch detached (see DEV.md):
#   ~/miniforge3/bin/python -c "import subprocess; subprocess.Popen(['zsh','scripts/run_rotx7.sh'], stdout=open('data/rotx7.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)"
set -e
cd "${0:A:h}/.."
caffeinate -ims -w $$ &
PY=~/python_mac/mace/.venv/bin/python   # tblite lives in the mace env (DEV.md)
OUT=data/rotx7
mkdir -p $OUT
echo "== start $(date)"
$PY scripts/make_gfn2_data.py --fragments --out $OUT --md-steps 1200 --sample-every 30 \
    --valid-systems dethread4 inputs/current_structures/*.xyz
for s in hexane chloroform acetone dmf water dmso; do
    echo "== $s $(date)"
    $PY scripts/make_gfn2_data.py --relabel --solvent $s --out $OUT $OUT/gfn2_gas_train.xyz $OUT/gfn2_gas_valid.xyz
done
echo "== done $(date)"
