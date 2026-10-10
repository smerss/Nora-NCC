#!/bin/bash
export TMP="D:\newtemps" TEMP="D:\newtemps" PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /d/NoRa_AGI
.venv/Scripts/python.exe nora_think.py --cortex column --no-lateral --days 6 --boot-iters 2500 --wake 60 --sleep-iters 300 --exam-per-level 10 --out models/think_col.pt > runs/think_col.log 2>&1
echo BITTI
