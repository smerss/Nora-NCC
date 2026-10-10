#!/bin/bash
export TMP="D:\newtemps" TEMP="D:\newtemps" PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /d/NoRa_AGI
.venv/Scripts/python.exe nora_think.py --days 5 --boot-iters 600 --wake 40 --sleep-iters 150 --exam-per-level 10 > runs/think_run.log 2>&1
echo BITTI
