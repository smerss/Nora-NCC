#!/bin/bash
export TMP="D:\newtemps" TEMP="D:\newtemps" PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /d/NoRa_AGI
PY=.venv/Scripts/python.exe
$PY grow_gpt.py --stage auto --pool codemath --batch 16 --adapt-pos head  --adapt-bn 64 --out models/ab_head64.pt  > runs/ab_head64.log 2>&1
$PY grow_gpt.py --stage auto --pool codemath --batch 16 --adapt-pos block --adapt-bn 16 --out models/ab_block16.pt > runs/ab_block16.log 2>&1
echo BITTI
