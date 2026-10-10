#!/bin/bash
# Aritmetik: aynı block beyni, normal vs ters iz. Müfredat: 1-basamak 350 iter -> mezuniyet think_math 200 iter.
export TMP="D:\newtemps" TEMP="D:\newtemps" PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /d/NoRa_AGI
PY=.venv/Scripts/python.exe; T=data/text
for arm in norm rev; do
  if [ $arm = rev ]; then S=_rev; F=--rev-trace; else S=; F=; fi
  $PY grow_gpt.py --stage refine --pool codemath --batch 16 --seg M --brain models/ab_block16.pt --data $T/think_1digit$S.txt --iters 350 --out models/ar_${arm}1.pt > runs/ar_${arm}1.log 2>&1
  $PY grow_gpt.py --stage refine --pool codemath --batch 16 --seg M --brain models/ar_${arm}1.pt --data $T/think_math$S.txt --iters 200 --out models/ar_${arm}2.pt > runs/ar_${arm}2.log 2>&1
  $PY grow_gpt.py --stage evalarith --seg M --brain models/ar_${arm}2.pt $F > runs/ar_${arm}_eval.log 2>&1
done
# A/B gürültü kontrolü: seed 1
$PY grow_gpt.py --stage auto --pool codemath --batch 16 --adapt-pos head  --adapt-bn 64 --seed 1 --out models/ab_head64_s1.pt  > runs/ab_head64_s1.log 2>&1
$PY grow_gpt.py --stage auto --pool codemath --batch 16 --adapt-pos block --adapt-bn 16 --seed 1 --out models/ab_block16_s1.pt > runs/ab_block16_s1.log 2>&1
echo BITTI
