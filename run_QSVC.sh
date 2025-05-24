#!/bin/bash
for e in linear full circular; do
  for f in 1 2 3 4 5 6; do
    for r in 0.1 1 10 100; do
      nohup python QSVC.py -e $e -f $f -r $r > QSVC/logs/QSVC_e_${e}_f_${f}_r_${r}.log &
    done
  done
done
