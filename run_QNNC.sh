#!/bin/bash
for e in linear full circular; do
  for f in 1 2 3 4 5 6; do
    for a in 1 2 3 4 5 6; do
      nohup python QNNC_hybrid.py -e $e -f $f -a $a > QNNC/logs/QNNC_e_${e}_f_${f}_a_${a}.log &
    done
  done
done
