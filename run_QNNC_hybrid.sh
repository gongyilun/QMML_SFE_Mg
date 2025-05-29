#!/bin/bash
filename=QNNC_hybrid.py
date=`cat ${filename}|grep 'date ='|awk '{print $3}'| cut -d "'" -f 2`
for e in linear full circular; do
  for f in 1 2 3 4 5 6; do
    for a in 1 2 3 4 5 6; do
      nohup python ${filename} -e $e -f $f -a $a > QNNC_hybrid/logs/QNNC_hybrid_e_${e}_f_${f}_a_${a}_${date}.log &
    done
  done
done
