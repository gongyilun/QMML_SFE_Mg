#!/bin/bash
filename=QSVR.py
date=`cat ${filename}|grep 'date ='|awk '{print $3}'| cut -d "'" -f 2`
for e in linear full circular; do
  for f in 1 2 3 4 5 6; do
    for r in 0.1 1 10 100; do
      for p in 0.1 0.01 0.001; do
        nohup python -u ${filename} -e $e -f $f -r $r -p $p > QSVR/logs/QSVR_e_${e}_f_${f}_r_${r}_p_${p}_${date}.log &
      done
    done
  done
done
