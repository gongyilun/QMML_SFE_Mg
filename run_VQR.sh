#!/bin/bash
filename=VQR.py
date=`cat ${filename}|grep 'date ='|awk '{print $3}'| cut -d "'" -f 2`
for e in linear full circular; do
  for f in 1 2 3 4 5; do
    for a in 1 2 3 4 5; do
      for p in True False; do
        nohup python -u ${filename} -e $e -f $f -a $a -p $p > VQR/logs/VQR_e_${e}_f_${f}_a_${a}_p_${p}_${date}.log &
      done
    done
  done
done