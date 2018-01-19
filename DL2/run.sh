#!/bin/bash

for i in "$@"; do
   echo Ejecutando... w4code$i.py
   python w4code$i.py > log$i.out 2>log$i.err
done
