#!/bin/bash

for f in *.log
do
  awk -F" " '$1=="500000"{printf "%s %s ",$(NF-1),$NF >> "energy.txt";}' $f 	
  INPUT="$(basename $f .log)"
  echo $INPUT | awk -F"_" '{printf "%s %s\n",$2,$NF >> "energy.txt";}'
done

#awk -F"_" '{print $NF}'
