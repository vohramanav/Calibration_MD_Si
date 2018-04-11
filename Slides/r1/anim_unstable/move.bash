#!/bin/bash

for i in {1..49}
  do
    let j=10*$i
    if [ $j -ge 10 ] && [ $j -lt 100 ]
    then
      mv animation00$j.png snap$i.png
    fi

    if [ $j -ge 100 ]
    then
      mv animation0$j.png snap$i.png
    fi
  done

  mv animation0000.png snap0.png
