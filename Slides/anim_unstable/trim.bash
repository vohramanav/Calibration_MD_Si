#!/bin/bash
for i in {1..49}
  do
    let j=10*$i
    if [ $j -ge 10 ] && [ $j -lt 100 ]
    then
      convert animation00$j.png -trim animation00$j.png
    fi
    
    if [ $j -ge 100 ] 
    then
      convert animation0$j.png -trim animation0$j.png
    fi

  done
  
  convert animation0000.png -trim animation0000.png



