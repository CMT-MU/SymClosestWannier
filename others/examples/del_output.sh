#!/bin/bash

files="./*"

dirary=()
for filepath in $files; do
  if [ -d $filepath ] ; then
    dirary+=("$filepath")
  fi
done


for seedname in ${dirary[@]}; do
  cd $seedname
  seedname=`echo $seedname | awk '{print substr($0, 3)}'`
  ls | grep -i -v -E "$seedname.amn|$seedname.eig|$seedname.win|$seedname.cwin|$seedname.band.gnu|$seedname.band.gnu.dat|sym" | xargs rm -rf
  if [ -d sym ]; then
    find sym -name "*.txt" | xargs rm
    find sym -name "*.dat" | xargs rm
    find sym -name "*.gnu" | xargs rm
    find sym -name "*.eps" | xargs rm
    find sym -name "*_info.py" | xargs rm
  fi
  cd ..
done


