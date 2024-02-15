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
  ls | grep -i -v -E "$seedname.cwin|$seedname.win|$seedname.nnkp|$seedname.eig|$seedname.amn|$seedname.mmn|$seedname.band.gnu|$seedname.band.gnu.dat|sym" | xargs rm -rf
  if [ -d sym ]; then
    find sym -name "*.eps" | xargs rm
    find sym -name "*.txt" | xargs rm
    find sym -name "*.cw" | xargs rm
    find sym -name "*.cwout" | xargs rm
    find sym -name "*.hdf5" | xargs rm
    find sym -name "plot_band.gnu" | xargs rm
  fi
  cd ..
done


