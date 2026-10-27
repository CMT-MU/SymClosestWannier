#!/bin/bash

# remove the outputs of run_examples.py, keeping the inputs and the SAMB data (seedname/seedname.pkl).

for dir in ./*/; do
  seedname=$(basename "$dir")
  (
    cd "$dir" || exit
    ls | grep -v -x -E "$seedname\.(cwin|win|nnkp|eig|amn|mmn|band\.gnu|band\.gnu\.dat|py)|$seedname|sym" | xargs rm -rf
    if [ -d "$seedname" ]; then
      find "$seedname" -type f ! -name "$seedname.pkl" -delete
    fi
  )
done
