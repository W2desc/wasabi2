#!/bin/sh

if [ "$#" -eq 0 ]; then
  echo "1. slice id"
  exit 0
fi

if [ "$#" -ne 1 ]; then
  echo "Error: bad number of arguments."
  echo "1. slice id"
  exit 1
fi

slice_id="$1"

if [ "$slice_id" -eq -1 ]; then # dowload all data
  # be nice and warn the use it is in for a hell of a download
  while true; do
    read -p "WARNING: you are about to download 129.3G of data. Do you want to pursue ?"
    case $yn in
      [Yy]* ) break;;
      [Nn]* ) exit;;
      * ) * echo "Please answer yes (y) or no (n).";;
    esac
  done

  while read -r line
  do
    wget "$line"
  done < data_links.txt

else # download only slice
  if [ "$slice_id" -lt 2 ] || [ "$slice_id" -gt 26 ]; then 
    echo "Error: this slice does note exists. Choose slice_id in [2,25]"
    exit 1
  fi

  link=$(grep slice"$slice_id".tar$ ./data_links.txt)
  echo "link: "$link""

  wget "$link" 
  tar -xvf slice22.tar
  rm slice22.tar
fi
