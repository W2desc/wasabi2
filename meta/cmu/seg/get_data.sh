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

# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

if [ "$slice_id" -eq -1 ]; then # dowload all segmentation data (<1G)

  while read -r line
  do
    filename=$(echo "$line" | cut -d' ' -f1)
    fileid=$(echo "$line" | cut -d' ' -f2)
    echo "filename: "$filename""
    echo "fileid: "$fileid""
    #gdrive_download "$fileid" "$filename"
    #tar -xvzf "$filename"
    #rm -f "$filename"
  done < data_links.txt

else # download only slice
  if [ "$slice_id" -lt 2 ] || [ "$slice_id" -gt 26 ]; then 
    echo "Error: this slice does note exists. Choose slice_id in [2,25]"
    exit 1
  fi

  filename=slice"$slice_id".tar.gz
  fileid=$(grep "$filename" ./data_links.txt | cut -d' ' -f2)
  echo "fileid: "$fileid""

  #gdrive_download "$fileid" "$filename"
  #tar -xvzf "$filename"
  #rm -f "$filename"
fi
