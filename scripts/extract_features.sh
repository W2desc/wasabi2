#!/bin/sh

LABEL_NUM=19

if [ "$#" -eq 0 ]; then
  echo "1. data {cmu_park, cmu_urban, symphony}"
  echo "2. slice_id"
  exit 0
fi

if [ "$#" -ne 2 ]; then
  echo "Error: wrong number of arguments"
  echo "1. data {cmu_park, cmu_urban, symphony}"
  echo "2. slice_id"
  exit 0
fi

data="$1"
slice_id="$2"

if [ "$data" = "cmu_park" ]; then
  bin_example=cmu
  cam_id="0 1"
  if [ "$slice_id" -eq -1 ]; then
    slice_id="22 23 24 25"
  fi
elif [ "$data" = "cmu_urban" ]; then
  bin_example=cmu
  cam_id="0 1"
  if [ "$slice_id" -eq -1 ]; then
    slice_id="6 7 8"
  fi
elif [ "$data" = "symphony" ]; then
  bin_example=symphony
  cam_id="0"
  if [ "$slice_id" -eq -1 ]; then
    slice_id="0"
  fi
else 
  echo "Error: wrong data "$data". Chose in {cmu_park, cmu_urban, symphony}."
  exit 1
fi

survey_id="-1 0 1 2 3 4 5 6 7 8 9"


# create output directory for local features
if [ "$data" = "cmu_park" ] || [ "$data" = "cmu_urban" ]; then
  for sid in $slice_id
  do
    echo "sid: "$sid""
    i=0
    while [ "$i" -lt "$LABEL_NUM" ];
    do
      for split in "query" "database"
      do
        #echo "mkdir -p res/"$data"/features/des/"$i"/slice"$sid"/"$split""
        mkdir -p res/"$data"/features/des/"$i"/slice"$sid"/"$split"
        ##mkdir -p res/"$data"/features/kp/"$i"/slice"$sid"/"$split"
      done
      i=$((i+1))
    done
  done

else # symphony
  i=0
  while [ "$i" -lt "$LABEL_NUM" ];
  do
    while read -r line
    do
      dirname="$line"
      if ! [ -d "res/symphony/features"/des/"$i"/"$dirname" ]; then
        #echo "mkdir -p "res/symphony/features"/des/"$i"/"$dirname""
        mkdir -p "res/symphony/features"/des/"$i"/"$dirname"
        ##mkdir -p "res/symphony/features"/kp/"$i"/"$dirname"
      fi
    done < meta/symphony/val_dir.txt
    i=$((i+1))
  done
fi


# extract local features
for sid in $slice_id
do
  for cid in $cam_id
  do
    for surid in $survey_id
    do
      echo "Slice: "$sid" Cam: "$cid" Survey: "$surid""

      #echo "#./build/examples/"$bin_example" "$sid" "$cid" "$surid""
      ./build/examples/"$bin_example" "$sid" "$cid" "$surid"

      if [ "$?" -ne 0 ]; then
        echo "Error during run "$sid" "$cid" "$surid""
        exit 1
      fi
    done
  done
done
