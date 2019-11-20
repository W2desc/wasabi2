#!/bin/sh

top_k=20 # number of db img to retrieve for each query img

if [ $# -eq 0 ]; then
  echo "Usage"
  echo "1. data {cmu_park, cmu_urban, symphony}"
  echo "2. slice_id"
  exit 0
fi

if [ $# -ne 2 ]; then
  echo "Bad number of arguments"
  echo "1. data {cmu_park, cmu_urban, symphony}"
  echo "2. slice_id"
  exit 1
fi

data="$1"
slice_id="$2"

if [ "$data" = "cmu_park" ] || [ "$data" = "cmu_urban" ]; then
  img_dir=meta/cmu/bgr
  meta_dir=meta/cmu/surveys/
  seg_dir=meta/cmu/seg
  survey_type=cmu
elif [ "$data" = "symphony" ]; then
  img_dir=meta/"$data"/bgr
  meta_dir=meta/"$data"/surveys/
  seg_dir=meta/"$data"/seg
  survey_type=symphony
else 
  echo "Error: wrong data "$data". Chose in {cmu_park, cmu_urban, symphony}."
  exit 1
fi

if [ "$slice_id" -eq -1 ]; then # retrieve all slices
  loop_file=scripts/"$data".txt
else
  loop_file=scripts/"$data"_"$slice_id".txt
fi

des_dir=res/"$data"/features/
if ! [ -d "$des_dir" ]; then
  echo "Error: you must first extract local features to disk."
  exit 1
fi

centroids=meta/codebooks/"$data"_wasabi2/centroids.txt
out_dir=res/"$data"/


if [ -d "$out_dir"/retrieval ]; then
    while true; do
        read -p ""$out_dir"/retrieval already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) 
              rm -rf "$out_dir"/retrieval/; 
              rm -rf "$out_dir"/perf/; 
              mkdir -p "$out_dir"/retrieval; 
              mkdir -p "$out_dir"/perf; 
              break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
    mkdir -p "$out_dir"/retrieval
    mkdir -p "$out_dir"/perf
fi


while read -r line
do
  slice_id=$(echo "$line" | cut -d' ' -f1)
  cam_id=$(echo "$line" | cut -d' ' -f2)
  for survey_id in $(echo "$line" | cut -d' ' -f 3-)
  do
      echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"
      python3 -m pywasabi2.retrieve \
        --trial -1 \
        --dist_pos 5 \
        --top_k "$top_k" \
        --lf_mode acc \
        --max_num_feat -1 \
        --agg_mode vlad \
        --centroids "$centroids" \
        --vlad_norm ssr \
        --data "$data" \
        --slice_id "$slice_id" \
        --cam_id "$cam_id" \
        --survey_id "$survey_id" \
        --img_dir "$img_dir" \
        --meta_dir "$meta_dir" \
        --des_dir "$des_dir" \
        --n_words -1 \
        --resize 0 \
        --w 640 \
        --h 480
      
      if [ "$?" -ne 0 ]; then
        echo "Error in slice "$slice_id" cam "$cam_id" survey_id "$survey_id""
        exit 1
      fi
    done
done < "$loop_file"



## LAKE
#img_dir="$WS_DIR"datasets/Extended-CMU-Seasons/
#meta_dir=meta/symphony/surveys/
#
#while read -r line
#do
#  slice_id=$(echo "$line" | cut -d' ' -f1)
#  cam_id=$(echo "$line" | cut -d' ' -f2)
#  for survey_id in $(echo "$line" | cut -d' ' -f 3-)
#  do
#      echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"
#      #python3 -m methods.vlad_bow.retrieve \
#      python3 -m methods.vlad_bow_acc.retrieve \
#        --trial "$trial" \
#        --dist_pos 5 \
#        --top_k "$top_k" \
#        --lf_mode acc \
#        --max_num_feat "$max_num_feat" \
#        --agg_mode vlad \
#        --centroids "$centroids" \
#        --vlad_norm ssr \
#        --data symphony \
#        --slice_id "$slice_id" \
#        --cam_id "$cam_id" \
#        --survey_id "$survey_id" \
#        --img_dir "$img_dir" \
#        --meta_dir "$meta_dir" \
#        --n_words "$n_words" \
#        --resize 0 \
#        --w 640 \
#        --h 480
#      
#      if [ "$?" -ne 0 ]; then
#        echo "Error in slice "$slice_id" cam "$cam_id" survey_id "$survey_id""
#        exit 1
#      fi
#    done
#done < "$instance"
#
#
