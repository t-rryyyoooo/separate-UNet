#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createPatch.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file.
if [ $which = "y" ];then
 JSON_NAME="createPatch.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

# From json file, read required variables.
readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly MODELWEIGHT_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".modelweight_path"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")
readonly OUTPUT_LAYER=$(cat ${JSON_FILE} | jq -r ".output_layer")
readonly INPUT_SIZE=$(cat ${JSON_FILE} | jq -r ".input_size")
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly GPU_IDS=$(cat ${JSON_FILE} | jq -r ".gpu_ids")
readonly LOG_FILE=$(eval echo $(cat ${JSON_FILE} | jq -r ".log_file"))

# Make directory to save LOG.
echo ${LOG_FILE}
mkdir -p `dirname ${LOG_FILE}`
date >> $LOG_FILE

for number in ${NUM_ARRAY[@]}
do
 image_path="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
 save_path="${SAVE_DIRECTORY}/layer_${OUTPUT_LAYER}/case_${number}"

 echo "image_path:${image_path}"
 echo "MODELWEIGHT_PATH:${MODELWEIGHT_PATH}"
 echo "save_path:${save_path}"
 echo "OUTPUT_LAYER:${OUTPUT_LAYER}"
 echo "INPUT_SIZE:${INPUT_SIZE}"
 echo "OVERLAP:${OVERLAP}"
 echo "GPU_IDS:${GPU_IDS}"

python3 createPatch.py ${image_path} ${MODELWEIGHT_PATH} ${save_path} --output_layer ${OUTPUT_LAYER} --input_size ${INPUT_SIZE} --overlap ${OVERLAP} --gpu_ids ${GPU_IDS}

 # Judge if it works.
 if [ $? -eq 0 ]; then
  echo "case_${number} done."
 
 else
  echo "case_${number}" >> $LOG_FILE
  echo "case_${number} failed"
 
 fi

done


