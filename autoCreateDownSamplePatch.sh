#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createDownSamplePatch.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="createDownSamplePatch.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly IMAGE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly PLANE_SIZE=$(cat ${JSON_FILE} | jq -r ".plane_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly IS_LABEL=$(cat ${JSON_FILE} | jq -r ".is_label")
readonly NUM_DOWN=$(cat ${JSON_FILE} | jq -r ".num_down")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")


for number in ${NUM_ARRAY[@]}
do
    image_path="${IMAGE_DIRECTORY}/case_${number}/${IMAGE_NAME}"
    save_path="${SAVE_DIRECTORY}/case_${number}"
    echo "IMAGE_PATH:${image_path}"
    echo "SAVE_PATH:${save_path}"
    echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
    echo "OVERLAP:${OVERLAP}"
    echo "NUM_DOWN:${NUM_DOWN}"

    if [ $IS_LABEL = "No" ];then
    echo "IS_LABEL:No"
    python3 createDownSamplePatch.py ${image_path} ${save_path} --label_patch_size ${LABEL_PATCH_SIZE} --plane_size ${PLANE_SIZE} --overlap ${OVERLAP} --num_down ${NUM_DOWN} 
    else
    echo "IS_LABEL:Yes"
    python3 createDownSamplePatch.py ${image_path} ${save_path} --label_patch_size ${LABEL_PATCH_SIZE} --plane_size ${PLANE_SIZE} --overlap ${OVERLAP} --num_down ${NUM_DOWN} --is_label
    fi

# Judge if it works.
    if [ $? -eq 0 ]; then
     echo "Done."

    else
     echo "Fail"

    fi
done
