#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name trainKfold_sep.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="trainKfold_sep.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

readonly IMAGE_PATH_LAYER_1=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_path_layer_1"))
readonly IMAGE_PATH_LAYER_2=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_path_layer_2"))
readonly IMAGE_PATH_THIN=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_path_thin"))
readonly LABEL_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".label_path"))
readonly MODEL_SAVEPATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_savepath"))
readonly TRAIN_LISTS=$(cat ${JSON_FILE} | jq -r ".train_lists")
readonly VAL_LISTS=$(cat ${JSON_FILE} | jq -r ".val_lists")
readonly LOG=$(eval echo $(cat ${JSON_FILE} | jq -r ".log"))
readonly IN_CHANNEL_1=$(cat ${JSON_FILE} | jq -r ".in_channel_1")
readonly IN_CHANNEL_2=$(cat ${JSON_FILE} | jq -r ".in_channel_2")
readonly IN_CHANNEL_THIN=$(cat ${JSON_FILE} | jq -r ".in_channel_thin")
readonly NUM_CLASS=$(cat ${JSON_FILE} | jq -r ".num_class")
readonly LEARNING_RATE=$(cat ${JSON_FILE} | jq -r ".learning_rate")
readonly BATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".batch_size")
readonly NUM_WORKERS=$(cat ${JSON_FILE} | jq -r ".num_workers")
readonly EPOCH=$(cat ${JSON_FILE} | jq -r ".epoch")
readonly GPU_IDS=$(cat ${JSON_FILE} | jq -r ".gpu_ids")
readonly API_KEY=$(cat ${JSON_FILE} | jq -r ".api_key")
readonly PROJECT_NAME=$(cat ${JSON_FILE} | jq -r ".project_name")
readonly EXPERIMENT_NAME=$(cat ${JSON_FILE} | jq -r ".experiment_name")
readonly KFOLD_LIST=$(cat ${JSON_FILE} | jq -r ".train_lists | keys[]")

for kfold in ${KFOLD_LIST[@]}
do
    image_path_layer_1="${IMAGE_PATH_LAYER_1}/${kfold}"
    image_path_layer_2="${IMAGE_PATH_LAYER_2}/${kfold}"
    image_path_thin="${IMAGE_PATH_THIN}/${kfold}"
    model_savepath="${MODEL_SAVEPATH}/${kfold}"
    train_list=$(echo ${TRAIN_LISTS} | jq -r ".$kfold")
    val_list=$(echo ${VAL_LISTS} | jq -r ".$kfold")
    experiment_name="${EXPERIMENT_NAME}_${kfold}"
    log="${LOG}/${kfold}"

    echo "image_path_layer_1:${image_path_layer_1}"
    echo "image_path_layer_2:${image_path_layer_2}"
    echo "image_path_thin:${image_path_thin}"
    echo "LABEL_PATH:${LABEL_PATH}"
    echo "model_savepath:${model_savepath}"
    echo "train_list:${train_list}"
    echo "val_list:${val_list}"
    echo "LOG:${LOG}"
    echo "IN_CHANNEL_1:${IN_CHANNEL_1}"
    echo "IN_CHANNEL_2:${IN_CHANNEL_2}"
    echo "IN_CHANNEL_THIN:${IN_CHANNEL_THIN}"
    echo "NUM_CLASS:${NUM_CLASS}"
    echo "LEARNING_RATE:${LEARNING_RATE}"
    echo "BATCH_SIZE:${BATCH_SIZE}"
    echo "NUM_WORKERS:${NUM_WORKERS}"
    echo "EPOCH:${EPOCH}"
    echo "GPU_IDS:${GPU_IDS}"
    echo "API_KEY:${API_KEY}"
    echo "PROJECT_NAME:${PROJECT_NAME}"
    echo "experiment_name:${experiment_name}"

    python3 train_sep.py ${image_path_layer_1} ${image_path_layer_2} ${image_path_thin} ${LABEL_PATH} ${model_savepath} --train_list ${train_list} --val_list ${val_list} --log ${log} --in_channel_1 ${IN_CHANNEL_1} --in_channel_2 ${IN_CHANNEL_2} --in_channel_thin ${IN_CHANNEL_THIN} --num_class ${NUM_CLASS} --lr ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --epoch ${EPOCH} --gpu_ids ${GPU_IDS} --api_key ${API_KEY} --project_name ${PROJECT_NAME} --experiment_name ${experiment_name}

done
