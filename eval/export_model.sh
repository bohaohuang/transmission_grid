#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/home/jordan/Bohao/code/models/research:/home/jordan/Bohao/code/models/research/slim

MODEL_NAME=$1
MODEL_ID=$2
CONFIG_DIR_LOCAL=$3
export CUDA_VISIBLE_DEVICES=$4
python /home/jordan/Bohao/code/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${CONFIG_DIR_LOCAL}\
    --trained_checkpoint_prefix /media/ei-edl01/user/bh163/models/eccv/towers/${MODEL_NAME}/model.ckpt-${MODEL_ID} \
    --output_directory /media/ei-edl01/user/bh163/models/eccv/export_model/$MODEL_NAME
