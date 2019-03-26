#!/bin/sh

PWD=$(cd -P -- "$(dirname -- "$0")" && pwd -P) # 这一行命令可以得到 run.sh 所在目录
ls -l $PWD
echo ${PWD}
echo ${DATA_DIR}
echo ${OUTPUT_DIR}
echo ${MODEL_DIR}

ENTER_DIR="/src/"
normal_size=270
channels=3
epochs=100
batch_size=80
classes=2
data_augmentation=True
model_name="ResNet_50"

python ${PWD}${ENTER_DIR}train.py --data_dir ${DATA_DIR}/machinelearn/train/ --output_dir ${OUTPUT_DIR} --model_dir ${MODEL_DIR} --normal_size ${normal_size} --channels ${channels} --epochs ${epochs} --batch_size ${batch_size} --classes ${classes} --data_augmentation ${data_augmentation} --model_name ${model_name}