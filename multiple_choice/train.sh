#!/bin/bash

display_usage() { 
	echo -e "\nUsage: $0 <gpus> <model_name_or_path> <batch_size> \n" 
	} 
# if less than two arguments supplied, display usage 
if [  $# -le 2 ] 
then 
    display_usage
    exit 1
fi 

# check whether user had supplied -h or --help . If yes display usage 
if [[ ( $@ == "--help") ||  $@ == "-h" ]] 
then 
    display_usage
    exit 0
fi 

BATCH_SIZE=$3
BATCH_WANTED=8 # 8 if 2 GPUs
GRAD_ACC_STEPS=$((BATCH_WANTED / BATCH_SIZE))

echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 python run_multiple_choice.py \
--model_name_or_path $2 \
--task_name persona \
--output_dir experiments \
--do_eval \
--do_train \
--warmup_steps 200 \
--per_device_train_batch_size $3 \
--per_device_eval_batch_size $3 \
--gradient_accumulation_steps $GRAD_ACC_STEPS
#--auto_find_batch_size \

