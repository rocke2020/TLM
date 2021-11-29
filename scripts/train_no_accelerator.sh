#!/bin/bash
TASK=sciie
SCALE=small
OUTPUT_DIR=./results
SAVENAME=$TASK-$SCALE-scale
MLM_WEIGHT=20
EXTERNAL_RATIO=999
LR=1e-4
WD=0.01
WARMUP=10000
dataset_dir=/data2/corpus/nlp_corpus/tlm

if [[ $TASK == "imdb" ]]
then
MAXLEN=512
else 
MAXLEN=128
fi

mkdir -p $OUTPUT_DIR

nohup python src/run_no_accelerator.py \
    --max_train_steps 150000 \
    --steps_to_eval 100000 \
    --steps_to_save 50000 \
    --steps_to_log 100 \
    --external_dataset_name small_external.csv \
    --preprocessing_num_workers 4 \
    --max_length $MAXLEN \
    --max_ckpts_to_keep 3 \
    --pad_to_max_length \
    --config_dir yxchar/tlm-${TASK}-${SCALE}-scale \
    --dataset_dir $dataset_dir \
    --from_scratch \
    --output_dir $OUTPUT_DIR/$SAVENAME \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --cuda_devices 0 \
    --task_name $TASK \
    --save_final \
    --mlm_weight $MLM_WEIGHT \
    --external_ratio $EXTERNAL_RATIO \
    --mask_task \
    --weight_decay $WD \
    --learning_rate $LR \
    --num_warmup_steps $WARMUP \
    > log.log 2>&1 &
