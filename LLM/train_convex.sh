export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export CXX=g++

export MASTER_ADDR="${CHIEF_IP:=localhost}"
MASTER_PORT=$((1 + $RANDOM % 99999))

gpu_num=8
if [ $gpu_num -eq 4 ]; then
    accum_num=32
elif [ $gpu_num -eq 8 ]; then
    accum_num=16
else
    echo "Unsupported GPU number: $gpu_num"
    exit 1
fi

model_name=llama_7b_convex
train_path=train/run_convex.py 
premodel=llama_7b_pre
model_save=checkpoints/$model_name
LOG_FILE=checkpoints/log.${model_name}

export TRANSFORMERS_CACHE=train/cache/
export HF_HOME=train/cache/
export TORCH_EXTENSIONS_DIR=train/cache/torch_extension/${model_name}
export OMP_NUM_THREADS=20
TOKENIZERS_PARALLELISM=false
# HOST_NUM will be 1
HOST_NUM=1
INDEX=0

train_files=train/alpaca.json
python -u -m torch.distributed.launch --nproc_per_node $gpu_num --master_port=$MASTER_PORT  --use_env \
    ${train_path} \
    --deepspeed train/deepspeed/deepspeed_config_zero3.json \
    --model_name_or_path ${premodel} \
    --train_file $train_files \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $accum_num \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 50 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --block_size 768 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing False \
    --output_dir ${model_save} \
    --cache_dir ${data_dir}/cache/ \
    --freeze_emb True \
    --overwrite_output_dir \
    --overwrite_cache \
    2>&1 |tee ${LOG_FILE}
