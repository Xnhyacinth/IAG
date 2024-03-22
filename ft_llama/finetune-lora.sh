set -e

if [[ $# -lt 4 ]]; then
  echo "It requires 4 args to run the script and the current # of bash args: $#"
  echo "image.sh <num_gpus> <gpus>"
  exit 1
fi

num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
dataset=${3:-"NQ-cbqa"}
batch_size=${4:-"4"}
block_size=${5:-"1024"}
output_model=output/${dataset}/hylora
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
random_port=$((RANDOM%(65535-1024+1)+1024))
while [[ $(ss -tln | grep ":$random_port") ]]; do
    random_port=$((RANDOM%(65535-1024+1)+1024))
done

export NCCL_P2P_DISABLE=1
deepspeed --include localhost:${gpus} --master_port ${random_port} hylora.py \
    --train_name hyperlora \
    --model_name_or_path models/llama2/13b \
    --tokenizer_name /models/llama2/13b \
    --train_files data/${dataset}/train.json \
    --validation_files  /data/${dataset}/eval.json \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 8 \
    --warmup_steps 400 \
    --load_in_bits 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 8 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 1 \
    --seed 29 \
    --disable_tqdm false \
    --ddp_find_unused_parameters true \
    --block_size ${block_size} \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    --do_train \
    --do_eval \
    # --bf16 true
    # --gradient_checkpointing \
