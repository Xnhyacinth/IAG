set -e

if [[ $# -lt 11 ]]; then
  echo "It requires 11 args to run the script and the current # of bash args: $#"
  echo "image.sh <num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher> <use_gold> <lr>"
  exit 1
fi

num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
accumulate_grad_batches=${3:-"1"}
echo "accumulate_grad_batches: ${accumulate_grad_batches}"
val_check_interval=${4:-"1000"}
val_check_interval=`expr ${val_check_interval} \* ${accumulate_grad_batches}`
echo "val_check_interval: ${val_check_interval}"
batch_size=${5:-"8"}
max_steps=${6:-"100000"}
name=${7:-"hylora"}
distill=${8:-"kl"}
train_teacher=${9:-"no"}
gold=${10:-"False"}
lr=${11:-"1e-4"}
n_c=${12:-"0"}
echo "batch_size: ${batch_size}"
echo "max_steps: ${max_steps}"
echo "lr: ${lr}"
export MASTER_ADDR=localhost
export MASTER_PORT="52995"

context_maxlength=512
if [ "$name" = "lora" ];then
  extra_args="--lora"
fi
if [ "$name" = "hylora" ];then
  extra_args="--hylora"
fi
if [ "$distill" = "kl" ];then
  extra_args="$extra_args --use_kl --do_distill"
  name="${name}_kl"
fi
if [ "$distill" = "all" ];then
  extra_args="$extra_args --use_kl --do_distill --use_attn --use_hidden"
  name="${name}_all"
  context_maxlength=256
fi
if [ "$train_teacher" = "train" ];then
  extra_args="$extra_args --train_teacher"
  name="${name}_train"
fi
if [ "$train_teacher" = "full" ];then
  extra_args="$extra_args --train_teacher"
  name="${name}_train_full"
fi
if [ "$gold" = "gold" ];then
  extra_args="$extra_args --gold"
  name="${name}_gold"
fi
name="${name}_lr${lr}"
if [ "$n_c" != "0" ];then
  hg_datapath="Xnhyacinth/NQ-Image"
  extra_args="$extra_args --hg_datapath ${hg_datapath} --n_c ${n_c}"
  name="${name}_hg_ctxs${n_c}"
fi
extra_args="$extra_args --name $name"
echo "name: ${name}"

deepspeed --include localhost:$gpus --master_port $MASTER_PORT main.py \
        --use_checkpoint \
        --accelerator gpu \
        --devices ${num_gpus} \
        --strategy ddp \
        --seed 29 \
        --precision bf16 \
        --accumulate_grad_batches ${accumulate_grad_batches} \
        --max_steps ${max_steps} \
        --lr ${lr} \
        --batch_size ${batch_size} \
        --weight_decay 0.01 \
        --text_maxlength 256 \
        --answer_maxlength 256 \
        --context_maxlength ${context_maxlength} \
        --val_check_interval ${val_check_interval} \
        --num_workers 4 \
        --default_root_dir output \
        --n_context 100 \
        --warmup_ratio 0.08 \
        --train_data data/NQ/train.json \
        --eval_data data/NQ/dev.json \
        --test_data data/NQ/test.json \
        --model_name t5-base \
        --teacher_model pretrained_models/nq_reader_base \
        --t_learning_rate 5e-05 \
        --alpha_kd 0.6 \
        --temperature 3.0 \
        --save_top_k 3 \
        ${extra_args}
        # --use_kl \
        # --lora \
        # --do_distill \
        # --use_attn \
        # --use_hidden \
        # --hylora \
        # --train_teacher \
        # --gradient_clip_val 0.1 \
        # \ deepspeed_stage_2
        # --use_lgtm \
        # --deepspeed ds_config.json
        # --use_lgtm \
        # --max_epochs 10 \ 
                # --auto_scale_batch_size binsearch \
        # --auto_lr_find True \
        # --per_device_train_batch_size 1 \
        # --per_device_eval_batch_size 1 \
        # --gradient_accumulation_steps 8 \
        # --gpus ${num_gpus} \
#t 5e-5 lr 1e-4
# my_distillation_c1_ds_1_all_rep_alllayers #/data2/huanxuan/FiD/nq