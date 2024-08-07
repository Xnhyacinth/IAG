set -e

if [[ $# -lt 22 ]]; then
  echo "It requires 22 args to run the script and the current # of bash args: $#"
  echo "image.sh <num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher> <use_gold> <lr> <num_compress> <size> 
  <dataset> <train> <select for alignment> <whether use context for hypernetwork> <num_docs for teacher> <rank> "
  exit 1
fi

num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
accumulate_grad_batches=${3:-"1"}
echo "accumulate_grad_batches: ${accumulate_grad_batches}"
val_check_interval=${4:-"1000"}
y_or_n=`echo $val_check_interval 1.0 | awk '{if($1 > $2) print 1; else print 0;}'`
if [ $y_or_n -eq 1 ];then
  val_check_interval=`expr ${val_check_interval} \* ${accumulate_grad_batches}`
fi
echo "val_check_interval: ${val_check_interval}"
batch_size=${5:-"8"}
max_steps=${6:-"100000"}
name=${7:-"hylora"}
distill=${8:-"kl"}
train_teacher=${9:-"no"}
gold=${10:-"gen"}
lr=${11:-"1e-4"}
n_c=${12:-"1"}
size=${13:-"base"}
dataset=${14:-"NQ"}
train=${15:-"train"}
select=${16:-"all"}
use_context=${17:-"no"}
n_context=${18:-"100"}
lora_rank=${19:-"32"}
pre=${20:-"no"}
test_fid=${21:-"No"}
model_dataset=${22:-"NQ"}

default_root_dir="output_${dataset}"
teacher_model="pretrained_models/nq_reader_$size"
echo "batch_size: ${batch_size}"
echo "max_steps: ${max_steps}"
echo "lr: ${lr}"
export MASTER_ADDR=localhost
random_port=$((RANDOM%(65535-1024+1)+1024))
while [[ $(ss -tln | grep ":$random_port") ]]; do
    random_port=$((RANDOM%(65535-1024+1)+1024))
done
export MASTER_PORT=${random_port}
context_maxlength=512
if [ "$name" = "lora" ];then
  extra_args="--lora"
fi
if [ "$name" = "lora_hyfn" ];then
  extra_args="--lora"
fi
if [ "$name" = "lora_ffn" ];then
  extra_args="--lora"
fi
if [ "$name" = "hylora" ];then
  extra_args="--hylora"
fi
if [ "$name" = "hyperlora" ];then
  extra_args="--hylora"
fi
if [ "$name" = "hyperlora_hyfn" ];then
  extra_args="--hylora"
fi
if [ "$name" = "hyperlora_ffn" ];then
  extra_args="--hylora"
fi
if [ "$distill" = "distill" ];then
  extra_args="$extra_args --do_distill"
  name="${name}_distill"
fi
if [ "$distill" = "kl" ];then
  extra_args="$extra_args --use_kl --do_distill"
  name="${name}_kl"
fi
if [ "$distill" = "hd" ];then
  extra_args="$extra_args --do_distill --use_attn --use_hidden"
  name="${name}_hd"
  context_maxlength=256
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
fi
if [ "$gold" = "cbqa" ];then
  extra_args="$extra_args --cbqa"
fi
if [ "$dataset" == "TQA" ];then
  teacher_model="pretrained_models/tqa_reader_$size"
fi
if [ "$dataset" == "WQ" ];then
  teacher_model="pretrained_models/wq_reader_$size"
fi
hg_datapath="dataset/Image/$dataset"
hg_datapath="${hg_datapath}-pred"
if [ "$pre" = "pred" ];then
  name="${name}_pred"
  hg_datapath="${hg_datapath}-pred"
  extra_args="$extra_args --pred"
fi
if [ "$pre" = "gen" ];then
  name="${name}_gen"
  hg_datapath="${hg_datapath}-gen"
  extra_args="$extra_args --pred"
fi
extra_args="$extra_args --hg_datapath ${hg_datapath} --n_c ${n_c}"

if [ "$n_c" != "0" ];then
  name="${name}_hg_ctxs${n_c}"
  echo "data from ${hg_datapath}"
fi
name="${name}_${gold}_lr${lr}_${size}_rank${lora_rank}"
file=main.py
if [ "$train" = "test" ];then
  file=test.py
  load_checkpoints_path="output_NQ/fid_hg_ctxs5_cbqa_lr1e-3_base_alllayers_5/ckpt/epoch=14-step=37110-val_em=24.47.ckpt"
  default_root_dir="output_test_${model_dataset}"
  name="${name}_test_${dataset}"
  extra_args="$extra_args --load_checkpoints_path $load_checkpoints_path"
fi
if [ "$select" = "select" ];then
  extra_args="$extra_args --select"
fi
if [ "$use_context" = "use_context" ];then
  extra_args="$extra_args --use_context"
  name="${name}_usecontext"
fi
name="${name}_${select}layers_${n_context}"
if [ "$test_fid" = "yes" ];then
  extra_args="$extra_args --test_fid"
  name="${name}_usefid"
fi
extra_args="$extra_args --name $name" # --resume_from_checkpoint output/hylora_all_lr1e-3/ckpt/last.ckpt
echo "name: ${name}"
echo "default_root_dir: ${default_root_dir}"
export NCCL_P2P_DISABLE=1
deepspeed --include localhost:$gpus --master_port $MASTER_PORT ${file} \
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
        --num_workers 5 \
        --default_root_dir ${default_root_dir} \
        --n_context ${n_context} \
        --warmup_ratio 0.08 \
        --train_data data/$dataset/train.json \
        --eval_data data/$dataset/dev.json \
        --test_data data/$dataset/test.json \
        --model_name t5-${size} \
        --teacher_model ${teacher_model} \
        --t_learning_rate 5e-05 \
        --alpha_kd 0.4 \
        --temperature 3.0 \
        --save_top_k 1 \
        --r ${lora_rank} \
        --lora_rank ${lora_rank} \
        ${extra_args}
                # --resume_from_checkpoint None \
        # data/NQ/train.json 
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
