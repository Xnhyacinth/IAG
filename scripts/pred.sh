set -e


if [[ $# -lt 2 ]]; then
  echo "It requires 2 args to run the script and the current # of bash args: $#"
  echo "<num_gpus> <gpus>"
  exit 1
fi

num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
# model="xgen-7b-8k"
# llama2-7b-chat-4k, chatglm2-6b, longchat-v1.5-7b-32k, xgen-7b-8k, openbuddy-llama-7b-v4-fp16
export CUDA_VISIBLE_DEVICES=${gpus}
hg_datapath=dataset/Image/
# models=("xgen-7b-8k" "llama2-7b-chat-4k" "chatglm2-6b" "openbuddy-llama-7b-v4-fp16" "longchat-v1.5-7b-32k")
# models=("xgen-7b-8k" "llama2-7b-chat-4k")
models=("llama2-13b")
# models=("xgen-7b-8k")
for model in "${models[@]}"; do
    python -u pred.py \
        --input-path ${hg_datapath} \
        --num-gpus ${num_gpus} \
        --max-new-tokens 100 \
        --batch-size 8 \
        --max-memory-per-gpu 24 \
        --model ${model} \
        --n_c 5 \
        --image
        # --image closedbook
  done