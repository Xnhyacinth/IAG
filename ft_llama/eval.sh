
dataset=NQ-image
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --base_model models/llama2/7b \
    --lora_weights output/${dataset}/hylora/ \
    --load_8bit \
    --test_file data/${dataset}/test.json \
    --batch_size 4

