# python src/preprocess.py \
#     ----model_id t5-base \
#     --train_data keywords_data/TQA/train.json \
#     --eval_data keywords_data/TQA/dev.json \

python data/process.py \
    --model_id /home/huanxuan/FiD/models/t5-base \
    --train_data data/NQ/train.json \
    --dev_data data/NQ/dev.json \
    --test_data data/NQ/test.json \
    --save_dataset_path dataset/NQ/