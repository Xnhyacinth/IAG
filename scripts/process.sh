
python data/process.py \
    --model_id t5-base \
    --train_data data/NQ/train.json \
    --dev_data data/NQ/dev.json \
    --test_data data/NQ/test.json \
    --save_dataset_path dataset/NQ/