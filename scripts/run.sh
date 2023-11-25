#<num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher>
# lora
# nohup bash scripts/image.sh 1 3 1 1000 32 50000 lora no no > logs/lora_nods_notrain.log 2>&1 &

# hylora_kl
# nohup bash scripts/image.sh 2 0,1 4 1000 4 50000 hylora kl no > logs/hylora_kl_notrain.log 2>&1 &

# lora_kl
# nohup bash scripts/image.sh 2 4,5 4 1000 4 50000 lora kl no > logs/lora_kl_notrain.log 2>&1 &

# hylora_all_gold
# nohup bash scripts/image.sh 2 5,6 4 1000 2 50000 hylora all no gold 1e-3 > logs/hylora_all_notrain_gold_lr1e-3.log 2>&1 &

# hylora_all_gold_train
nohup bash scripts/image.sh 2 2,3 8 1000 1 50000 hylora all full gold 1e-3 > logs/hylora_all_train_gold_lr1e-3_full.log 2>&1 &