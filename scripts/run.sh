#<num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher>
# hylora_all_gold_train
# nohup bash scripts/image.sh 2 2,3 8 1000 1 50000 hylora all full gold 1e-3 > logs/hylora_all_train_gold_lr1e-3_full.log 2>&1 &

# hylora_cbqa
# nohup bash scripts/image.sh 2 0,1 4 1000 2 50000 hylora no no cbqa 1e-3 > logs/hylora_nods_notrain_cbqa_lr1e-3.log 2>&1 &

# hylora_cbqa
# nohup bash scripts/image.sh 2 0,1 4 1000 4 50000 fid no no cbqa 1e-3 1 > logs/fid_nods_notrain_cbqa_lr1e-3.log 2>&1 &

# hylora_all_gold
# nohup bash scripts/image.sh 2 5,6 4 1000 2 50000 hylora all no gold 1e-3 > logs/hylora_all_notrain_gold_lr1e-3.log 2>&1 &

# hylora_all
# nohup bash scripts/image.sh 2 5,6 4 1000 2 50000 hylora all no no 1e-3 1 > logs/hylora_all_notrain_gen_lr1e-3.log 2>&1 &

# hylora_all_gold_train
# nohup bash scripts/image.sh 2 2,3 8 1000 1 50000 hylora all full gold 1e-3 > logs/hylora_all_train_gold_lr1e-3_full.log 2>&1 &

# hylora_kl_large
# nohup bash scripts/image.sh 2 0,1 4 1000 2 50000 hylora kl no gold 1e-3 1 large NQ > logs/hylora_kl_notrain_gold_lr1e-3_large.log 2>&1 &

# hylora_kl_base_5
# nohup bash scripts/image.sh 2 0,1 4 1000 2 50000 hylora kl no gen 1e-3 5 base NQ > logs/hylora_kl_notrain_gen_lr1e-3_base_5.log 2>&1 &

# hylora_cbqa
# nohup bash scripts/image.sh 2 0,1 4 1000 2 50000 hylora no no cbqa 1e-3 > logs/hylora_nods_notrain_cbqa_lr1e-3.log 2>&1 &

# tqa_hylora_kl_gold
# nohup bash scripts/image.sh 2 2,7 4 1000 2 50000 hylora kl no gold 1e-3 1 TQA > logs_tqa/hylora_kl_notrain_gold_lr1e-3.log 2>&1 &


# tqa_hylora_kl_gold
# nohup bash scripts/image.sh 2 2,7 4 1000 2 50000 hylora kl no gold 1e-3 1 TQA > logs_tqa/hylora_kl_notrain_gold_lr1e-3.log 2>&1 &

# tqa_hylora_cbqa
# nohup bash scripts/image.sh 2 2,3 4 1000 2 50000 hylora all no gen 1e-3 5 base NQ train > logs/hylora_all_notrain_gen_lr1e-3_base_5.log 2>&1 &

# hylora_cbqa
# nohup bash scripts/image.sh 2 6,7 2 1000 8 50000 hylora no no cbqa 1e-3 1 base NQ train > logs/hylora_nods_notrain_cbqa_lr1e-3_base_1.log 2>&1 &

# test 
# bash scripts/image.sh 2 3,4 4 1000 64 50000 hylora kl no gen 1e-3 5 base NQ test