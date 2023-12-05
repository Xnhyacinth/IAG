#<num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher>

# hylora_all_base
# nohup bash scripts/image.sh 2 2,3 4 1000 2 50000 hylora all no gen 1e-3 10 base NQ train all > logs/hylora_all_notrain_gen_lr1e-3_base_10_alllayers.log 2>&1 &

# hylora_hd_base_5
# nohup bash scripts/image.sh 2 2,3 4 1000 2 50000 hylora hd no gen 1e-3 10 base NQ > logs/hylora_hd_notrain_gen_lr1e-3_base_10.log 2>&1 &

# hylora_kl_large
# nohup bash scripts/image.sh 2 1,6 4 2500 2 50000 hylora kl no gen 1e-4 10 large NQ train all > logs/hylora_kl_notrain_gen_lr1e-4_large_10_all.log 2>&1 &
# nohup bash scripts/image.sh 2 1,6 4 2500 2 50000 hylora kl no gen 1e-4 10 large TQA train all > logs_tqa/hylora_kl_notrain_gen_lr1e-4_large_10_all.log 2>&1 &

# hylora_cbqa
# nohup bash scripts/image.sh 2 0,1 4 1000 2 50000 hylora no no cbqa 1e-3 > logs/hylora_nods_notrain_cbqa_lr1e-3.log 2>&1 &

# tqa_hylora_kl_gold
# nohup bash scripts/image.sh 2 2,7 4 1000 2 50000 hylora kl no gold 1e-3 1 TQA > logs_tqa/hylora_kl_notrain_gold_lr1e-3.log 2>&1 &

# tqa_hylora_cbqa
# nohup bash scripts/image.sh 2 2,3 4 1000 2 50000 hylora all no gen 1e-3 5 base TQA train > logs_tqa/hylora_all_notrain_gen_lr1e-3_base_5.log 2>&1 &

# tqa_hylora_gen_kl
nohup bash scripts/image.sh 2 4,5 4 2500 2 50000 hylora kl no gen 1e-3 50 base TQA train > logs_tqa/hylora_kl_notrain_gen_lr1e-3_base_50.log 2>&1 &

# test 
# bash scripts/image.sh 2 3,4 4 1000 64 50000 hylora kl no gen 1e-3 5 base NQ test