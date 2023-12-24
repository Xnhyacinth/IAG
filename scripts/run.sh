#<num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher>
###
 # Copyright (c) 2023 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2023-12-23 02:00:10
### 
# nohup bash scripts/image.sh 1 4 2 0.5 8 70000 hyperlora kl no cbqa 1e-3 5 base NQ train all use_context 10 32 > logs/hyperlora_kl_notrain_cbqa_lr1e-3_base_5_all_use_context_70000_10_r32.log 2>&1 &

nohup bash scripts/image.sh 1 5 2 1.0 8 70000 hyperlora kl no cbqa 1e-3 5 base WQ train all use_context 10 32 > logs_wq/hyperlora_kl_notrain_cbqa_lr1e-3_base_5_all_use_context_70000_10_r32.log 2>&1 &



# hylora_all_base
# nohup bash scripts/image.sh 2 2,3 4 1000 2 50000 hylora all no gen 1e-3 10 base NQ train all > logs/hylora_all_notrain_gen_lr1e-3_base_10_alllayers.log 2>&1 &

# hylora_hd_base_5
# nohup bash scripts/image.sh 2 2,3 4 1000 2 50000 hylora hd no gen 1e-3 10 base NQ > logs/hylora_hd_notrain_gen_lr1e-3_base_10.log 2>&1 &

# fid
# nohup bash scripts/image.sh 2 0,1 4 0.5 4 40000 fid no no cbqa 1e-3 5 base NQ train all no 5 > logs/fid_nods_notrain_cbqa_lr1e-3_base_5_all_5.log 2>&1 &
# nohup bash scripts/image.sh 1 1 8 2500 1 50000 fid no no gen 1e-4 10 large TQA train all no 1 > logs_tqa/fid_nods_notrain_gen_lr1e-4_large_10_all_1.log 2>&1 &
# nohup bash scripts/image.sh 1 1 8 1.0 2 10000 fid no no gen 1e-4 10 base WQ train all no 10 > logs_wq/fid_nods_notrain_gen_lr1e-4_large_10_all_10.log 2>&1 &

# hylora_kl_large
# nohup bash scripts/image.sh 2 0,1 8 2500 1 50000 hylora kl no gen 1e-4 10 large NQ train all > logs/hylora_kl_notrain_gen_lr1e-4_large_10_all.log 2>&1 &
# nohup bash scripts/image.sh 2 4,5 8 2500 1 50000 hylora all no gen 1e-4 10 large NQ train select > logs/hylora_all_notrain_gen_lr1e-4_large_10_select.log 2>&1 &
# nohup bash scripts/image.sh 2 2,3 8 2500 1 50000 hylora kl no gen 1e-4 10 large TQA train all > logs_tqa/hylora_kl_notrain_gen_lr1e-4_large_10_all.log 2>&1 &

# hylora_kl_base_usecontext
# nohup bash scripts/image.sh 2 6,7 4 2500 2 50000 hylora kl no gen 1e-3 20 base NQ train all use_context > logs/hylora_kl_notrain_gen_lr1e-3_base_20_all_usecontext.log 2>&1 &
# nohup bash scripts/image.sh 2 2,3 4 2500 2 50000 hylora kl no gen 1e-3 20 base TQA train all use_context > logs_tqa/hylora_kl_notrain_gen_lr1e-3_base_20_all_usecontext.log 2>&1 &
# nohup bash scripts/image.sh 2 6,7 4 1.0 2 15000 hylora kl no gen 1e-3 10 base WQ train all use_context > logs_wq/hylora_kl_notrain_gen_lr1e-3_base_10_all_usecontext.log 2>&1 &

# hylora_kl_base_usecontext_cbqa
# nohup bash scripts/image.sh 1 4 4 0.5 8 70000 hyperlora kl no cbqa 1e-3 5 base NQ train all use_context 10 32 > logs/hyperlora_kl_notrain_cbqa_lr1e-3_base_5_all_use_context_70000_10_r32.log 2>&1 &
# sleep 10
# nohup bash scripts/image.sh 1 6 4 0.5 8 70000 hyperlora all no cbqa 1e-3 5 base TQA train all use_context 10 128 > logs_tqa/hyperlora_all_notrain_cbqa_lr1e-3_base_5_all_use_context_70000_10_r128.log 2>&1 &
# sleep 10
# nohup bash scripts/image.sh 1 7 4 1.0 4 15000 hyperlora all no cbqa 1e-3 5 base WQ train all use_context 10 64 > logs_wq/hyperlora_all_notrain_cbqa_lr1e-3_base_5_all_use_context_15000_10_r64.log 2>&1 &

# hylora_kl_base
# nohup bash scripts/image.sh 2 6,7 4 2500 2 50000 hylora kl no gen 1e-3 20 base NQ train all no > logs/hylora_kl_notrain_gen_lr1e-3_base_20_all.log 2>&1 &
# nohup bash scripts/image.sh 2 2,3 4 2500 2 50000 hylora kl no gen 1e-3 20 base TQA train all no > logs_tqa/hylora_kl_notrain_gen_lr1e-3_base_20_all.log 2>&1 &
# nohup bash scripts/image.sh 2 2,3 4 1.0 2 15000 hylora kl no gen 1e-3 10 base WQ train all no > logs_wq/hylora_kl_notrain_gen_lr1e-3_base_10_all.log 2>&1 &

# hylora_cbqa
# nohup bash scripts/image.sh 1 0 8 1.0 2 120000 hylora no no cbqa 1e-4 5 large NQ train all no 1 > logs/hylora_nods_notrain_cbqa_lr1e-4_large_5_all.log 2>&1 &
# sleep 10
# nohup bash scripts/image.sh 1 1 8 1.0 2 120000 hylora no no cbqa 1e-4 5 large TQA train all no 1 > logs_tqa/hylora_nods_notrain_cbqa_lr1e-4_large_5_all.log 2>&1 &
# nohup bash scripts/image.sh 1 3 4 1.0 4 10000 hylora no no cbqa 1e-3 5 base WQ train all no 100 > logs_wq/hylora_nods_notrain_cbqa_lr1e-3_base_5_all_1000.log 2>&1 &

# lora_cbqa
# nohup bash scripts/image.sh 1 0 2 0.5 16 50000 lora no no cbqa 5e-4 10 base NQ train all no 1 > logs/lora_nods_notrain_cbqa_lr5e-4_base_10_all.log 2>&1 &
# sleep 5
# nohup bash scripts/image.sh 1 0 4 0.5 4 50000 lora no no cbqa 1e-4 10 large TQA train all no 1 > logs_tqa/lora_nods_notrain_cbqa_lr1e-4_large_10_all.log 2>&1 &
# # sleep 5
# nohup bash scripts/image.sh 1 6 1 1.0 16 10000 lora no no cbqa 1e-3 5 base WQ train all no 10 > logs_wq/lora_nods_notrain_cbqa_lr1e-3_base_5_all_10.log 2>&1 &
# sleep 5
# lora_gen
# nohup bash scripts/image.sh 1 5 8 1.0 2 50000 lora no no gen 1e-3 5 large NQ train all no 1 > logs/lora_nods_notrain_gen_lr1e-3_large_5_all.log 2>&1 &
# nohup bash scripts/image.sh 1 6 4 1.0 4 50000 lora no no gen 1e-3 5 large TQA train all no 1 > logs_tqa/lora_nods_notrain_gen_lr1e-3_large_5_all.log 2>&1 &
# sleep 5
# nohup bash scripts/image.sh 1 7 4 1.0 4 10000 lora no no gen 1e-4 5 large WQ train all no 1 > logs_wq/lora_nods_notrain_gen_lr1e-4_large_5_all.log 2>&1 &

# hylora_kl_gold
# nohup bash scripts/image.sh 2 2,7 4 1000 2 50000 hylora kl no gold 1e-3 1 TQA > logs_tqa/hylora_kl_notrain_gold_lr1e-3.log 2>&1 &

# test 
# bash scripts/image.sh 1 0 4 1.0 2 50000 hylora kl no gen 1e-3 5 base WQ test all use_context 10 yes TQA
# bash scripts/image.sh 1 0 4 1.0 16 50000 fid no no cbqa 1e-3 5 base WQ test all no 1 TQA