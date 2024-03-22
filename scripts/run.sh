#<num_gpus> <gpus> <accumulate_grad_batches> <val_check_interval> <batch_size> <max_steps> <name> <distill> <train_teacher> <data> <lr> <num_compress> <size>
#<dataset> <train> <select for alignment> <whether use context for hypernetwork> <num_docs for teacher> <rank> 

nohup bash scripts/image.sh 2 0,1 2 0.5 8 50000 hyperlora_ffn kl no gen 1e-3 5 base NQ train all use_context 10 32 gen > logs/hyperlora_ffn_kl_notrain_gen_lr1e-3_base_5_all_use_context_80000_10_r32_gen.log 2>&1 &

