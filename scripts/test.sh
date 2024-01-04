# nohup bash scripts/image.sh 2 2,7 4 1000 4 50000 hylora no no cbqa 1e-3 1 TQA > logs_tqa/hylora_kl_notrain_gold_lr1e-3.log 2>&1 &
# bash scripts/image.sh 1 0 4 1.0 2 50000 hylora kl no gen 1e-3 5 base WQ test all use_context 10 yes TQA
# bash scripts/image.sh 1 1 4 1.0 16 50000 hylora kl no cbqa 1e-3 5 base TQA test all use_context 10 no WQ

x=4
dataset=NQ
m=fid
ds=no
cbqa=fid
# for d in NQ TQA
#     do
#         nohup bash scripts/image.sh 1 ${x} 4 1.0 16 50000 ${m} ${ds} no cbqa 1e-3 5 base ${d} test all use_context 1 16 no ${dataset} > logs_test/${m}_${ds}_train${dataset}_${d}.log 2>&1 &
#         sleep 10
#         echo ${x}
#         let x+=1
#     done
# d=WQ
d=NQ
nohup bash scripts/image.sh 1 ${x} 4 1.0 1 50000 ${m} ${ds} no ${cbqa} 1e-3 5 base ${d} test all use_context 100 16 yes ${dataset} > logs_test/${m}_${ds}_train${dataset}_${d}.log 2>&1 &
d=TQA
x=4
sleep 10
# nohup bash scripts/image.sh 1 ${x} 4 1.0 64 50000 ${m} ${ds} no cbqa 1e-3 5 base ${d} test all use_context 1 32 no ${dataset} > logs_test/${m}_${ds}_train${dataset}_${d}.log 2>&1 &
d=WQ
x=5
sleep 10
# nohup bash scripts/image.sh 1 ${x} 4 1.0 32 50000 ${m} ${ds} no cbqa 1e-3 5 base ${d} test all use_context 1 32 no ${dataset} > logs_test/${m}_${ds}_train${dataset}_${d}.log 2>&1 &
# bash scripts/image.sh 1 1 4 1.0 16 50000 hylora kl no cbqa 1e-3 5 base TQA test all no 1 no NQ