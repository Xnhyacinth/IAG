x=1
dataset=NQ
m=fid
ds=no
cbqa=fid
d=NQ
nohup bash scripts/image.sh 1 ${x} 4 1.0 1 50000 ${m} ${ds} no ${cbqa} 1e-4 5 base ${d} test all use_context 100 32 no ${dataset} > logs_test/${m}_${ds}_train${dataset}_${d}_${cbqa}_100.log 2>&1 &
