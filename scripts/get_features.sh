
d='test'
dataset='NQ'
datapath='compress_data'
num=0
cuda=0,1,2,3
export CUDA_VISIBLE_DEVICES=${cuda}
nohup python -u data/get_features.py --dataset ${dataset} --d ${d} --datapath ${datapath} --num ${num} --cuda 0 > logs/data/${dataset}/${d}_${num}.log 2>&1 &
