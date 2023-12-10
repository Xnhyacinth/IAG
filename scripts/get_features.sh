

d='test'
dataset='WQ'
datapath='/home/huanxuan/FiD/compress_data'
num=1
cuda=7
export CUDA_VISIBLE_DEVICES=${cuda}
nohup python -u data/get_features.py --dataset ${dataset} --d ${d} --datapath ${datapath} --num ${num} --cuda ${cuda} > logs/data/${dataset}/${d}_${num}.log 2>&1 &
# for d in train test dev
#     do
#         nohup python data/get_features.py --dataset ${dataset} --d ${d} --datapath ${datapath} > logs/data/${dataset}/${d}.log 2>&1 &
#         sleep 20
#     done