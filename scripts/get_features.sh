export CUDA_VISIBLE_DEVICES=0

d='test'
dataset='NQ'
datapath='/home/huanxuan/FiD/compress_data'
nohup python -u data/get_features.py --dataset ${dataset} --d ${d} --datapath ${datapath} --num 5 > logs/data/${dataset}/${d}.log 2>&1 &
# for d in train test dev
#     do
#         nohup python data/get_features.py --dataset ${dataset} --d ${d} --datapath ${datapath} > logs/data/${dataset}/${d}.log 2>&1 &
#         sleep 20
#     done