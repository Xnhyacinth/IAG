
# nohup bash finetune-lora.sh 2 8,9 WQ-image 1 1024 > logs/WQ-image.log 2>&1 &
# nohup bash finetune-lora.sh 2 2,3 WQ 1 1024 > logs/WQ.log 2>&1 &
# nohup bash finetune-lora.sh 2 8,9 NQ-image 1 1024 > logs/NQ-image.log 2>&1 &
nohup bash finetune-lora.sh 4 6,7,8,9 NQ-image 1 512 > logs/NQ-image.log 2>&1 &