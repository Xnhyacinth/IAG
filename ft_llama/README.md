# 用Lora和deepspeed微调LLaMA2-Chat

在两块P100（16G）上微调Llama-2-7b-chat模型。

数据源采用了alpaca格式，由train和validation两个数据源组成。

## 1、显卡要求

16G显存及以上（P100或T4及以上），一块或多块。

## 2、Clone源码

```bash
git clone https://github.com/git-cloner/llama2-lora-fine-tuning
cd llama2-lora-fine-tuning
```

## 3、安装依赖环境

```bash
# 创建虚拟环境
conda create -n llama2 python=3.9 -y
conda activate llama2
# 下载github.com上的依赖资源（需要反复试才能成功，所以单独安装）
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1
pip install git+https://github.com/PanQiWei/AutoGPTQ.git -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip install git+https://github.com/huggingface/peft -i https://pypi.mirrors.ustc.edu.cn/simple
pip install git+https://github.com/huggingface/transformers -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装其他依赖包
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 验证bitsandbytes
python -m bitsandbytes
```

## 4、下载原始模型

```bash
python model_download.py --repo_id daryl149/llama-2-7b-chat-hf
```

## 5、扩充中文词表

```bash
# 使用了https://github.com/ymcui/Chinese-LLaMA-Alpaca.git的方法扩充中文词表
# 扩充完的词表在merged_tokenizes_sp（全精度）和merged_tokenizer_hf（半精度）
# 在微调时，将使用--tokenizer_name ./merged_tokenizer_hf参数
python merge_tokenizers.py \
  --llama_tokenizer_dir ./models/daryl149/llama-2-7b-chat-hf \
  --chinese_sp_model_file ./chinese_sp.model
```

## 6、微调参数说明

有以下几个参数可以调整：

| 参数                        | 说明                       | 取值                                                         |
| --------------------------- | -------------------------- | ------------------------------------------------------------ |
| load_in_bits                | 模型精度                   | 4和8，如果显存不溢出，尽量选高精度8                          |
| block_size                  | token最大长度              | 首选2048，内存溢出，可选1024、512等                          |
| per_device_train_batch_size | 训练时每块卡每次装入批量数 | 只要内存不溢出，尽量往大选                                   |
| per_device_eval_batch_size  | 评估时每块卡每次装入批量数 | 只要内存不溢出，尽量往大选                                   |
| include                     | 使用的显卡序列             | 如两块：localhost:1,2（特别注意的是，序列与nvidia-smi看到的不一定一样） |
| num_train_epochs            | 训练轮数                   | 至少3轮                                                      |

## 7、微调

```bash
chmod +x finetune-lora.sh
# 微调
./finetune-lora.sh
# 微调（后台运行）
pkill -9 -f finetune-lora
nohup ./finetune-lora.sh > train.log  2>&1 &
tail -f train.log
```

## 8、测试

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --base_model './models/daryl149/llama-2-7b-chat-hf' \
    --lora_weights 'output/checkpoint-2000' \
    --load_8bit #不加这个参数是用的4bit
```

