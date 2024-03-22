FROM python:3.8.18-bullseye
RUN sed -i 's/deb.debian.org/ftp.cn.debian.org/g' /etc/apt/sources.list
RUN apt update
RUN apt install -y libhdf5-dev
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install transformers==4.23.1
RUN pip install accelerate
RUN pip install seqeval
RUN pip install datasets
RUN pip install evaluate
RUN pip install nltk
RUN pip install datasets
RUN pip install argparse
RUN pip install tensorboard
RUN pip install rouge_score
RUN pip install deepspeed
RUN pip install peft
RUN pip install tqdm
RUN pip install bitsandbytes
RUN pip install apex
RUN pip install torchtyping
RUN pip install pytorch_lightning==1.6.3