from datasets import load_dataset

# import datasets 
# config = datasets.DownloadConfig(resume_download=True, max_retries=100) 
# # dataset = datasets.load_dataset( "codeparrot/self-instruct-starcoder", cache_dir="./hf_cache", download_config=config)
# data = load_dataset("Xnhyacinth/Image", 'TQA', download_config=config)
# print(data)
# data.save_to_disk('dataset/Image/TQA')

from huggingface_hub import snapshot_download
import transformers
# snapshot_download(repo_id='google/flan-t5-base',
#                   repo_type='model',
#                   local_dir='./models/flan-t5-base',
#                   resume_download=True)
a = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            't5-large', resume_download=True
        )
print(a.config.d_model)