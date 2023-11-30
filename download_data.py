from datasets import load_dataset

import datasets 
config = datasets.DownloadConfig(resume_download=True, max_retries=100) 
# dataset = datasets.load_dataset( "codeparrot/self-instruct-starcoder", cache_dir="./hf_cache", download_config=config)
data = load_dataset("Xnhyacinth/Image", 'TQA', download_config=config)
print(data)
data.save_to_disk('dataset/Image/TQA')
