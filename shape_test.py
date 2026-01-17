import datasets
import os
from huggingface_hub import snapshot_download
os.makedirs("shapenet",exist_ok=True)
snapshot_download(repo_id="gpt2", allow_patterns=["*.zip"], local_dir="shapenet")