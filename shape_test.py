import datasets
import os
from huggingface_hub import snapshot_download
os.makedirs("shapenet",exist_ok=True)
snapshot_download(repo_id="ShapeNet/ShapeNetCore", allow_patterns=["*.zip"], local_dir="shapenet",repo_type="dataset")