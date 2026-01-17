import datasets
import os
from huggingface_hub import snapshot_download
import zipfile
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


src_dir="shapenet"
os.makedirs(src_dir,exist_ok=True)
snapshot_download(repo_id="ShapeNet/ShapeNetCore", allow_patterns=["*.zip"], local_dir=src_dir,repo_type="dataset")

from nltk.corpus import wordnet as wn

files=[file for file in os.listdir(src_dir) if file.endswith(".zip")]
print(files)

for zf in files:
    print(f"processing {zf}...",end="")
    dest=zf.split(".")[0]
    dest=wn.synset_from_pos_and_offset("n",dest).name()
    break
    
    dest_path=os.path.join(src_dir,dest)
    os.makedirs(dest_path,exist_ok=True)
    with zipfile.ZipFile(os.path.join(src_dir,zf), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    print("done!")
