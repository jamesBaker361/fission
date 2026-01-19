import datasets
import os
from huggingface_hub import snapshot_download
import zipfile
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import math
import bpy

from render_images import render_obj


src_dir="shapenet"
os.makedirs(src_dir,exist_ok=True)
snapshot_download(repo_id="ShapeNet/ShapeNetCore", allow_patterns=["*.zip"], local_dir=src_dir,repo_type="dataset")

from nltk.corpus import wordnet as wn



'''
files=[file for file in os.listdir(src_dir) if file.endswith(".zip")]
print(files)
for zf in files:
    print(f"processing {zf}...",end="")
    dest=zf.split(".")[0]
    dest=wn.synset_from_pos_and_offset("n",int(dest)).name().split(".")[0]
    print(dest)

    
    dest_path=os.path.join(src_dir,dest)
    os.makedirs(dest_path,exist_ok=True)
    with zipfile.ZipFile(os.path.join(src_dir,zf), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    print("done!")'''
    
OUT_DIR = "scale_renders"
os.makedirs(OUT_DIR,exist_ok=True)

NUM_VIEWS = 2
NUM_VIEWS_Z=2
NUM_VIEWS_RANDOM=10
RADIUS = 2.0
ELEVATION = math.radians(30)

IMAGE_RES = 256
ENGINE = "CYCLES"  # or "BLENDER_EEVEE"
CSV_PATH=os.path.join(OUT_DIR,"metadata.csv")
with open(CSV_PATH,"w") as csv_file:
    csv_file.write(f"file_path,category,instance,location,rotation\n")
    
count=0

files=[file for file in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,file))]
for file in files:
    for subdir in os.listdir(os.path.join(src_dir,file)):
        subdir_path=os.path.join(src_dir,file,subdir)
        if os.path.isdir(subdir_path):
            print(subdir)
            for subsubdir in os.listdir(subdir_path):
                print(subsubdir)
                sub_subdir_path=os.path.join(subdir_path,subsubdir)
                bpy.ops.wm.read_factory_settings(use_empty=True)
                model_path=os.path.join(sub_subdir_path,"models","model_normalized.obj")
                bpy.ops.wm.obj_import(filepath=model_path)
                print("subsusdir path" , model_path)
                count=render_obj(
                    OUT_DIR,ENGINE,IMAGE_RES,NUM_VIEWS,RADIUS,NUM_VIEWS_Z, 
                    NUM_VIEWS_RANDOM,
                    CSV_PATH,file,subsubdir,count,
                )
                print("images ",count)
                
                exit(0)