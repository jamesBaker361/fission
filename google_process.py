import os
import zipfile
import bpy
import math
import shutil
from render_images import render_obj

bpy.app.binary_path=os.path.join("blend","blender-5.0.1-linux-x64","blender")
src_dir="google"


'''for zf in files:
    print(f"processing {zf}...",end="")
    dest=zf.split(".")[0]
    dest_path=os.path.join(src_dir,dest)
    os.makedirs(dest_path,exist_ok=True)
    with zipfile.ZipFile(os.path.join(src_dir,zf), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    print("done!")'''
    
files=[file for file in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,file))]
print(files)

OUT_DIR = "renders"

os.makedirs(OUT_DIR,exist_ok=True)

NUM_VIEWS = 8
NUM_VIEWS_Z=4
NUM_VIEWS_RANDOM=20
RADIUS = 2.0
ELEVATION = math.radians(30)

IMAGE_RES = 512
ENGINE = "CYCLES"  # or "BLENDER_EEVEE"
CSV_PATH=os.path.join(OUT_DIR,"metadata.csv")
with open(CSV_PATH,"w") as csv_file:
    csv_file.write(f"file_path,category,instance,location,rotation\n")
    
count=0
for f,file in enumerate(files):
    subfolders=os.listdir(os.path.join(src_dir,file))
    print(subfolders)
    bpy.ops.wm.read_factory_settings()
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        print("removing mesh", mesh)
        bpy.data.meshes.remove(mesh)
    shutil.copy(os.path.join(src_dir,file,"materials","textures","texture.png"), os.path.join(src_dir,file,"meshes","texture.png"))
    bpy.ops.wm.obj_import(filepath=os.path.join(src_dir,file,"meshes","model.obj"),)
    count=render_obj(
        OUT_DIR,ENGINE,IMAGE_RES,NUM_VIEWS,RADIUS,NUM_VIEWS_Z,NUM_VIEWS_RANDOM,CSV_PATH,file,"first",count
    )
    exit(0)