import os
import zipfile

src_dir="google"
files=[file for file in os.listdir(src_dir) if file.endswith(".zip")]

for zf in files:
    print(f"processing {zf}...",end="")
    dest=zf.split(".")[0]
    dest_path=os.path.join(src_dir,dest)
    os.makedirs(dest_path,exist_ok=True)
    with zipfile.ZipFile(os.path.join(src_dir,zf), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    print("done!")
    
