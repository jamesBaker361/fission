import os
import wget
import gzip
import csv
import string
import random
import subprocess
import requests
from experiment_helpers.gpu_details import print_details
from PIL import Image

from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#sbatch --err=slurm_chip/tsv.err --out=slurm_chip/tsv.out runpycpu_chip.sh download_tsv.py 

print_details()

root="wiki_data"
partitions=["test","val","train"]
os.makedirs(root,exist_ok=True)
for partition_name in partitions:
    
    os.makedirs(os.path.join(root,partition_name),exist_ok=True)
    limit=5
    num="05"
    if partition_name=="train":
        limit=10
        num="10"
    
    for i in range(limit):
        name=f"wit_v1.{partition_name}.all-0000{i}-of-000{num}.tsv.gz"
        url=f"https://storage.googleapis.com/gresearch/wit/{name}"
        path=os.path.join(root,partition_name,name)
        if not os.path.exists(path):
            try:
                wget.download(url, path)
                print(f"donwloaded {url} to {path}")
            except Exception as e:
                print("could not download ",url)
                print(e)
        else:
            print(f"{path} exists")
            
count_dict={d:0 for d in partitions}

session = requests.Session()

retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; vision-language-dataset/1.0)"
})




for partition_name in partitions:
    limit=5
    if partition_name=="train":
        limit=10
    found_file_path=os.path.join(root,partition_name,"processed.txt")
    found_set=set()
    count=0
    if os.path.exists(found_file_path):
        with open(found_file_path,"r") as found_file:
            lines=found_file.readlines()
            found_set=set(lines)
            count=len(found_set)
    print(partition_name," count = ",count)
    print("already found ",len(found_set))
    caption_file_path=os.path.join(root,partition_name,'captions.csv')
    with open(found_file_path,"a") as found_file:
        
        with open(caption_file_path,"a") as caption_file:
            for gz_file in os.listdir(os.path.join(root,partition_name)):
                if gz_file.endswith("tsv.gz"):
                    gz_path=os.path.join(root,partition_name,gz_file)
                    with gzip.open(gz_path, 'rt') as tar:
                        rd = csv.DictReader(tar, delimiter="\t", quotechar='"')
                        for k,row in enumerate(rd):

                            if row['language']=='en':
                                
                                text=row['caption_reference_description']
                                image_url=row["image_url"]
                                if image_url not in found_set:
                                    image_path=os.path.join(root,partition_name,f"image_{count}.jpg")
                                    caption_file.write(f"{image_path},{image_url},{text}\n")
                                    #print(["wget","-O",image_path,image_url,])
                                    #subprocess.run(["wget","-O",image_path,image_url,],capture_output=True)
                                    safe_url = quote(image_url, safe=":/?=&%")
                                    try:
                                        r = session.get(safe_url, stream=True, timeout=10)
                                        r.raise_for_status()

                                        with open(image_path, "wb") as f:
                                            for chunk in r.iter_content(chunk_size=8192):
                                                if chunk:
                                                    f.write(chunk)

                                        found_set.add(image_url)
                                        count += 1
                                        
                                    
                                        
                                        Image.open(image_url)

                                    except Exception as e:
                                        print("FAILED:", image_url)
                                        print(e)
                                    count+=1
                                    found_file.write(f"{image_url}\n")
                    print("finished ",gz_file)