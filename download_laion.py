import datasets
import os
import requests
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

root="laion"
os.makedirs(root, exist_ok=True)
hf_data=datasets.load_dataset("laion/relaion-pop",split="train")

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

found_file_path=os.path.join(root,"processed.txt")
found_set=set()
count=0
if os.path.exists(found_file_path):
    with open(found_file_path,"r") as found_file:
        lines=found_file.readlines()
        found_set=set(lines)
        count=len(found_set)
print(" count = ",count)
print("already found ",len(found_set))
caption_file_path=os.path.join(root,'captions.csv')
with open(found_file_path,"a") as found_file:
    
    with open(caption_file_path,"a") as caption_file:
        for r,row in enumerate(hf_data):
            image_url=row["url"]
            text=row["llava_caption"]
            if image_url not in found_set:
                image_path=os.path.join(root,f"image_{count}.jpg")
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
            