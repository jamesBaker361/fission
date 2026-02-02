import wget
import os

url="https://ailb-web.ing.unimore.it/publicfiles/drive/Datasets/Dress%20Code/DressCode.zip"
directory="dresscode"
dest="DressCode.zip"
os.makedirs(directory,exist_ok=True)
path =os.path.join(directory,dest)
if not os.path.exists(path):
    import requests

    url = "https://ailb-web.ing.unimore.it/publicfiles/drive/Datasets/Dress%20Code/DressCode.zip"

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)