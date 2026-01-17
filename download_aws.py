import requests
from tqdm import tqdm
#https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar


for output_file in ['abo-listings.tar','abo-3dmodels.tar']:
    
    url = f"https://amazon-berkeley-objects.s3.amazonaws.com/archives/{output_file}"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(output_file, "wb") as f, tqdm(
        desc=output_file,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Downloading {output_file} finished!")
    break
