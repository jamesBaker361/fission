import os
from PIL import Image


with open(os.path.join("laion","captions.csv"),"rt") as file:
    with open(os.path.join("laion","clean_captions.csv"),"wt") as caption_file:
        for r,row in enumerate(file.readlines()):
            row=row.split(",")
            if os.path.exists(row[0]):
                try:
                    Image.open(row[0]).convert("RGB")
                    caption_file.write(f"{row}\n")
                except:
                    pass