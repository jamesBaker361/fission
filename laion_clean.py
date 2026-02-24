import os
from PIL import Image


with open(os.path.join("laion","captions.csv"),"rt") as file:
    with open(os.path.join("laion","clean_captions.csv"),"w") as caption_file:
        with open(os.path.join("laion","rgba_clean_captions.csv"),"w") as rgba_caption_file:
            for r,row in enumerate(file.readlines()):
                first_index=row.find(',')
                second_index=row.find(',',first_index+1)
                path=row[:first_index]
                url=row[first_index+1:second_index]
                caption=row[second_index+1:].replace(',', ' ')
                if os.path.exists(path):
                    try:
                        Image.open(path).convert("RGB")
                        caption_file.write(f"{path},{url},{caption}\n")
                    except:
                        try:
                            Image.open(path).convert("RGBA")
                            rgba_caption_file.write(f"{path},{url},{caption}\n")
                        except Exception as err:
                            print(err)