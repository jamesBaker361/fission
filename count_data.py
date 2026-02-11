import os

for d in ["google_renders","shapenet_renders","laion","wiki_data","laion","laion2"]:
    print(d,len(os.listdir(d)))