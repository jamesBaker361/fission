import os
from data_helpers import *

for d in ["google_renders","shapenet_renders","laion","wiki_data","laion","laion2"]:
    print(d,len(os.listdir(d)))
    
for dataclass in [LaionDataset, SBUDataset]:
    print(dataclass,len(dataclass((32,32))))