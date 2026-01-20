from torch.utils.data import Dataset
import os
import csv
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

class ImageData(Dataset):
    def __init__(self,render_dir:str,dim:int):
        super().__init__()
        self.render_dir=render_dir
        self.dim=dim
        metadata_path=os.path.join(render_dir,"metadata.csv")
        self.metadata=[]
        self.image_processor=VaeImageProcessor()
        with open(metadata_path) as file:
            metadata=csv.DictReader(file)
            self.metadata=[r for r in metadata]
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row=self.metadata[index]
        output= {
            key:row[key] for key in ["category","instance"]
        }
        output["location"]=[float(l) for l in row["location"].split("_")]
        output["rotation"]=[float(r) for r in row["rotation"].split("_")]
        
        image= Image.open(row["file_path"]).resize((self.dim,self.dim))
        
        image_pt=self.image_processor.preprocess(image)[0]
        
        output["image"]=image_pt
        
        return output
    
if __name__=='__main__':
    data=ImageData("scale_renders")
    print("len",len(data))
    for batch in data:
        break
    
    
    print(batch)