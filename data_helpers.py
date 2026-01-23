from torch.utils.data import Dataset
import os
import csv
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import itertools

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
    
class ImageDataPaired(Dataset):
    def __init__(self,render_dir:str,dim:int):
        super().__init__()
        self.render_dir=render_dir
        self.dim=dim
        metadata_path=os.path.join(render_dir,"metadata.csv")
        self.metadata=[]
        self.image_processor=VaeImageProcessor()
        self.metadata_dict={}
        with open(metadata_path) as file:
            metadata=csv.DictReader(file)
            self.metadata=[r for r in metadata]
            print(self.metadata)
            for i,row in enumerate( self.metadata):
                key=row["category"]+"_"+row["instance"]
                if key not in self.metadata_dict:
                    self.metadata_dict[key]=[]
                self.metadata_dict[key].append(i)
        self.index_pair_list=[]
        print(self.metadata_dict)
        for key,value in self.metadata_dict.items():
            pairs= list(itertools.combinations(value, 2))
            print("pairs",pairs)
            self.index_pair_list+=pairs
    
    def __len__(self):
        return len(self.index_pair_list)
    
    def __getitem__(self, index):
        index_pair=self.index_pair_list[index]
        row_0=self.metadata[index_pair[0]]
        row_1=self.metadata[index_pair[1]]
        
        output= {
            key:row_0[key] for key in ["category","instance"]
        }
        
        location_0=[float(l) for l in row_0["location"].split("_")]
        rotation_0=[float(r) for r in row_0["rotation"].split("_")]
        
        location_1=[float(l) for l in row_1["location"].split("_")]
        rotation_1=[float(r) for r in row_1["rotation"].split("_")]
        
        image_0= Image.open(row_0["file_path"]).resize((self.dim,self.dim))
        
        image_pt_0=self.image_processor.preprocess(image_0)[0]
        
        output["image_0"]=image_pt_0
        
        image_1= Image.open(row_1["file_path"]).resize((self.dim,self.dim))
        
        image_pt_1=self.image_processor.preprocess(image_1)[0]
        
        output["image_1"]=image_pt_1
        
        location=[l0-l1 for l0,l1 in zip(location_0,location_1)]
        rotation=[r0-r1 for r0,r1 in zip(rotation_0,rotation_1)]
        
        output["location"]=location
        output["rotation"]=rotation
        
        return output
        
        
            
        
    
if __name__=='__main__':
    data=ImageDataPaired("scale_renders",dim=128)
    print("len",len(data))
    for batch in data:
        break
    
    
    print(batch)