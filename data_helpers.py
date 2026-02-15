from torch.utils.data import Dataset
import os
import csv
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer
from PIL import Image
import itertools
from typing import Tuple
import kagglehub
import torchvision.transforms as T
import csv
import torch
import json



class ShapeNetImageData(Dataset):
    def __init__(self,render_dir:str,dim:Tuple[int],limit:int=-1):
        super().__init__()
        self.render_dir=render_dir
        self.dim=dim
        metadata_path=os.path.join(render_dir,"metadata.csv")
        self.metadata=[]
        self.image_processor=VaeImageProcessor()
        #self.tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        with open(metadata_path) as file:
            metadata=csv.DictReader(file)
            for r,row in enumerate(metadata):
                if r==limit:
                    break
                self.metadata.append(row)
                
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row=self.metadata[index]
        output= {
            key:row[key] for key in ["category","instance"]
        }
        output["location"]=[float(l) for l in row["location"].split("_")]
        output["rotation"]=[float(r) for r in row["rotation"].split("_")]
        
        image= Image.open(row["file_path"]).resize((self.dim))
        
        image_pt=self.image_processor.preprocess(image)[0]
        
        output["image"]=image_pt
        
        return output
    
class ShapeNetImageDataPaired(Dataset):
    def __init__(self,render_dir:str,dim:Tuple[int],limit:int=-1,partition:str="training",test_frac=0.1):
        super().__init__()
        self.render_dir=render_dir
        self.dim=dim
        self.partition=partition
        metadata_path=os.path.join(render_dir,"metadata.csv")
        self.metadata=[]
        self.image_processor=VaeImageProcessor()
        self.metadata_dict={}
        #self.tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        with open(metadata_path) as file:
            metadata=csv.DictReader(file)
            self.metadata=[r for r in metadata]
            #print(self.metadata)
            for i,row in enumerate( self.metadata):
                key=row["category"]+"_"+row["instance"]
                if key not in self.metadata_dict:
                    self.metadata_dict[key]=[]
                self.metadata_dict[key].append(i)
        self.index_pair_list=[]
        keys=[k for k in self.metadata_dict.keys()]
        n_test=int(test_frac*len(keys))
        
        
        if partition =="training" and test_frac>0:
            keys=keys[n_test:]
        else:
            keys=keys[:n_test]
            
        self.keys=keys
        
        print(f"total {len(keys)} keys ")
        
        for key,value in self.metadata_dict.items():
            if key in keys:
                pairs= list(itertools.combinations(value, 2))
                #print("pairs",pairs)
                self.index_pair_list+=pairs
                
        self.limit=limit
        self.index_pair_list=self.index_pair_list[:limit]
    
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
        
        image_0= Image.open(row_0["file_path"]).resize((self.dim)).convert("RGB")
        
        image_pt_0=self.image_processor.preprocess(image_0)[0]
        
        output["image_0"]=image_pt_0
        
        image_1= Image.open(row_1["file_path"]).resize((self.dim)).convert("RGB")
        
        image_pt_1=self.image_processor.preprocess(image_1)[0]
        
        output["image_1"]=image_pt_1
        
        location=[l0-l1 for l0,l1 in zip(location_0,location_1)]
        rotation=[r0-r1 for r0,r1 in zip(rotation_0,rotation_1)]
        
        output["location"]=torch.tensor(location)
        output["rotation"]=torch.tensor(rotation)
        output["location_0"]=torch.tensor(location_0)
        output["location_1"]=torch.tensor(location_1)
        output["rotation_0"]=torch.tensor(rotation_0)
        output["rotation_1"]=torch.tensor(rotation_1)
        
        return output
        
        
            
class VirtualTryOnData(Dataset):
    def __init__(self,partition:str,dim:Tuple[int],limit:int=-1):
        super().__init__()        
        self.partition=partition #one of [train test]
        self.dim=dim #(h,w)
        self.path=kagglehub.dataset_download("marquis03/high-resolution-viton-zalando-dataset")
        self.image_processor=VaeImageProcessor()
        #self.tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        txt_file=os.path.join(self.path,f"{partition}_pairs.txt")
        self.file_paths=[]
        with open(txt_file,"rt") as file:
            for n,line in enumerate(file.readlines()):
                line_list=line.split()
                self.file_paths.append(line_list[0])
                self.file_paths.append(line_list[1])
                if n==limit:
                    break
                
        self.transforms=T.Compose([
            T.ColorJitter(),
            T.RandomInvert()
        ])
                
    def __len__(self):
        return len(self.file_paths)
    
    
    def __getitem__(self, index):
        file=self.file_paths[index]
        
        output= {
            key:self.image_processor.preprocess(Image.open(os.path.join(self.path,self.partition,key,file)).resize(self.dim) )[0] for key in ["image","cloth"]
        }
        
        output["segmentation"]=self.transforms(
            self.image_processor.preprocess(Image.open(os.path.join(self.path,self.partition,"image-parse-v3",file.replace("jpg","png"))).convert('RGB').resize(self.dim) )[0]
        )
        output["agnostic"]=self.transforms(
            self.image_processor.preprocess(Image.open(os.path.join(self.path,self.partition,"agnostic-v3.2",file)).convert('RGB').resize(self.dim) )[0]
        )
        #output["segmentation"]=self.image_processor.preprocess(Image.open(os.path.join(self.path,self.partition,"image-parse-v3",file.replace("jpg","png"))).convert('RGB').resize(self.dim) )[0]
        
        return output
    

        
        
class TextImageWikiData(Dataset):
    def __init__(self,partition:str,dim:Tuple[int],limit:int=-1):
        super().__init__()
        self.partition=partition #one of [train test val]
        self.dim=dim #(h,w)
        self.image_processor=VaeImageProcessor()
        self.tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        self.path_list=[]
        self.caption_list=[]
        
        with open(os.path.join("wiki_data",partition,"captions.csv"),"rt") as file:
            for r,row in enumerate(file.readlines()):
                row=row.split(",")
                self.path_list.append(row[0])
                self.caption_list.append(row[-1])
                if r==limit:
                    break
                
    def __len__(self):
        return len(self.path_list)
    
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess(Image.open(self.path_list[index]).convert("RGB").resize(self.dim))[0],
            "text":self.caption_list[index],
            "input_ids":self.tokenizer(self.caption_list[index],padding="max_length",max_length=self.tokenizer.model_max_length, return_tensors="pt",).input_ids,
            
            
        }
        
class LaionDataset(Dataset):
    def __init__(self,dim:Tuple[int],limit:int=-1):
        super().__init__()
        self.dim=dim #(h,w)
        self.image_processor=VaeImageProcessor()
        self.tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        self.path_list=[]
        self.caption_list=[]
        self.limit=limit
        with open(os.path.join("laion","captions.csv"),"rt") as file:
            count=0
            for r,row in enumerate(file.readlines()):
                row=row.split(",")
                if os.path.exists(row[0]):
                    if count==limit:
                        break
                    count+=1
                    self.path_list.append(row[0])
                    self.caption_list.append(row[-1])
                    
                    
        
    def __len__(self):
        return len(self.path_list)
    
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess(Image.open(self.path_list[index]).convert("RGB").resize(self.dim))[0],
            "text":self.caption_list[index],
            "input_ids":self.tokenizer(self.caption_list[index],padding="max_length",max_length=self.tokenizer.model_max_length, return_tensors="pt",).input_ids,
            
            
        }
        
class PersonaDataset(Dataset):
    def __init__(self,dim:Tuple[int],limit:int=-1):
        super().__init__()
        self.image_processor=VaeImageProcessor()
        self.tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        self.dim=dim
        with open(os.path.join("pcs_dataset","info.json"),"r") as file:
            info=json.load(file)
            
        subject_with_cls=info["subjects"]["subject_with_cls"] #dict
        live_subjects=info["subjects"]["live_subjects"]
        prompt_object=info["subjects"]["prompt_object"]
        prompt_live=info["subjects"]["prompt_live"]
        
        id_with_gender=info["face"]["id_with_gender"] #dict
        prompt_accesory=info["face"]["prompt_accessory"]
        prompt_context=info["face"]["prompt_context"]
        prompt_action=info["face"]["prompt_action"]
        prompt_style=info["face"]["prompt_style"]
        
        face_prompt_list=prompt_accesory+prompt_context+prompt_action+prompt_style
        
        self.caption_list=[]
        self.image_path_list=[]
        
        for key,name in subject_with_cls.items():
            img_dir_path=os.path.join("pcs_dataset","subjects",key)
            for x in range(10):
                img_path=os.path.join(img_dir_path,f"0{x}.jpg")
                if os.path.exists(img_path):
                    for prompt in prompt_object:
                        self.caption_list.append(prompt.format(name, " "))
                        self.image_path_list.append(img_path)
                
        for name in live_subjects:
            img_dir_path=os.path.join("pcs_dataset","subjects",name)
            for x in range(10):
                img_path=os.path.join(img_dir_path,f"0{x}.jpg")
                if os.path.exists(img_path):
                    for prompt in prompt_live:
                        self.caption_list.append(prompt.format(name, " "))
                        self.image_path_list.append(img_path)
                        
        for key,name in id_with_gender.items():
            img_path=os.path.join("pcs_dataset","face",key,"face.jpg")
            for prompt in face_prompt_list:
                self.caption_list.append(prompt.format(name, " "))
                self.image_path_list.append(img_path)
                
        self.caption_list=self.caption_list[:limit]
        self.image_path_list=self.image_path_list[:limit]
        
    def __len__(self):
        return len(self.caption_list)
    
    
    def __getitem__(self, index):
        text=self.caption_list[index]
        return {
             "image":self.image_processor.preprocess(Image.open(self.image_path_list[index]).convert("RGB").resize(self.dim))[0],
            "text":text,
            "input_ids":self.tokenizer(text,padding="max_length",max_length=self.tokenizer.model_max_length, return_tensors="pt",).input_ids,
            
        }
            
                
        
        

    
if __name__=='__main__':
    for dataset_class in [ShapeNetImageDataPaired]:
        data=dataset_class("shapenet_renders",dim=(64,64),limit=10) #shapenet renders is the big directory
        print(dataset_class,"len",len(data))
        for batch in data:
            print(batch["image_0"].max(),batch["image_0"].min())
            break
    for dataset_class in [TextImageWikiData,VirtualTryOnData]:
        data=dataset_class("test",dim=(64,64),limit=10)
        print(dataset_class,"len",len(data))
        for batch in data:
            print(batch["image"].max(),batch["image"].min())
            break
        
    for dataset_class in [PersonaDataset,LaionDataset]:
        data=dataset_class(dim=(64,64),limit=10)
        print(dataset_class,"len",len(data))
        for batch in data:
            print(batch["image"].max(),batch["image"].min())
            break
        
    
    
    print(batch)