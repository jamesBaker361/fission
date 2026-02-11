import time
from experiment_helpers.init_helpers import parse_args,default_parser,repo_api_init
from experiment_helpers.gpu_details import print_details
from fission_unet2 import FissionUNet2DConditionModel,MID_BLOCK
from peft import LoraConfig
from diffusers import AutoencoderKL
from data_helpers import VirtualTryOnData,TextImageWikiData,ShapeNetImageDataPaired
from torch.utils.data import random_split
from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.saving_helpers import CONFIG_NAME
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
from experiment_helpers.image_helpers import concat_images_horizontally
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel,CLIPTokenizer
from typing import List
import os
import wandb
import json
import random
import torch.nn.functional as F
from argparse import ArgumentParser
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


VELOCITY="v_prediction"
EPSILON="epsilon"
SAMPLE="sample"

THREE_D="3_d" #for this task we always condition the left partial on a lower timestep than the right partial and also condition each partial on the camera parameters
T2I="text_to_image" # for this task we always condition the left partial on a lower timestep than the right partial and the right is conditioned on a text caption to
FASHION_SEG="fashion_seg" #left is given clothes to denoise, right is given segmentation map but target is person with clothes?
FASHION="fashion" #left is given clothes to denoise, right is person with clothes?


parser=default_parser()
parser.add_argument("--task",type=str,default=FASHION)
parser.add_argument("--pretrained",action="store_true",help="whaether to use pretrained model for partials")
parser.add_argument("--prediction_type",type=str,help=f" one of {VELOCITY}, {EPSILON} or {SAMPLE}",default=EPSILON)
parser.add_argument("--pretrained_path",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--use_lora",action="store_true",help="whether to use lora ")
parser.add_argument("--metadata_proj",action="store_true",help="whether to use metadata proj")
parser.add_argument("--metadata_proj_dim",type=int,default=4,help="project to this dim before getting embedding")
parser.add_argument("--partial_metadata_proj",action="store_true",help="whether to use metadata proj")
parser.add_argument("--partial_metadata_proj_dim",type=int,default=4,help="project to this dim before getting embedding")
parser.add_argument("--zero_fraction",default=0.0,type=float,help="default fraction of zero-ing for the right unet")
parser.add_argument("--left_residual_fraction",default=0.0,type=float,help="default fraction of using left unet residuals for the right unet")
parser.add_argument("--n_mid_blocks",type=int,default=2, help="n mid blocks")
parser.add_argument("--height",type=int,default=256)
parser.add_argument("--width",type=int,default=256)
parser.add_argument("--rank",type=int,default=4,help="lora rank")
parser.add_argument("--left_lesser",action="store_true",help="if true, the left [0] unet will always have a lower noise level")
parser.add_argument("--name",type=str,default="fission_test")
parser.add_argument("--dont_save",action="store_true",help="dont save flag for testing")
parser.add_argument("--num_inference_steps",type=int,default=10)
parser.add_argument("--val_inference_limit",type=int,default=10)
#parser.add_argument("--")

def normalize(images:torch.Tensor)->torch.Tensor: #for FID
    #[-1,1] to [0,255]
    _images=images*128
    _images=_images+128
    _images=_images.to(torch.uint8)
    
    return _images

@torch.no_grad()
def inference(args:ArgumentParser,
              fission:FissionUNet2DConditionModel,
              scheduler:DDIMScheduler,
              left_input:torch.Tensor,
              device:torch.DeviceObjType, #not sure if this is correct class
              # scale:float,  we might NOT do this if we just assume we scaled it before passing it to the inferencce function
              shared_encoder_hidden_states:torch.Tensor=None,
              encoder_hidden_states_list:List[torch.Tensor]=None,
              partial_metadata_list:List[torch.Tensor]=None,
            )->torch.Tensor:
    
    batch_size=left_input.size()[0]
    left_timesteps=torch.tensor([1 for _ in range(batch_size)],device=device).long()
    right_input=torch.randn_like(left_input,device=device)
    for t,right_timesteps in enumerate(scheduler.timesteps):
        predicted=fission.forward([left_input,right_input],[left_timesteps,right_timesteps],
                                          encoder_hidden_states_list=encoder_hidden_states_list,
                                          partial_metadata_list=partial_metadata_list,
                                          shared_encoder_hidden_states=shared_encoder_hidden_states,return_dict=False)
        right_output=predicted[1]
        right_input=scheduler.step(right_output,right_timesteps,right_input).prev_sample
        
    return right_input
        
    

def main(args):
    api,accelerator,device=repo_api_init(args)
    
    #models
    n_inputs=2
    num_metadata:int =1
    use_metadata=False
    metadata_proj:bool=args.metadata_proj
    metadata_proj_dim:int=args.metadata_proj_dim
    partial_use_metadata:bool=False
    partial_num_metadata:int =1
    partial_metadata_proj:bool=args.partial_metadata_proj
    partial_metadata_proj_dim:int=args.partial_metadata_proj_dim
    shared_mid_block_type="UNetMidBlock2D"
    tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
    if args.task==T2I:
        shared_mid_block_type="UNetMidBlock2DCrossAttn"
    if args.task==THREE_D:
        partial_use_metadata=True
        partial_num_metadata=6
        
    if args.pretrained:
        fission=FissionUNet2DConditionModel.from_pretrained_partials(args.pretrained_path,
                                                            n_inputs=n_inputs,
                                                            shared_layer_type=MID_BLOCK,
                                                            n_mid_blocks=args.n_mid_blocks,
                                                            shared_mid_block_type=shared_mid_block_type,
                                                            use_metadata=use_metadata,
                                                            partial_use_metadata=partial_use_metadata,
                                                            partial_num_metadata=partial_num_metadata,
                                                            partial_metadata_proj=partial_metadata_proj,
                                                            partial_metadata_proj_dim=partial_metadata_proj_dim)
    else:
        fission=FissionUNet2DConditionModel(n_inputs=n_inputs,
                                            shared_layer_type=MID_BLOCK,
                                            n_mid_blocks=args.n_mid_blocks,
                                            use_metadata=use_metadata,
                                            shared_mid_block_type=shared_mid_block_type,
                                            partial_use_metadata=partial_use_metadata,
                                            partial_num_metadata=partial_num_metadata,
                                            partial_metadata_proj=partial_metadata_proj,
                                            partial_metadata_proj_dim=partial_metadata_proj_dim)
        
    fission.tokenizer=tokenizer
    if args.use_lora:
        for p in fission.partial_list:
            p.requires_grad_(False)
        fission.set_adapter(
            LoraConfig(r=args.rank,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],)
        )
        
    params=[p for p in fission.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,args.lr)
        
    vae=AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="vae")
    vae.requires_grad_(False)
    scale=vae.config.scaling_factor
    text_model=CLIPTextModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="text_encoder")
    text_model.requires_grad_(False)
    
    fission.to(device)
    text_model.to(device)
    vae.to(device)
    
    image_processor=VaeImageProcessor()
    
    #data
    dim=(args.height,args.width)
    if args.task in (FASHION,FASHION_SEG):
        train_dataset=VirtualTryOnData("train",dim,args.limit)
        val_dataset,train_dataset=random_split(train_dataset,[0.1,0.9])
        test_dataset=VirtualTryOnData("test",dim,args.limit)
    elif args.task ==T2I:
        train_dataset=TextImageWikiData("train",dim,args.limit)
        val_dataset=TextImageWikiData("val",dim,args.limit)
        test_dataset=TextImageWikiData("test",dim,args.limit)
    elif args.task==THREE_D:
        train_dataset=ShapeNetImageDataPaired("shapenet_renders",dim,args.limit)
        test_dataset,val_dataset,train_dataset=random_split(train_dataset,[0.1,0.1,0.8])
        
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)
        
    scheduler=DDIMScheduler(prediction_type=args.prediction_type)
    scheduler.set_timesteps(args.num_inference_steps)
    
    
        
    train_loader,test_loader,val_loader,fission,vae,optimizer,scheduler,text_model=accelerator.prepare(train_loader,test_loader,val_loader,fission,vae,optimizer,scheduler,text_model)
    
    save_path=os.path.join(args.save_dir,args.name)
    os.makedirs(save_path,exist_ok=True)
    
    config_dict={
        "train":{
        "start_epoch":1
        }
    }
    config_path=os.path.join(save_path,CONFIG_NAME)
    
    def save(epoch:int):
        if  not args.dont_save:
            fission.save_pretrained(save_path)
            config_dict["train"]["start_epoch"]=epoch
            with open(config_path,"w") as config_file:
                json.dump(config_dict,config_file, indent=4)
        
    
    start_epoch = 1  # fresh training
    
    if os.path.exists(config_path):
        with open(config_path,"rt") as config_file:
            data=json.load(config_file)
            
        if "train" in data and "start_epoch" in data["train"]:
            start_epoch = data["train"]["start_epoch"]
    if len(os.listdir(save_path))!=0:
        fission=FissionUNet2DConditionModel.from_pretrained(save_path)

    
    ssim_metric=StructuralSimilarityIndexMeasure(data_range=(-1.0,1.0)).to(device)
    psnr_metric=PeakSignalNoiseRatio(data_range=(-1.0,1.0)).to(device)
    lpips_metric=LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    fid_metric=FrechetInceptionDistance(feature=64,normalize=False).to(device) #expects images in [0,255]
    kid_metric = KernelInceptionDistance(subset_size=50).to(device)
    
        
    test_metric_dict={
        "psnr":[],
        "ssim":[],
        "lpips":[],
        "fid":[],
        "kid":[],
        "clip_t":[],
        "clip_i":[]
    }
    
    
    @optimization_loop(accelerator,train_loader,args.epochs,args.val_interval,args.limit,val_loader,
                       test_loader,save,start_epoch)
    def batch_function(batch,training,misc_dict):
        count=misc_dict["b"]*args.batch_size
        partial_metadata_list=None
        encoder_hidden_states_list=None
        shared_encoder_hidden_states=None
        
        if args.task==THREE_D:
            left_src=batch["image_0"]
            right_src=batch["image_1"]
            
            try:
            
                left_metadata=torch.cat([torch.tensor(batch["rotation_0"]),
                                            torch.tensor(batch["location_0"])],dim=-1).to(device)
                right_metadata=torch.cat([torch.tensor(batch["rotation_1"]),
                                            torch.tensor(batch["location_1"])],dim=-1).to(device)
            except ValueError:
                print(batch["rotation_0"])
                left_metadata=torch.cat([batch["rotation_0"],
                                            batch["location_0"]],dim=-1).to(device)
                right_metadata=torch.cat([batch["rotation_1"],
                                          batch["location_1"]],dim=-1).to(device)
            
            if args.batch_size==1 and len(left_metadata.size())==1:
                left_metadata=left_metadata.unsqueeze(0)
                right_metadata=right_metadata.unsqueeze(0)
            
            partial_metadata_list=[left_metadata,right_metadata]
        elif args.task==FASHION:
            left_src=batch["cloth"]
            right_src=batch["agnostic"]
        elif args.task ==FASHION_SEG:
            left_src=batch["cloth"]
            right_src=batch["segmentation"]
        elif args.task==T2I:
            left_src=batch["image"]
            right_src=left_src.clone()
            
            shared_encoder_hidden_states=text_model(batch["input_ids"].to(device),return_dict=False)[0]
            encoder_hidden_states_list=[shared_encoder_hidden_states.clone() for _ in range(n_inputs)]
            
            #print('left_src.size()',left_src.size())
            #print('batch["input_ids"].to(device)',batch["input_ids"].size())
            #print("enocer ",shared_encoder_hidden_states.size())
            
        left_src=left_src.to(device)
        right_src=right_src.to(device)    
        batch_size=left_src.size()[0]
            
        left_src=scale*vae.encode(left_src).latent_dist.sample()
        #print(";eft src",left_src.size())
        right_src=scale*vae.encode(right_src).latent_dist.sample()   
        if misc_dict["mode"] in ["train","val"]:
            right_timesteps=[random.randint(1,scheduler.config.num_train_timesteps-1) for _ in range(batch_size)]
            if args.left_lesser:
                left_timesteps=[random.randint(0,r-1) for r in right_timesteps]
            else:
                left_timesteps=[random.randint(0,scheduler.config.num_train_timesteps-1) for _ in range(batch_size)]
                
            right_timesteps=torch.tensor(right_timesteps,device=device).long()
            left_timesteps=torch.tensor(left_timesteps,device=device).long()
             
                
            left_noise=torch.randn_like(left_src,device=device)
            right_noise=torch.randn_like(right_src,device=device)
            
            left_input=scheduler.add_noise(left_src,left_noise,left_timesteps)
            #print("elft input",left_input.size())
            right_input=scheduler.add_noise(right_src,right_noise,right_timesteps)
            
            if args.prediction_type==EPSILON:
                left_target=left_noise
                right_target=right_noise
            elif args.prediction_type==SAMPLE:
                left_target=left_src
                right_target=right_src
            elif args.prediction_type==VELOCITY:
                left_target=scheduler.get_velocity(left_src, left_noise, left_timesteps)
                right_target=scheduler.get_velocity(right_src, right_noise, right_timesteps)
            
            if random.random() < args.zero_fraction:
                zero_list=[1]
            else:
                zero_list=[]
            
            if training:
                with accelerator.accumulate(params):
                    with accelerator.autocast():
                        predicted=fission.forward([left_input,right_input],[left_timesteps,right_timesteps],
                                          encoder_hidden_states_list=encoder_hidden_states_list,
                                          partial_metadata_list=partial_metadata_list,
                                          shared_encoder_hidden_states=shared_encoder_hidden_states,return_dict=False,zero_list=zero_list)
                        loss = torch.stack([
                            F.mse_loss(t.float(), p.float())
                            for t, p in zip([left_target,right_target], predicted)
                        ]).mean()
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                with torch.no_grad():
                    predicted=fission.forward([left_input,right_input],[left_timesteps,right_timesteps],
                                            encoder_hidden_states_list=encoder_hidden_states_list,
                                            partial_metadata_list=partial_metadata_list,
                                            shared_encoder_hidden_states=shared_encoder_hidden_states,return_dict=False,)
                    loss = torch.stack([
                                F.mse_loss(t.float(), p.float())
                                for t, p in zip([left_target,right_target], predicted)
                            ]).mean()
                    
                    if count <args.val_inference_limit:
                        predicted_right_output=inference(args,fission,scheduler,left_src,device,
                                             shared_encoder_hidden_states,encoder_hidden_states_list,partial_metadata_list)
                        actual_right_pil=image_processor.postprocess( vae.decode(right_src/scale,return_dict=False)[0])
                        predicted_right_pil=image_processor.postprocess( vae.decode(predicted_right_output/scale,return_dict=False)[0])
                        for n,fake in enumerate(predicted_right_pil):
                            real=actual_right_pil[n]
                            concat=concat_images_horizontally([real,fake])
                            accelerator.log({
                                f"val_{count+n}":wandb.Image(concat)
                            })
                
                
            
        else:
            predicted_right_output=inference(args,fission,scheduler,left_src,device,
                                             shared_encoder_hidden_states,encoder_hidden_states_list,partial_metadata_list)
            loss=F.mse_loss(predicted_right_output,right_src).mean()
            count=misc_dict["b"]*args.batch_size
            actual_right_pil=image_processor.postprocess( vae.decode(right_src/scale,return_dict=False)[0])
            predicted_right_pil=image_processor.postprocess( vae.decode(predicted_right_output/scale,return_dict=False)[0])
            for n,fake in enumerate(predicted_right_pil):
                real=actual_right_pil[n]
                concat=concat_images_horizontally([real,fake])
                accelerator.log({
                    f"test_{count+n}":wandb.Image(concat)
                })
            if args.task in [FASHION,FASHION_SEG]:
                ssim_score=ssim_metric(predicted_right_output/scale,right_src/scale)
                psnr_score=psnr_metric(predicted_right_output/scale,right_src/scale)
                lpips_score=lpips_metric(predicted_right_output/scale,right_src/scale)
                
                for name,score in zip(["ssim","psnr","lpips"],[ssim_score,psnr_score,lpips_score]):
                    test_metric_dict[name].append(score.cpu().detach().numpy())
                
                for m in [fid_metric,kid_metric]:
                    m.update(normalize(right_src/scale),real=True)
                    m.update(normalize(predicted_right_output/scale),real=False)
                
        
        return loss.cpu().detach().numpy()
        
    batch_function()
    if args.task in [FASHION,FASHION_SEG]:
        test_metric_dict["fid"].append(fid_metric.compute().cpu().detach().numpy())
        test_metric_dict["kid"].append(kid_metric.compute().cpu().detach().numpy())
        
    test_metric_dict={k:v for k,v in test_metric_dict.items() if len(v)>0}
    
    print(test_metric_dict)
    accelerator.log(test_metric_dict)
    

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parse_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")