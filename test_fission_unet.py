from fission_unet2 import FissionUNet2DConditionModel,UNET,MID_BLOCK
import unittest
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig
import torch.nn.functional as F
import os

class TestFission(unittest.TestCase):
    #sbatch --err=slurm_chip/fission.err --out=slurm_chip/fission.out runpycpu_chip.sh test_fission_unet.py 
    def test_init(self):

        for shared_layer_type in [UNET, MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
                FissionUNet2DConditionModel(3,shared_layer_type,2)
        
    def test_from_pretrained_partials(self):

        n_inputs=2
        for shared_layer_type in [UNET, MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
                FissionUNet2DConditionModel.from_pretrained_partials("SimianLuo/LCM_Dreamshaper_v7",n_inputs,shared_layer_type,2)
                
    def test_forward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        for batch_size in [2,3]:
            for n_inputs in [2,3]:
                with self.subTest(batch_size=batch_size):
                    with self.subTest(n_inputs=n_inputs):
        
                        sample_list=[torch.zeros((batch_size,4,32,32)) for _ in range(n_inputs)]
                        timestep_list=[torch.randn((batch_size)) for _ in range(n_inputs)]
                        for shared_layer_type in [ MID_BLOCK]:
                            with self.subTest(shared_layer_type=shared_layer_type):
                        
                                fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
                                
                                device,dtype=fission.get_device_dtype()
                                sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                                timestep_list=[torch.randn((batch_size)).to(device,dtype)  for _ in range(n_inputs)]
                                fission.forward(sample_list,timestep_list)
                
    def test_forward_metadata(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=3
        batch_size=2
        
        for shared_layer_type in [ MID_BLOCK]:
            for num_metadata in [1,3]:
                with self.subTest(shared_layer_type=shared_layer_type):
                    with self.subTest(num_metadata=num_metadata):
                        
        
                        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2,num_metadata=3).to(device)
                        device,dtype=fission.get_device_dtype()
                        metadata=torch.ones((batch_size,num_metadata)).to(device,dtype) 
                        sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                        timestep_list=[torch.randn((batch_size)).to(device,dtype)  for _ in range(n_inputs)]
                        fission.forward(sample_list,timestep_list,metadata=metadata)
    
    def test_forward_metadata_proj_dim(self):
        # sbatch --err=slurm_chip/test_forward_metadata_proj_dim.err --out=slurm_chip/test_forward_metadata_proj_dim.out runpycpu_chip.sh test_fission_unet.py TestFission.test_forward_metadata_proj_dim
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=3
        batch_size=2
        
        for shared_layer_type in [ MID_BLOCK]:
            for num_metadata in [1,3]:
                for metadata_proj_dim in [1,2,3,4]:
                    with self.subTest(shared_layer_type=shared_layer_type):
                        with self.subTest(num_metadata=num_metadata):
                            with self.subTest(metadata_proj_dim=metadata_proj_dim):
                            
            
                                fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2,metadata_proj=True,num_metadata=3,metadata_proj_dim=metadata_proj_dim).to(device)
                                device,dtype=fission.get_device_dtype()
                                metadata=torch.ones((batch_size,num_metadata)).to(device,dtype) 
                                sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                                timestep_list=[torch.randn((batch_size)).to(device,dtype)  for _ in range(n_inputs)]
                                fission.forward(sample_list,timestep_list,metadata=metadata)
                        
    def test_forward_metadata_partial(self):
        
        # sbatch --err=slurm_chip/test_forward_metadata_partial.err --out=slurm_chip/test_forward_metadata_partial.out runpycpu_chip.sh test_fission_unet.py TestFission.test_forward_metadata_partial
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=3
        batch_size=2
        
        for shared_layer_type in [ MID_BLOCK]:
            for num_metadata in [1,3]:
                with self.subTest(shared_layer_type=shared_layer_type):
                    with self.subTest(num_metadata=num_metadata):
                        
        
                        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2,partial_metadata_proj=True,partial_num_metadata=3).to(device)
                        device,dtype=fission.get_device_dtype()
                        metadata=[torch.ones((batch_size,num_metadata)).to(device,dtype) for _ in range(n_inputs)]
                        sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                        timestep_list=[torch.randn((batch_size)).to(device,dtype)  for _ in range(n_inputs)]
                        fission.forward(sample_list,timestep_list,partial_metadata_list=metadata)
                        
        
                
    def test_forward_str(self):
        # sbatch --err=slurm_chip/test_forward_str.err --out=slurm_chip/test_forward_str.out runpycpu_chip.sh test_fission_unet.py TestFission.test_forward_str
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch_size in [2,3]:
            for n_inputs in [2,3]:
                with self.subTest(batch_size=batch_size):
                    with self.subTest(n_inputs=n_inputs):
        
                        str_list=[["hsdak" for _ in range(batch_size)] for __ in range(n_inputs)]
                        
                        
                        tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
                        text_model=CLIPTextModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="text_encoder")
                        
                        for shared_layer_type in [MID_BLOCK]:
                            with self.subTest(shared_layer_type=shared_layer_type):
                        
                                fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2,tokenizer=tokenizer,text_model=text_model).to(device)
                                device,dtype=fission.get_device_dtype()
                                sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                                timestep_list=[torch.randn((batch_size)).to(device,dtype)  for _ in range(n_inputs)]
                                fission.forward(sample_list,timestep_list,str_list=str_list,)
                
    def test_forward_ids(self):
        # sbatch --err=slurm_chip/test_forward_ids.err --out=slurm_chip/test_forward_ids.out runpycpu_chip.sh test_fission_unet.py  TestFission.test_forward_ids
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch_size in [2,3]:
            for n_inputs in [2,3]:
                with self.subTest(batch_size=batch_size):
                    with self.subTest(n_inputs=n_inputs):
        
                        tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
                        text_model=CLIPTextModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="text_encoder")
                        
                        str_list=[["hsdak" for _ in range(batch_size)] for __ in range(n_inputs)]
                        token_id_list=[tokenizer(s,padding="max_length",
                                    max_length=tokenizer.model_max_length, return_tensors="pt",).input_ids for s in str_list]
                        print('batch_size,n_inputs,token_id_list',batch_size,n_inputs,token_id_list)
                        
        
        
                        
                        for shared_layer_type in [MID_BLOCK]:
                            with self.subTest(shared_layer_type=shared_layer_type):
                        
                                fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2,text_model=text_model).to(device)
                                device,dtype=fission.get_device_dtype()
                                sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                                timestep_list=[torch.randn((batch_size)).to(device,dtype)  for _ in range(n_inputs)]
                                fission.forward(sample_list,timestep_list,token_id_list=token_id_list)
                
    def test_load_lora(self):
        
        for shared_layer_type in [MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
                path="lora_test"
                fission=FissionUNet2DConditionModel(3,"UNetMidBlock2DCrossAttn",shared_layer_type,2)
                device,dtype=fission.get_device_dtype()
                config=LoraConfig(r=4,
                    lora_alpha=4,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],)
                fission.set_adapter(config)
                fission.save_lora_adapter(path)
                fission.unload_lora()
                fission.set_adapter(config)
                fission.load_lora_adapter(path)
                
    def test_zero(self):
        #sbatch --err=slurm_chip/test_zero.err --out=slurm_chip/test_zero.out runpycpu_chip.sh test_fission_unet.py TestFission.test_zero
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        
        for batch_size in [2,3]:
            for n_inputs in [2,3]:
                for zero_list in [[0],[1],[1,0]]:
                    with self.subTest(batch_size=batch_size):
                        with self.subTest(n_inputs=n_inputs):
                            with self.subTest(zero_list=zero_list):
        
                                
                                for shared_layer_type in [ MID_BLOCK]:
                                    with self.subTest(shared_layer_type=shared_layer_type):
                                
                                        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
                                        device,dtype=fission.get_device_dtype()
                                        sample_list=[torch.zeros((batch_size,4,32,32)).to(device,dtype) for _ in range(n_inputs)]
                                        timestep_list=[torch.randn((batch_size)).to(device,dtype) for _ in range(n_inputs)]
                                        device,dtype=fission.get_device_dtype()
                                        fission.forward(sample_list,timestep_list,zero_list=zero_list)
                                        
    def test_train(self):
        # sbatch --err=slurm_chip/test_train.err --out=slurm_chip/test_train.out runpycpu_chip.sh test_fission_unet.py TestFission.test_train
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        dim=4
        shared_layer_type =MID_BLOCK
        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
        device,dtype=fission.get_device_dtype()
        timestep_list=[torch.randn((batch_size)).to(device,dtype) for _ in range(n_inputs)]
        optimizer=torch.optim.Adam([p for p in fission.parameters()])
        inputs=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        targets=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        
        optimizer.zero_grad()
        generated_outputs=fission.forward(inputs,timestep_list)
        print("gen put",generated_outputs)
        loss = torch.stack([
            F.mse_loss(i.float(), o.float())
            for i, o in zip(targets, generated_outputs)
        ]).mean()
        
        loss.backward()
        optimizer.step()
        
    def test_train_lora(self):
        
        # sbatch --err=slurm_chip/test_train_lora.err --out=slurm_chip/test_train_lora.out runpycpu_chip.sh test_fission_unet.py TestFission.test_train_lora
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        dim=4
        shared_layer_type =MID_BLOCK
        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
        device,dtype=fission.get_device_dtype()
        
        fission.requires_grad_(False)
        '''for partial in fission.partial_list:
            self.assertFalse(partial.requires_grad)'''
            
        
        config=LoraConfig(r=4,
                    lora_alpha=4,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],)
        fission.set_adapter(config)
        
        device,dtype=fission.get_device_dtype()
        
        timestep_list=[torch.randn((batch_size)).to(device,dtype) for _ in range(n_inputs)]
        optimizer=torch.optim.Adam([p for p in fission.parameters()])
        inputs=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        targets=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        
        optimizer.zero_grad()
        generated_outputs=fission.forward(inputs,timestep_list)
        
        loss = torch.stack([
            F.mse_loss(i.float(), o.float())
            for i, o in zip(targets, generated_outputs)
        ]).mean()
        
        loss.backward()
        optimizer.step()
        
    def test_train_zero(self):
        # sbatch --err=slurm_chip/test_train_zero.err --out=slurm_chip/test_train_zero.out runpycpu_chip.sh test_fission_unet.py TestFission.test_train_zero
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        dim=4
        shared_layer_type =MID_BLOCK
        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
        device,dtype=fission.get_device_dtype()
        timestep_list=[torch.randn((batch_size)).to(device,dtype) for _ in range(n_inputs)]
        optimizer=torch.optim.Adam([p for p in fission.parameters()])
        inputs=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        targets=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        
        optimizer.zero_grad()
        generated_outputs=fission.forward(inputs,timestep_list,zero_list=[0])
        print("gen put",generated_outputs)
        loss = torch.stack([
            F.mse_loss(i.float(), o.float())
            for i, o in zip(targets, generated_outputs)
        ]).mean()
        
        loss.backward()
        optimizer.step()
        
    def test_mixin(self):
        # sbatch --err=slurm_chip/test_mixin.err --out=slurm_chip/test_mixin.out runpycpu_chip.sh test_fission_unet.py TestFission.test_mixin
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        dim=4
        shared_layer_type =MID_BLOCK
        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
        
        name,param=[pair for pair in fission.named_parameters()][0]
        
        param.zero_()
        
        clone=param.clone()
        
        save_dir="test_pretreining"
        os.makedirs(save_dir,exist_ok=True)
        fission.save_pretrained(save_dir)
        
        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
        
        name,param=[pair for pair in fission.named_parameters()][0]
        
        self.assertFalse(param.sum()==0)
        
        fission=FissionUNet2DConditionModel.from_pretrained(save_dir)
        name,param=[pair for pair in fission.named_parameters()][0]
        
        self.assertTrue(param.sum()==0)
        
    def test_train_left_residuals(self):
        # sbatch --err=slurm_chip/test_train_zero.err --out=slurm_chip/test_train_zero.out runpycpu_chip.sh test_fission_unet.py TestFission.test_train_zero
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_inputs=2
        batch_size=2
        dim=4
        shared_layer_type =MID_BLOCK
        fission=FissionUNet2DConditionModel(n_inputs,shared_layer_type,2).to(device)
        device,dtype=fission.get_device_dtype()
        timestep_list=[torch.randn((batch_size)).to(device,dtype) for _ in range(n_inputs)]
        optimizer=torch.optim.Adam([p for p in fission.parameters()])
        inputs=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        targets=[torch.randn((batch_size,4,dim,dim)).to(device,dtype) for _ in range(n_inputs)]
        
        optimizer.zero_grad()
        generated_outputs=fission.forward(inputs,timestep_list,left_residuals=True)
        print("gen put",generated_outputs)
        loss = torch.stack([
            F.mse_loss(i.float(), o.float())
            for i, o in zip(targets, generated_outputs)
        ]).mean()
        
        loss.backward()
        optimizer.step()
        
    
        
                    
if __name__=="__main__":
    unittest.main()