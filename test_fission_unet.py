from fission_unet2 import FissionUNet2DConditionModel,UNET,MID_BLOCK
import unittest
import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TestFission(unittest.TestCase):
    def test_init(self):
        for shared_layer_type in [UNET, MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
                FissionUNet2DConditionModel(3,"UNetMidBlock2DCrossAttn",2,shared_layer_type=shared_layer_type)
        
    def test_from_pretrained(self):
        for shared_layer_type in [UNET, MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
                FissionUNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7","unet",2,shared_layer_type=shared_layer_type)
                
    def test_forward(self):
        n_inputs=2
        batch_size=2
        
        sample_list=[torch.zeros((batch_size,4,32,32)) for _ in range(n_inputs)]
        timestep_list=[torch.randn((batch_size)) for _ in range(n_inputs)]
        for shared_layer_type in [UNET, MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
        
                fission=FissionUNet2DConditionModel(n_inputs,"UNetMidBlock2DCrossAttn",2,shared_layer_type=shared_layer_type)
                fission.forward(sample_list,timestep_list)
                
    def test_forward_str(self):
        n_inputs=2
        batch_size=2
        
        str_list=[["hsdak" for _ in range(batch_size)] for __ in range(n_inputs)]
        sample_list=[torch.zeros((batch_size,4,32,32)) for _ in range(n_inputs)]
        timestep_list=[torch.randn((batch_size)) for _ in range(n_inputs)]
        
        tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="tokenizer")
        text_model=CLIPTextModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="text_model")
        
        for shared_layer_type in [UNET, MID_BLOCK]:
            with self.subTest(shared_layer_type=shared_layer_type):
        
                fission=FissionUNet2DConditionModel(n_inputs,"UNetMidBlock2DCrossAttn",2,shared_layer_type=shared_layer_type)
                fission.forward(sample_list,timestep_list,str_list=str_list)
        
        
                    
if __name__=="__main__":
    unittest.main()