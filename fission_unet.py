from diffusers.models.unets.unet_2d_blocks import get_down_block,get_mid_block,get_up_block
from diffusers import UNet2DConditionModel
import torch
from typing import Optional,List
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)

class FissionUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self,
                 n_inputs:int, #how many splits
                 total_up_blocks:int,
                 shared_up_blocks:int,
                 total_mid_blocks:int,
                 shared_mid_blocks:int,
                 num_layers:int,
                 in_channels:int,
                 out_channels:int,
                 block_out_channels:List[int],
                 down_block_type_list: List[str] = [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                    ],
                mid_block_type_list: List[str] = ["UNetMidBlock2DCrossAttn"],
                up_block_type_list: List[str] = ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        
                 
                 resnet_act_fn: str="silu",
                 resnet_eps:float=1e-05,
                transformer_layers_per_block: int = 1,
                num_attention_heads: Optional[int] = None,
                resnet_groups: Optional[int] = 32,
                cross_attention_dim: Optional[int] = 768,
                downsample_padding: Optional[int] = 1,
                dual_cross_attention: bool = False,
                use_linear_projection: bool = False,
                only_cross_attention: bool = False,
                upcast_attention: bool = False,
                resnet_time_scale_shift: str = "default",
                attention_type: str = "default",
                resnet_skip_time_act: bool = False,
                resnet_out_scale_factor: float = 1.0,
                cross_attention_norm: Optional[str] = None,
                attention_head_dim: Optional[int] = 8,
                downsample_type: Optional[str] = 'conv',
         
                dropout: float = 0.0,
                resolution_idx: Optional[int] = None,
                upsample_type: Optional[str] = None,
                use_cross_attention: Optional[bool]=False,
                output_scale_factor: float = 1.0,
                mid_block_only_cross_attention: bool = False,
                conv_in_kernel: int = 3,
                conv_out_kernel: int = 3,
                time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = 256,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embedding_act_fn: str = "silu",
        n_metadata:int=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert shared_mid_blocks<=total_mid_blocks, "shared_mid_blocks>total_mid_blocks"
        assert shared_up_blocks<=total_up_blocks, "shared_up_blocks>total_up_blocks"
        assert len(block_out_channels) in [total_up_blocks,total_up_blocks+1], "len(block_out_channels) not in [total_up_blocks,total_up_blocks+1]"
        
        self.n_inputs=n_inputs

        self.down=[[] for _ in range(n_inputs)]
        self.middle=[[]for _ in range(n_inputs)]
        self.up=[[] for _ in range(n_inputs)]
        self.shared_up=[]
        self.shared_down=[]
        self.shared_middle=[]
        
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )
        
        conv_channels=block_out_channels[0]
        
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in_list=torch.nn.ModuleList([ torch.nn.Conv2d(
            in_channels, conv_channels, kernel_size=conv_in_kernel, padding=conv_in_padding
        ) for _ in range(n_inputs)])
        
        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out_list = torch.nn.ModuleList([torch.nn.Conv2d(
            conv_channels, out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        ) for _ in range(n_inputs)])
        
        #if len(block_out_channels)==total_up_blocks:
        mid_block_channels=block_out_channels[-1]
        if len(block_out_channels)==total_up_blocks+1:
            block_out_channels=block_out_channels[:-1]
            
        print('block_out_channels',block_out_channels)
        
        down_in_channel_list=[block_out_channels[0]]+block_out_channels[:-1]
        down_out_channel_list=block_out_channels.copy()
        
        reversed_block_out_channels=block_out_channels[::-1]
        up_prev_output_channel_list=[reversed_block_out_channels[0]]+reversed_block_out_channels[:-1]
        up_out_channel_list=reversed_block_out_channels.copy()
        up_in_channel_list=reversed_block_out_channels[1:]+[reversed_block_out_channels[-1]]
        
        
        
        unshared_up_blocks=total_up_blocks-shared_up_blocks
        unshared_mid_blocks=total_mid_blocks-shared_mid_blocks
        
        shared_block_out_channels=block_out_channels[unshared_up_blocks:]
        block_out_channels=block_out_channels[:unshared_up_blocks+1]
        
        
        
        print('block_out_channels',block_out_channels)
        print('shared_block_out_channels=',shared_block_out_channels)
        
        
        reversed_block_out_channels=block_out_channels[::-1]
        reversed_shared_block_out_channels=shared_block_out_channels[::-1]
        
        print('reversed_shared_block_out_channels',reversed_shared_block_out_channels)
        print(reversed_block_out_channels, "reversed")
        
        temb_channels=time_embed_dim
        
        generic_kwargs={
            "temb_channels":temb_channels,
            "resnet_eps":resnet_eps,
            "resnet_act_fn":resnet_act_fn,
            "transformer_layers_per_block":transformer_layers_per_block,
            "num_attention_heads":num_attention_heads,
            "resnet_groups":resnet_groups,
            "cross_attention_dim":cross_attention_dim,
            "dual_cross_attention":dual_cross_attention,
            "use_linear_projection":use_linear_projection,
            "only_cross_attention":only_cross_attention,
            "upcast_attention":upcast_attention,
            "resnet_time_scale_shift":resnet_time_scale_shift,
            "attention_type":attention_type,
            "resnet_skip_time_act":resnet_skip_time_act,
            "resnet_out_scale_factor":resnet_out_scale_factor,
            "cross_attention_norm":cross_attention_norm,
            "attention_head_dim":attention_head_dim,
            "dropout":dropout,
          #  "downsample_padding":downsample_padding
        }
        mid_generic_kwargs=generic_kwargs.copy()
        del mid_generic_kwargs["only_cross_attention"]
        del mid_generic_kwargs['resnet_out_scale_factor']
       # del mid_generic_kwargs["downsample_padding"]
        
        up_list=[]
        
        for s in range(total_up_blocks):
            input_channel = down_in_channel_list[s] 
            output_channel= down_out_channel_list[s]
            up_block_type=up_block_type_list[s]
            down_block_type=down_block_type_list[s]
            is_final_down_block=s==total_up_blocks-1
            if s<unshared_up_blocks:
                for n in range(n_inputs):
                    self.down[n].append(
                        get_down_block(
                            down_block_type=down_block_type,
                            num_layers=num_layers,
                            in_channels=input_channel,
                            out_channels=output_channel,
                            downsample_type=downsample_type,
                            downsample_padding=downsample_padding,
                            add_downsample=not is_final_down_block,
                            **generic_kwargs
                        )
                    )
                    if n==0:
                        print(f"\t unshared down_block {down_block_type} {input_channel} {output_channel} {is_final_down_block}")
            else:
                self.shared_down.append(
                    get_down_block(
                        down_block_type=down_block_type,
                        num_layers=num_layers,
                        in_channels=shared_input_channel,
                        out_channels=shared_output_channel,
                        downsample_padding=downsample_padding,
                        #temb_channels=temb_channels,
                        add_downsample=not is_final_down_block,
                        **generic_kwargs
                    )
                )
            
                print(f"\t shared {down_block_type} {shared_input_channel}  {shared_output_channel} {is_final_down_block}")
                
        for m in range(total_mid_blocks):
            if m<unshared_mid_blocks:
                for n in range(n_inputs):
                    mid_block_type=mid_block_type_list[m]
                    self.middle[n].append(
                        get_mid_block(
                            mid_block_type=mid_block_type,
                            in_channels=mid_block_channels,
                            **mid_generic_kwargs
                        )
                    )
                    if n==0:
                        print(f"\t unshared mid block {mid_block_type} {unshared_mid_in_channels}")
            else:
                self.shared_middle.append(
                    get_mid_block(
                            mid_block_type=mid_block_type,
                            in_channels=mid_block_channels,
                            **mid_generic_kwargs
                        )
                )
        
        for u in range(total_up_blocks):
            reverse_input_channel=up_in_channel_list[s]
            reverse_output_channel = up_out_channel_list[s]
            prev_output_channel = up_prev_output_channel_list[s]
            
            if u<shared_up_blocks:
                
                
        
        
        
        for n in range(n_inputs):
            if unshared_up_blocks>0:
                for s in range(unshared_up_blocks):
                    input_channel = down_in_channel_list[s] 
                    output_channel= down_out_channel_list[s]
                    up_block_type=up_block_type_list[s]
                    down_block_type=down_block_type_list[s]
                    
                    is_final_up_block = False
                    is_final_down_block=False
                    
                    if s==unshared_up_blocks-1:
                        is_final_up_block==True
                        if shared_up_blocks==0:
                            is_final_down_block=True
                    self.down[n].append(
                        get_down_block(
                            down_block_type=down_block_type,
                            num_layers=num_layers,
                            in_channels=input_channel,
                            out_channels=output_channel,
                            downsample_type=downsample_type,
                            downsample_padding=downsample_padding,
                            add_downsample=not is_final_down_block,
                            **generic_kwargs
                        )
                    )
                    
                    if n==0:
                        print(f"\t unshared down_block {down_block_type} {input_channel} {output_channel} {is_final_down_block}")
                    
                    
                    reverse_input_channel=up_in_channel_list[s]
                    reverse_output_channel = up_out_channel_list[s]
                    prev_output_channel = up_prev_output_channel_list[s]
                    
                    
                    
                    
                    self.up[n].append(
                    get_up_block(up_block_type=up_block_type,
                                num_layers=num_layers+1,
                        in_channels=reverse_input_channel,
                        out_channels=reverse_output_channel,
                        prev_output_channel=prev_output_channel,
                        add_upsample=not is_final_up_block,
                        **generic_kwargs)
                    )
                    
                    if n==0:
                        up_list.append(f"\t unshared up_block {up_block_type} {reverse_input_channel} {reverse_output_channel}  {prev_output_channel} {is_final_up_block}")
                
                
                
                #self.down[n]=self.down[n][::-1]
            if unshared_up_blocks>0:
                unshared_mid_in_channels=mid_block_channels
            else:
                unshared_mid_in_channels=conv_channels
            
            for m in range(unshared_mid_blocks):
                mid_block_type=mid_block_type_list[m]
                self.middle[n].append(
                    get_mid_block(
                        mid_block_type=mid_block_type,
                        in_channels=unshared_mid_in_channels,
                        **mid_generic_kwargs
                    )
                )
                if n==0:
                    print(f"\t mid block {mid_block_type} {unshared_mid_in_channels}")

        shared_str=''
        for s in range(shared_up_blocks):
            shared_input_channel=down_in_channel_list[s+unshared_up_blocks]
            if s==0:
                shared_input_channel=shared_input_channel*n_inputs
            shared_output_channel = down_out_channel_list[s+unshared_up_blocks]
            
            up_block_type=up_block_type_list[s+unshared_up_blocks]
            down_block_type=down_block_type_list[s+unshared_up_blocks]
            
            is_final_up_block = False
            is_final_down_block=False
            
            if s==shared_up_blocks-1:
                is_final_down_block=True
                if unshared_up_blocks==0:
                    is_final_up_block=True
                    
            
            self.shared_down.append(
                get_down_block(
                        down_block_type=down_block_type,
                        num_layers=num_layers,
                        in_channels=shared_input_channel,
                        out_channels=shared_output_channel,
                        downsample_padding=downsample_padding,
                        #temb_channels=temb_channels,
                        add_downsample=not is_final_down_block,
                        **generic_kwargs
                    )
            )
            
            print(f"\t shared {down_block_type} {shared_input_channel}  {shared_output_channel} {is_final_down_block}")
            
            shared_reverse_output_channel = up_out_channel_list[s+unshared_up_blocks]
            shared_reverse_input_channel = up_in_channel_list[s+unshared_up_blocks]
            shared_prev_output_channel = up_prev_output_channel_list[s+unshared_up_blocks]
            
            #if s==0:
                
            
            self.shared_up.append(
                
                get_up_block(up_block_type,
                             num_layers+1,
                             shared_reverse_input_channel,
                             shared_reverse_output_channel,
                             shared_prev_output_channel,
                    #temb_channels,
                    add_upsample=not is_final_up_block,
                    **generic_kwargs)
            )
            
            shared_str+=f"\t shared {up_block_type} {shared_reverse_input_channel} {shared_reverse_output_channel} {shared_prev_output_channel} {is_final_up_block}\n"
        
            
        up_list.insert(0,shared_str)
        if unshared_up_blocks>0:
            shared_mid_in_channels=unshared_mid_in_channels*n_inputs
        if shared_up_blocks>0:
            shared_mid_in_channels=shared_block_out_channels[-1]
            
        for m in range(shared_mid_blocks):
            mid_block_type=mid_block_type_list[m+unshared_mid_blocks]
            self.shared_middle.append(
                get_mid_block(
                    mid_block_type=mid_block_type,
                    in_channels= shared_mid_in_channels, #this might need to change
                    **mid_generic_kwargs
                )
            )
            print(f"\t shared {mid_block_type} {shared_mid_in_channels}")
            
        if unshared_mid_blocks >0 or unshared_up_blocks>0:
            
            if unshared_mid_blocks>0: #unshared mid blocks
                merge_input_dim=unshared_mid_in_channels*n_inputs
                merge_output_dim=shared_mid_in_channels
            else:
                if shared_up_blocks>0: #shared up blocks
                    merge_input_dim= up_out_channel_list[shared_up_blocks+unshared_up_blocks-1] #the output dim of the last 
            
        self.shared_middle=torch.nn.ModuleList(self.shared)
        self.shared_down=torch.nn.ModuleList(self.shared_down)
        self.shared_up=torch.nn.ModuleList(self.shared_up)
        self.down=torch.nn.ModuleList([torch.nn.ModuleList(m) for m in  self.down])
        self.middle=torch.nn.ModuleList([torch.nn.ModuleList(m) for m in self.middle])
        self.up=torch.nn.ModuleList([torch.nn.ModuleList(m) for m in self.up])
        
        
        for row in up_list:
            print(row)
        
        
        self.time_embedding = TimestepEmbedding( #use same embedding model for both 
            timestep_input_dim,
            time_embed_dim,
            act_fn=time_embedding_act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        
        self.metadata_embedding=TimestepEmbedding( #for relative difference in camera angle and position for left and right n=6? 
            n_metadata,time_embed_dim,
            act_fn=time_embedding_act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        
        self.use_cross_attention=use_cross_attention
        
        
            
    def forward(self,
                sample_list:List[torch.Tensor],
                timestep_list:List[torch.Tensor],
                metadata:torch.Tensor=None,
                encoder_hidden_states_list:List[torch.Tensor]=None,
                combined_encoder_hidden_states:torch.Tensor=None):
        
        t_emb_list=[self.time_embedding(self.get_time_embed(sample,timestep) ) for sample,timestep in zip(sample_list,timestep_list)]
        
        print("time emb",self.get_time_embed(sample_list[0],timestep_list[0]).size())
        print("t emb list",t_emb_list[0].size())
        
        
        if metadata is not None:
            metadata_emb=self.metadata_embedding(metadata)
        
        residuals=[[] for _ in range(self.n_inputs)]
        processed_sample_list=[]
        for n in range(self.n_inputs):
            sample=sample_list[n]
            sample=self.conv_in_list[n](sample)
            
            residuals[n]=(sample,)
            
            
            for d,downsample_block in enumerate(self.down[n]):
                if n==0:
                    print("down",sample.size())
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb= t_emb_list[n],
                        encoder_hidden_states=encoder_hidden_states_list[n],
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb= t_emb_list[n],
                        #encoder_hidden_states=encoder_hidden_states_list[n],
                    )
                residuals[n]+=res_samples
                
            for m,mid_block in enumerate(self.middle[n]):
                if hasattr(mid_block, "has_cross_attention") and mid_block.has_cross_attention:
                    sample = mid_block(
                        sample,
                        temb= t_emb_list[n],
                        encoder_hidden_states=encoder_hidden_states_list[n],
                    )
                else:
                    sample = mid_block(
                        sample,
                        temb= t_emb_list[n],
                       # encoder_hidden_states=encoder_hidden_states_list[n],
                    )
            processed_sample_list.append(sample)
        sample=torch.cat(processed_sample_list,dim=1)
        shared_residuals=(sample,)
        if metadata is not None:
            shared_temb=torch.sum(torch.stack(t_emb_list+[metadata_emb]),dim=0)
        else:
            shared_temb=sum(t_emb_list)
            
        #print("shared ",shared_temb.size())
        #shared_temb=self.time_embedding(shared_temb)
        #print("shared 2",shared_temb.size())
        for d,downsample_block in enumerate(self.shared_down):
            #print("shared down",sample.size())
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=shared_temb,
                    encoder_hidden_states=encoder_hidden_states_list[n],
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=shared_temb,
                    #encoder_hidden_states=encoder_hidden_states_list[n],
                )
            shared_residuals+=res_samples
            print(d)
            for r in res_samples:
                print("\t res sample",r.size())
            
            
        
        for m,mid_block in enumerate(self.shared):
            print("shared mid",sample.size())
            if hasattr(mid_block, "has_cross_attention") and mid_block.has_cross_attention:
                sample = mid_block(
                    sample,
                    temb=shared_temb,
                    encoder_hidden_states=combined_encoder_hidden_states,
                )
            else:
                sample = mid_block(
                    sample,
                    temb=shared_temb,
                    # encoder_hidden_states=encoder_hidden_states_list[n],
                )
        print("shraed residuals",len(shared_residuals))   
        for u,upsample_block in enumerate(self.shared_up):
            res_samples = shared_residuals[-len(upsample_block.resnets) :]
            shared_residuals = shared_residuals[: -len(upsample_block.resnets)]
            
            #if type(res_samples)==tuple:
            print(u)
            for r in res_samples:
                print("\t res sample",r.size())
            if hasattr(upsample_block,"has_cross_attention") and upsample_block.has_cross_attention:
                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                '''if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]'''
                    
                    
                sample = upsample_block(
                        hidden_states=sample,
                        temb=shared_temb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states_list[n], #need to change this...
                    )
            else:
                sample = upsample_block(
                        hidden_states=sample,
                        temb=shared_temb,
                        res_hidden_states_tuple=res_samples,
                    )
                    
                    
                    
        split_sample_list=[sample for ]
        final_sample_list=[]
        for n in range(self.n_inputs):
            sample=split_sample_list[n]
            down_block_res_samples=residuals[n]
            for u,upsample_block in enumerate(self.up[n]):
                is_final_block = u == len(self.up_blocks) - 1
                
                if n==0:
                    print("up sampe",sample.size())

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
                if hasattr(upsample_block,"has_cross_attention") and upsample_block.has_cross_attention:
                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    '''if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]'''
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=t_emb_list[n],
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states_list[n],
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=t_emb_list[n],
                        res_hidden_states_tuple=res_samples,
                    )
            sample=self.conv_out_list[n](sample)
            final_sample_list.append(sample)
            
        return final_sample_list
        



    
class PrintingUnet(UNet2DConditionModel):
    def __init__(self, sample_size = None, in_channels = 4, out_channels = 4, center_input_sample = False, flip_sin_to_cos = True, freq_shift = 0, down_block_types = ..., mid_block_type = "UNetMidBlock2DCrossAttn", up_block_types = ..., only_cross_attention = False, block_out_channels = ..., layers_per_block = 2, downsample_padding = 1, mid_block_scale_factor = 1, dropout = 0, act_fn = "silu", norm_num_groups = 32, norm_eps = 0.00001, cross_attention_dim = 1280, transformer_layers_per_block = 1, reverse_transformer_layers_per_block = None, encoder_hid_dim = None, encoder_hid_dim_type = None, attention_head_dim = 8, num_attention_heads = None, dual_cross_attention = False, use_linear_projection = False, class_embed_type = None, addition_embed_type = None, addition_time_embed_dim = None, num_class_embeds = None, upcast_attention = False, resnet_time_scale_shift = "default", resnet_skip_time_act = False, resnet_out_scale_factor = 1, time_embedding_type = "positional", time_embedding_dim = None, time_embedding_act_fn = None, timestep_post_act = None, time_cond_proj_dim = None, conv_in_kernel = 3, conv_out_kernel = 3, projection_class_embeddings_input_dim = None, attention_type = "default", class_embeddings_concat = False, mid_block_only_cross_attention = None, cross_attention_norm = None, addition_embed_type_num_heads = 64):
        super().__init__(sample_size, in_channels, out_channels, center_input_sample, flip_sin_to_cos, freq_shift, down_block_types, mid_block_type, up_block_types, only_cross_attention, block_out_channels, layers_per_block, downsample_padding, mid_block_scale_factor, dropout, act_fn, norm_num_groups, norm_eps, cross_attention_dim, transformer_layers_per_block, reverse_transformer_layers_per_block, encoder_hid_dim, encoder_hid_dim_type, attention_head_dim, num_attention_heads, dual_cross_attention, use_linear_projection, class_embed_type, addition_embed_type, addition_time_embed_dim, num_class_embeds, upcast_attention, resnet_time_scale_shift, resnet_skip_time_act, resnet_out_scale_factor, time_embedding_type, time_embedding_dim, time_embedding_act_fn, timestep_post_act, time_cond_proj_dim, conv_in_kernel, conv_out_kernel, projection_class_embeddings_input_dim, attention_type, class_embeddings_concat, mid_block_only_cross_attention, cross_attention_norm, addition_embed_type_num_heads)
        
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            print("down ",i, input_channel,output_channel,is_final_block)
            
        print("mid ",block_out_channels[-1])
        
        reversed_block_out_channels = list(reversed(block_out_channels))
        
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            
            print("up ",i,input_channel,output_channel,prev_output_channel,is_final_block)
            
            
    def fake_forward(self,sample, timestep):
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, None)
        
        sample = self.conv_in(sample)
        
        down_block_res_samples = (sample,)
        for d,downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}


                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=None,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
            
            print(d,len(res_samples))
            for s in res_samples:
                print("\t",s.size())
                
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            print(i,len(res_samples))
            for s in res_samples:
                print("\t",s.size())
        
    
if __name__=="__main__":
    batch_size=1
    timestep=torch.randn((batch_size))
    
    block_out_channels=[16,32,96,128]
    
    p=PrintingUnet(block_out_channels=block_out_channels,down_block_types=[
        "DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D"
    ],up_block_types=[
        "UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"
    ],norm_num_groups=2,mid_block_type ="UNetMidBlock2D",layers_per_block=2)
    
    #p.forward(torch.randn((batch_size,4,64,64)),timestep,encoder_hidden_states=None)
    
    p.fake_forward(torch.randn((batch_size,4,64,64)),timestep,)
    
    for u,up_block in enumerate(p.up_blocks):
        break
        print(u)
        for r,resnet in enumerate(up_block.resnets):
            print("|-",r)
            print("  |-",resnet.in_channels) 
            print("  |-",resnet.out_channels)
    
    #exit(0)
    
    n_inputs=1
    total_up_blocks=4
    total_mid_blocks=1
    num_layers=2
    in_channels=4
    out_channels=4
    block_out_channels=block_out_channels
    batch_size=1
    down_block_type_list=[
        "DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D"
    ]
    
    mid_block_type_list=["UNetMidBlock2D","UNetMidBlock2D","UNetMidBlock2D"]
    
    up_block_type_list=[
        "UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"
    ]
    
    timestep_list=[torch.randn((batch_size)) for _ in range(n_inputs)]
    
    shared_mid_blocks=1
    shared_up_blocks=3
    
    unet=FissionUNet2DConditionModel(
                        n_inputs,
                        total_up_blocks,
                        shared_up_blocks,
                        total_mid_blocks,
                        shared_mid_blocks,
                        num_layers,
                        in_channels,out_channels,
                        block_out_channels,
                        resnet_act_fn="relu",
                        resnet_groups=4,
                        down_block_type_list=down_block_type_list,
                        mid_block_type_list=mid_block_type_list,
                        up_block_type_list=up_block_type_list
                    )
    for u,up_block in enumerate(unet.shared_up):
        break
        print(u)
        for r,resnet in enumerate(up_block.resnets):
            print("|-",r)
            print("  |-",resnet.in_channels) 
            print("  |-",resnet.out_channels) 
    print("my turn!")
    unet.forward([torch.randn((batch_size,4,64,64)) for _ in range(n_inputs)] ,timestep_list)