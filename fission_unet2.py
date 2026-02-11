from diffusers.models.unets.unet_2d_blocks import get_down_block,get_mid_block,get_up_block,UpBlock2D,CrossAttnUpBlock2D,apply_freeu
from diffusers import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.activations import get_activation
from experiment_helpers.metadata_unet import MetadataUNet2DConditionModel,WEIGHTS_PATH,JSON_PATH,MetadataMixin
import torch
from typing import Optional,List,Tuple,Dict,Any,Union
from transformers import CLIPTokenizer, CLIPTextModel
import json
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
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from peft import LoraConfig
import os

from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

UNET="unet"
MID_BLOCK="mid_block"
WEIGHT_SUFFIX="_weight_partial_unet.safetensors"

class PartialUNet2DConditionModel(MetadataUNet2DConditionModel):
    def forward_down(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        emb:Optional[torch.Tensor]=None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        metadata:List[Union[torch.Tensor, float, int]]=None,
        return_dict: bool = True,):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        if emb is None:
            # 1. time
            t_emb = self.get_time_embed(sample=sample, timestep=timestep)
            emb = self.time_embedding(t_emb, timestep_cond)

            class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
            if class_emb is not None:
                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            aug_emb = self.get_aug_embed(
                emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
            if self.config.addition_embed_type == "image_hint":
                aug_emb, hint = aug_emb
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)
                
        # CUSTOM: metadata
        if self.metadata_embedding is not None:
            assert metadata is not None
            if self.metadata_proj:
                assert len(metadata.shape) == 3 and metadata.shape[1] == self.num_metadata, \
                    f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"
            else:
                assert len(metadata.shape) == 2 and metadata.shape[1] == self.num_metadata, \
                    f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"

            md_bsz = metadata.shape[0]
            # invalid_metadata_mask = metadata == -1.  # (N, num_md)
            if self.metadata_proj is False:
                metadata = self.time_proj(metadata.view(-1)).view(md_bsz, self.num_metadata, -1)  # (N, num_md, D)
            # metadata[invalid_metadata_mask] = 0.
            metadata = metadata.to(dtype=self.dtype)
            for i, md_embed in enumerate(self.metadata_embedding):
                md_emb = md_embed(metadata[:, i, :])  # (N, D)
                emb = emb + md_emb  # (N, D)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:

            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
            
        return sample,emb,down_block_res_samples,is_controlnet,is_adapter,lora_scale,forward_upsample_size,upsample_size
            
    def forward_mid(self,
                    sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
        is_controlnet:bool,
        is_adapter:bool,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        
                    ):
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual
            
        return sample
    
    def forward_up(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
        down_block_res_samples:tuple[torch.Tensor],
        forward_upsample_size:bool,
        lora_scale:float,
        upsample_size:int,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,):
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
            
        return sample


    def forward(self,
                sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        emb:Optional[torch.Tensor]=None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,):
        
        #print("partial forward ",emb.size())
        sample,emb,down_block_res_samples,is_controlnet,is_adapter,lora_scale,forward_upsample_size,upsample_size=self.forward_down(
            sample=sample,
            timestep=timestep,
            emb=emb,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
        
        
        sample=self.forward_mid(
            sample,timestep,encoder_hidden_states=encoder_hidden_states,
            emb=emb,is_controlnet=is_controlnet,is_adapter=is_adapter,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
        
        sample=self.forward_up(
            sample=sample,
            timestep=timestep,
            
            encoder_hidden_states=encoder_hidden_states,
            emb=emb,down_block_res_samples=down_block_res_samples,
            forward_upsample_size=forward_upsample_size,
            lora_scale=lora_scale,
            upsample_size=upsample_size,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
        
        return sample
    
    def get_down_block_res_sample_dims(self,dim:Tuple[int],max_position_embeddings:int,cross_attention_dim:int,device:torch.DeviceObjType):
        if getattr(self, "down_block_res_sample_dims_dict",None) is None:
            self.down_block_res_sample_dims_dict={}
        
        if dim in self.down_block_res_sample_dims_dict:
            return self.down_block_res_sample_dims_dict[dim]
        
        null_sample=torch.zeros((1,4,dim[0],dim[1]),device=device)
        null_timestep=torch.zeros((1),device=device)
        null_encoder_hidden_states=torch.zeros((1,max_position_embeddings,cross_attention_dim),device=device)
        null_metadata=torch.zeros((1,self.num_metadata))
        '''if self.use_metadata and self.metadata_proj:
            null_metadata=torch.zeros((1,self.metadata_proj_dim))'''
            
        null_metadata=null_metadata.to(device)
        
        sample,emb,down_block_res_samples,is_controlnet,is_adapter,lora_scale,forward_upsample_size,upsample_size=self.forward_down(
            null_sample,null_timestep,null_encoder_hidden_states,metadata=null_metadata)
        
        result=[t.size()[1:] for t in down_block_res_samples]
        
        self.down_block_res_sample_dims_dict[dim]=result
        return result
    
    def get_null_down_block_res_sample(self,dim:Tuple[int],max_position_embeddings:int,cross_attention_dim:int,batch_size:int,device:torch.DeviceObjType):
        dim_list=self.get_down_block_res_sample_dims(dim,max_position_embeddings,cross_attention_dim,device)
        down_block_res_samples=()
        for d in dim_list:
            zero=torch.zeros((batch_size,*d),device=device)
            down_block_res_samples+=(zero,)
            
        return down_block_res_samples
    
        
        
    
class FissionUNet2DConditionModel(ModelMixin,MetadataMixin,ConfigMixin):
    @classmethod
    def from_pretrained_partials(cls,path:str,
                        n_inputs:int,
                        shared_layer_type:str,
                 n_mid_blocks:int,
                 use_metadata:bool=False,
                 num_metadata:int =1,
                 metadata_proj:bool=False,
                 metadata_proj_dim:int=4,
                 partial_use_metadata:bool=False,
                 partial_num_metadata:int =1,
                 partial_metadata_proj:bool=False,
                 partial_metadata_proj_dim:int=4,
                        shared_down_block_types: Tuple[str]=("CrossAttnDownBlock2D","CrossAttnDownBlock2D",),
                 shared_mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
                 shared_up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D",),
                 shared_block_out_channels: Tuple[int] = (1280, 1280),
                 **kwargs):
        
        model=cls(n_inputs=n_inputs,
                  shared_layer_type=shared_layer_type,
                  n_mid_blocks=n_mid_blocks,
                  use_metadata=use_metadata,
                  num_metadata=num_metadata,
                  metadata_proj=metadata_proj,
                  metadata_proj_dim=metadata_proj_dim,
                  partial_use_metadata=partial_use_metadata,
                  partial_num_metadata=partial_num_metadata,
                  partial_metadata_proj=partial_metadata_proj,
                  partial_metadata_proj_dim=partial_metadata_proj_dim,
                  shared_mid_block_type=shared_mid_block_type,
                  shared_up_block_types=shared_up_block_types,
                  shared_down_block_types=shared_down_block_types,
                  shared_block_out_channels=shared_block_out_channels,
                  partial_list=torch.nn.ModuleList([PartialUNet2DConditionModel.from_pretrained(path,subfolder="unet") for _ in range(n_inputs)]),
                  **kwargs)
        
        return model
    
    def __init__(self, n_inputs:int, 
                 shared_layer_type:str,
                 n_mid_blocks:int,
                 use_metadata:bool=False,
                 num_metadata:int =1,
                 metadata_proj:bool=False,
                 metadata_proj_dim:int=4,
                 partial_use_metadata:bool=False,
                 partial_num_metadata:int =1,
                 partial_metadata_proj:bool=False,
                 partial_metadata_proj_dim:int=4,
                 sample_size = None, in_channels = 4, out_channels = 4, center_input_sample = False, flip_sin_to_cos = True, freq_shift = 0, 
                 down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
                up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        
                 only_cross_attention = False, 
                 block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
                 shared_down_block_types: Tuple[str]=("CrossAttnDownBlock2D","CrossAttnDownBlock2D",),
                 shared_mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
                 shared_up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D",),
                 shared_block_out_channels: Tuple[int] = (1280, 1280),
                 layers_per_block = 2, downsample_padding = 1, mid_block_scale_factor = 1, dropout = 0, act_fn = "silu", norm_num_groups = 32, 
                 norm_eps = 0.00001, cross_attention_dim = 768, transformer_layers_per_block = 1, reverse_transformer_layers_per_block = None, 
                 encoder_hid_dim = None, encoder_hid_dim_type = None, attention_head_dim = 8, num_attention_heads = 4, dual_cross_attention = False, 
                 use_linear_projection = False, class_embed_type = None, addition_embed_type = None, addition_time_embed_dim = None, num_class_embeds = None, 
                 upcast_attention = False, resnet_time_scale_shift = "default", resnet_skip_time_act = False, resnet_out_scale_factor = 1, time_embedding_type = "positional", 
                 time_embedding_dim = None, time_embedding_act_fn = None, timestep_post_act = None, time_cond_proj_dim = None, conv_in_kernel = 3, 
                 conv_out_kernel = 3, projection_class_embeddings_input_dim = None, attention_type = "default", class_embeddings_concat = False, 
                 mid_block_only_cross_attention = None, cross_attention_norm = None, addition_embed_type_num_heads = 64,max_position_embeddings:int=77,
                 partial_list:torch.nn.ModuleList=None,
                 tokenizer:CLIPTokenizer=None, text_model:CLIPTextModel=None,
                 ):
        super().__init__()
        if partial_list is None:
            self.partial_list=torch.nn.ModuleList([PartialUNet2DConditionModel(sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=None, # ValueError: At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19.
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            ) for _ in range(n_inputs)])
        else:
            self.partial_list=partial_list
            
        self.partial_use_metadata=partial_use_metadata
        
        for unet in self.partial_list:
            unet.init_metadata(
                partial_use_metadata,partial_num_metadata,partial_metadata_proj, partial_metadata_proj_dim
            )
        self.n_inputs=n_inputs
        self.shared_layer_type=shared_layer_type
        self.cross_attention_dim=cross_attention_dim
        self.max_position_embeddings=max_position_embeddings
        self.tokenizer=tokenizer
        self.text_model=text_model
        
        
        shared_block_in_channels=block_out_channels[-1]*n_inputs
        
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )
        
        self.timestep_input_dim=timestep_input_dim
        self.time_embed_dim=time_embed_dim
        
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        
        self.init_metadata(use_metadata,num_metadata,metadata_proj,metadata_proj_dim)
        
        if shared_layer_type==UNET:
            self.shared_blocks=PartialUNet2DConditionModel(sample_size=sample_size,
            in_channels=shared_block_in_channels,
            out_channels=shared_block_in_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=shared_down_block_types,
            mid_block_type=shared_mid_block_type,
            up_block_types=shared_up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=shared_block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=None, # ValueError: At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19.
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,)
        elif shared_layer_type==MID_BLOCK:
            self.n_mid_blocks=n_mid_blocks
            blocks_time_embed_dim = time_embed_dim
            self.shared_blocks=torch.nn.ModuleList(
                [get_mid_block(shared_mid_block_type,temb_channels=time_embed_dim,in_channels=shared_block_in_channels,
                             resnet_eps=norm_eps,resnet_act_fn=act_fn,resnet_groups=norm_num_groups,output_scale_factor=mid_block_scale_factor,
                             transformer_layers_per_block=transformer_layers_per_block,num_attention_heads=num_attention_heads,
                             cross_attention_dim=cross_attention_dim,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim,
            dropout=dropout,  
                               ) for _ in range(n_mid_blocks)] 
            )
            
    
        
    def set_adapter(self,adapter_config:LoraConfig,adapter_name: str = "default"):
        for unet in self.partial_list:
            unet.add_adapter(adapter_config,adapter_name)
            
    def save_lora_adapter(
        self,
        save_directory,
        adapter_name: str = "default",
        upcast_before_saving: bool = False,
        safe_serialization: bool = True,
        weight_name: Optional[str] = WEIGHT_SUFFIX,
    ):
        for u,unet in enumerate(self.partial_list):
            unet.save_lora_adapter(
                save_directory,adapter_name,upcast_before_saving,safe_serialization,f"{u}{weight_name}"
            )
            
    def load_lora_adapter(self, pretrained_model_name_or_path_or_dict, prefix=None, hotswap: bool = False, weight_name=WEIGHT_SUFFIX,**kwargs):
        for u,unet in enumerate(self.partial_list):
            unet.load_lora_adapter(pretrained_model_name_or_path_or_dict,prefix,hotswap,weight_name=f"{u}{weight_name}",**kwargs)
            
    def unload_lora(self):
        for unet in self.partial_list:
            unet.unload_lora()
        
    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim
    
    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb
    
    def print_devices_and_dtypes(self):
        p = next(self.parameters())
        device, dtype = p.device, p.dtype
        print("self",device,dtype)
        if len(self.partial_list)>0:
            p = next(self.partial_list[0].parameters())
            device, dtype = p.device, p.dtype
            print("partial",device,dtype)
            
        for name in ["shared_blocks","metadata_embedding","time_embedding","text_model"]:
            if getattr(self,name,None) is not None:
                model=getattr(self,name)
                p = next(model.parameters())
                device, dtype = p.device, p.dtype
                print(name,device,dtype)
                
    def get_device_dtype(self)->Tuple:
        p = next(self.parameters())
        device, dtype = p.device, p.dtype
        return device,dtype
    
    
    def shared_forward(self,sample_list,
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,):
        

        sample=torch.cat(sample_list,dim=1)
        

        
        
        if self.shared_layer_type == UNET:
            sample=self.shared_blocks.forward(sample,
                                              None,
                encoder_hidden_states=encoder_hidden_states,
                emb=emb,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,).sample[0]
            
        elif self.shared_layer_type==MID_BLOCK:
            #t_emb = self.get_time_embed(sample=sample, timestep=timestep)
            
            for _ in range(self.n_mid_blocks):
                if hasattr(self.shared_blocks[0], "has_cross_attention") and self.shared_blocks[0].has_cross_attention:
                    sample=self.shared_blocks[0].forward(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.shared_blocks[0].forward(sample, emb)
        
        return sample.chunk(self.n_inputs,dim=1)
        
        
    def forward(self,sample_list: List[torch.Tensor],
        timestep_list: List[Union[torch.Tensor, float, int]],
        str_list:List[List[str]]=None,
        token_id_list:List[torch.Tensor]=None,
        encoder_hidden_states_list: List[torch.Tensor]=None,
        shared_encoder_hidden_states: torch.Tensor=None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        metadata:List[Union[torch.Tensor, float, int]]=None,
        partial_metadata_list:List[List[Union[torch.Tensor, float, int]]]=None,
        zero_list:List[int]=[],
        left_residuals:bool=False,
        ):
        new_sample_list=[]
        down_block_res_samples_list=[]
        
        emb_list=[self.time_embedding(self.get_time_embed(sample=sample, timestep=timestep), timestep_cond) for timestep,sample in zip(timestep_list,sample_list)]
        emb=emb_list[0]
        for e in emb_list[1:]:
            emb=emb+e
            
        # CUSTOM: metadata
        if self.metadata_embedding is not None:
            assert metadata is not None
            if self.metadata_proj:
                assert len(metadata.shape) == 3 and metadata.shape[1] == self.num_metadata, \
                    f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"
            else:
                assert len(metadata.shape) == 2 and metadata.shape[1] == self.num_metadata, \
                    f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"

            md_bsz = metadata.shape[0]
            # invalid_metadata_mask = metadata == -1.  # (N, num_md)
            if self.metadata_proj is False:
                metadata = self.time_proj(metadata.view(-1)).view(md_bsz, self.num_metadata, -1)  # (N, num_md, D)
            # metadata[invalid_metadata_mask] = 0.
            metadata = metadata.to(dtype=self.dtype)
            for i, md_embed in enumerate(self.metadata_embedding):
                md_emb = md_embed(metadata[:, i, :])  # (N, D)
                emb = emb + md_emb  # (N, D)
            
        batch_size=sample_list[0].size()[0]
        dim=(sample_list[0].size()[-2],sample_list[0].size()[-1])
        device,dtype=self.get_device_dtype()
        
        if encoder_hidden_states_list is None and shared_encoder_hidden_states is None:
            
            
                
            if token_id_list is not None and self.text_model is not None:
                encoder_hidden_states_list=[self.text_model(input_ids.to(device)).last_hidden_state for input_ids in token_id_list ]
                step=10 #concat the first n=step tokens of each token id sequence
                #print("shared???",torch.cat([input_ids[:,0:step] for i,input_ids in enumerate(token_id_list) ],dim=1))
                shared_encoder_hidden_states=self.text_model(torch.cat([input_ids[:,0:step] for i,input_ids in enumerate(token_id_list) ],dim=1)).last_hidden_state
            elif str_list is not None and self.tokenizer is not None and self.text_model is not None:
                
                #assume str list is (n x b)
                batched_str_list=[]
                #print(f"parsing str list n ={len(str_list[0])} b= {len(str_list)}",)
                for i in range(len(str_list[0])):
                    s=" "
                    for j in range(len(str_list)):
                        s+=str_list[j][i]+" "
                    batched_str_list.append(s)
                
                #print("batche dstr list",batched_str_list)
                shared_input_ids=torch.stack([self.tokenizer(bs,padding="max_length",max_length=self.tokenizer.model_max_length, return_tensors="pt",).input_ids for bs in batched_str_list]).to(device)
                #print("shard input ids",shared_input_ids.size())
                shared_encoder_hidden_states=self.text_model(
                    shared_input_ids
                    ).last_hidden_state
                token_id_list=[self.tokenizer(s,padding="max_length",
                    max_length=self.tokenizer.model_max_length, return_tensors="pt",).input_ids for s in str_list]
                encoder_hidden_states_list=[self.text_model(input_ids.to(device)).last_hidden_state for input_ids in token_id_list ]
            else:
                encoder_hidden_states_list=[torch.zeros((batch_size,self.max_position_embeddings,self.cross_attention_dim)).to(device,dtype) for _ in range(self.n_inputs)]
                shared_encoder_hidden_states=torch.zeros((batch_size,self.max_position_embeddings,self.cross_attention_dim)).to(device,dtype)
        if len(encoder_hidden_states_list)==0:
            encoder_hidden_states_list=[torch.zeros((batch_size,self.max_position_embeddings,self.cross_attention_dim)).to(device,dtype) for _ in range(self.n_inputs)]
                        
        #print('encoder_hidden_states_list',encoder_hidden_states_list[0].size())
        #print("encoer shared",shared_encoder_hidden_states.size())
        #print("sample_list",sample_list[0].size())
            
        for n,(sample,timestep,encoder_hidden_states,) in enumerate(zip(sample_list,timestep_list,encoder_hidden_states_list)):
            if partial_metadata_list is not None and partial_metadata_list is not []:
                partial_metadata=partial_metadata_list[n]
            else:
                partial_metadata=None
            sample,emb,down_block_res_samples,is_controlnet,is_adapter,lora_scale,forward_upsample_size,upsample_size=self.partial_list[n].forward_down(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
                metadata=partial_metadata,
            )
            #print('forward_down emb',emb.size())
            new_sample_list.append(sample)
            down_block_res_samples_list.append(down_block_res_samples)
        sample_list=self.shared_forward(new_sample_list, 
                                 shared_encoder_hidden_states,
                                  emb=emb,
                                  class_labels=class_labels,
                                    timestep_cond=timestep_cond,
                                    attention_mask=attention_mask,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_kwargs,
                                    down_block_additional_residuals=down_block_additional_residuals,
                                    mid_block_additional_residual=mid_block_additional_residual,
                                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                                    encoder_attention_mask=encoder_attention_mask,
                                    return_dict=return_dict,)
        
        final_sample_list=[]
        '''for data_list,name in zip(
            [timestep_list,encoder_hidden_states_list,down_block_res_samples_list,emb_list,sample_list],
            ['timestep_list','encoder_hidden_states_list','down_block_res_samples_list','emb_list','sample_list']
        ):
            print(name,len(data_list))'''
        if left_residuals:
            for j in range(1,len(down_block_res_samples_list)):
                down_block_res_samples_list[j]=down_block_res_samples_list[0]
        for n,(timestep,encoder_hidden_states,down_block_res_samples,emb,sample) in enumerate(
                zip(timestep_list,encoder_hidden_states_list,down_block_res_samples_list,emb_list,sample_list)):
            if n in zero_list:
                
                down_block_res_samples=self.partial_list[n].get_null_down_block_res_sample(dim,self.tokenizer.model_max_length,self.cross_attention_dim,batch_size,device) #(self,dim:int,max_position_embeddings:int,cross_attention_dim:int,batch_size:int):
            final_sample=self.partial_list[n].forward_up(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                emb=emb,
                down_block_res_samples=down_block_res_samples,
                forward_upsample_size=forward_upsample_size,
                lora_scale=lora_scale,
                upsample_size=upsample_size,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
                #metadata=partial_metadata,
            )
            final_sample_list.append(final_sample)
        
        return final_sample_list
        
            
        
        
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for use_metadata in [True,False]:
        for n_inputs in [1,2,3]:
            for partial_use_metadata in [True,False]:
                fission=FissionUNet2DConditionModel(n_inputs,MID_BLOCK,2,use_metadata=use_metadata,num_metadata=3,partial_use_metadata=partial_use_metadata)
                print('use_metadata,partial_use_metadata,n_inputs',use_metadata,partial_use_metadata,n_inputs, print(len(fission.state_dict())))
                print("partial[0]",len(fission.partial_list[0].state_dict()))