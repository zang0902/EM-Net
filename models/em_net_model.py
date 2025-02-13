# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Sequence, Tuple, Union, Optional

from monai.networks.blocks.dynunet_block import get_conv_layer, UnetBasicBlock, UnetResBlock
# from monai.networks.layers.utils import get_act_layer, get_norm_layer
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
import numpy as np
from mamba_ssm import Mamba
import torch.nn.functional as F 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, stage=1, d_state = 16, d_conv = 4, expand = 2, pos_embed=True, in_shape=[128, 128, 128]):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
        )
        x = in_shape[0] // 2**(stage+1)
        y = in_shape[1] // 2**(stage+1)
        z = in_shape[2] // 2**(stage+1)
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, x*y*z, self.dim))  # Temp n*n*n/2

        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim//4)
        self.conv51 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

    def forward(self, x):
        B, C = x.shape[:2]
 
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        if self.pos_embed is not None:  
            x_flat = x_flat + self.pos_embed    
        x_norm = self.norm(x_flat)
        x_mamba = self.norm1(self.mamba(x_norm))
        x_spatial = self.norm2(self.mamba(self.mlp(x_norm.transpose(-1, -2).reshape(B, C, *img_dims)).reshape(B, C, n_tokens).transpose(-1, -2)))

        out = x_flat + self.gamma * (x_mamba + x_spatial)
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)

        attn = self.conv51(out)
        out = out + self.conv8(attn)

        return out

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class FFParser_n(nn.Module):
    def __init__(self, dim, h=128, w=239, d=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, d, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W, D = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfftn(x, s=(H, W, D), dim=(2, 3, 4), norm='ortho')

        x = x.reshape(B, C, H, W, D)

        return x

class Spectral_Layer(nn.Module):
    def __init__(self, dim, stage=1, in_shape=[128, 128, 128]):
        super().__init__()
        self.dim = dim

        self.h = in_shape[0] // 2**(stage+1)
        self.w = in_shape[1] // 2**(stage+1)
        self.d = in_shape[2] // 2**(stage+2) + 1

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim//2)
        self.ffp_module = FFParser_n(dim, h=self.h, w=self.w, d=self.d)

    def forward(self, x):
        B, C = x.shape[:2]
        # B, C, DIM1, DIM2, DIM3
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # print(x.shape,'shape')

        x_reshape = x.reshape(B, C, n_tokens).transpose(-1, -2)
        norm1_x = self.norm1(x_reshape)
        norm1_x = norm1_x.reshape(B, C, *img_dims)
        x_fft = self.ffp_module(norm1_x)
        # print(x_fft.shape, 'xfft')
        norm2_x_fft = self.norm2(x_fft.reshape(B, C, n_tokens).transpose(-1, -2))
        x_spatial = self.mlp(norm2_x_fft.transpose(-1, -2).reshape(B, C, *img_dims))
        out_all = x + x_spatial
        new_out = out_all.transpose(-1, -2).reshape(B, C, *img_dims)
        return new_out
    
class EMNetUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
        stage: int = 1,
        in_shape: list = [128, 128, 128],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        # print("<<<<< this is new up block >>>>>>>>>")
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.decoder_block = nn.ModuleList()
        
        if res_block:
            stage_blocks = []
            for i in range(3):
                stage_blocks.append(MambaLayer(dim=out_channels, stage=stage-1, 
                                               d_state=16, d_conv=4, expand=2, 
                                               pos_embed=True, in_shape=in_shape))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        else:
            conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.decoder_block.append(conv_block)

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)
        return out

class EMNet(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        conv_decoder: bool = False,
        fft_nums: list = [2,2,2,2],  # FFT layer numbers in each stage
        res_block: bool = True,
        ds: bool = False,
        spatial_dims=3,
        in_shpae=[128, 128, 128],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.ds = ds  # Deep supervision
        self.spatial_dims = spatial_dims
        self.mamba_encoder = MambaEncoder(depths=self.depths,
                                          dims=self.feat_size,
                                          fft_nums=fft_nums,
                                          drop_path_rate=self.drop_path_rate,
                                          layer_scale_init_value=self.layer_scale_init_value,
                                          in_shape=in_shpae,
                            )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.use_conv = conv_decoder
        if self.use_conv:
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.decoder4 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                in_shape=in_shpae,
                stage=4,
            )
            self.decoder3 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=3,
                in_shape=in_shpae,
            )
            self.decoder2 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=2,
                in_shape=in_shpae,
            )
            self.decoder1 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                # upsample_kernel_size=2,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=False,
                stage=1,
                in_shape=in_shpae,
            )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)  # Final outputs, stage 1
        if self.ds:
            self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
            self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        # b,1,96,96,96-> b,48,48,48,48; b,96,24,24,24; b,192,12,12,12; b,384,6,6,6
        outs = self.mamba_encoder(x_in) 
        enc1 = self.encoder1(x_in)

        #### Normal skip connections
        dec3 = self.decoder4(outs[3], outs[2])
        dec2 = self.decoder3(dec3, outs[1])
        dec1 = self.decoder2(dec2, outs[0])

        # b,48,96,96,96 -> b,48,96,96,96
        out = self.decoder1(dec1, enc1)
        # b,48,96,96,96 -> b,out_c,96,96,96
        if self.ds:
            logits = [self.out(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out(out)
        return logits
    
class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], fft_nums = [2,2,0,0],
                 dims=[48, 96, 192, 384], in_shape = [128, 128, 128],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        
        self.depths = depths
        self.fft_nums = fft_nums
        self.dims = dims
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.length = len(self.dims)
        out_indices = [i for i in range(self.length)]
        stem = nn.Sequential(
            #   nn.Conv3d(in_chans, self.dims[0], kernel_size=7, stride=2, padding=3),
            # nn.Conv3d(in_chans, self.dims[0], kernel_size=2, stride=2),
            # nn.Conv3d(self.dims[0], self.dims[0], kernel_size=2, stride=2),
            nn.Conv3d(in_chans, self.dims[0], kernel_size=4, stride=4),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        for i in range(self.length-1):
            downsample_layer = nn.Sequential(
                LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.length):
            stage = nn.Sequential(
                *[Spectral_Layer(dim=self.dims[i], stage=i+1, in_shape=in_shape) 
                  if j<self.fft_nums[i]
                  else MambaLayer(dim=self.dims[i], stage=i+1, in_shape=in_shape) 
                  for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(self.length):
            layer = norm_layer(self.dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(self.dims[i_layer], 4 * self.dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(self.length):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class EMNet_Bot(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        conv_decoder: bool = False,
        fft_nums: list = [2,2,2,2],  # FFT layer numbers in each stage
        res_block: bool = True,
        ds: bool = False,
        spatial_dims=3,
        in_shpae=[128, 128, 128],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.ds = ds  # Deep supervision
        self.spatial_dims = spatial_dims
        self.mamba_encoder = MambaEncoder_Bot(depths=self.depths,
                                          dims=self.feat_size,
                                          fft_nums=fft_nums,
                                          drop_path_rate=self.drop_path_rate,
                                          layer_scale_init_value=self.layer_scale_init_value,
                                          in_shape=in_shpae,
                            )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder_hidden = MambaLayer(dim=self.hidden_size, stage=4, in_shape=in_shpae) 
        self.use_conv = conv_decoder
        if self.use_conv:
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.decoder4 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                in_shape=in_shpae,
                stage=4,
            )
            self.decoder3 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=3,
                in_shape=in_shpae,
            )
            self.decoder2 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=2,
                in_shape=in_shpae,
            )
            self.decoder1 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                # upsample_kernel_size=2,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=False,
                stage=1,
                in_shape=in_shpae,
            )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)  # Final outputs, stage 1
        if self.ds:
            self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
            self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        # b,1,96,96,96-> b,48,48,48,48; b,96,24,24,24; b,192,12,12,12; b,384,6,6,6
        outs = self.mamba_encoder(x_in) 
        enc1 = self.encoder1(x_in)
        hidden = self.encoder_hidden(outs[3])

        #### Normal skip connections
        dec3 = self.decoder4(hidden, outs[2])
        dec2 = self.decoder3(dec3, outs[1])
        dec1 = self.decoder2(dec2, outs[0])

        # b,48,96,96,96 -> b,48,96,96,96
        out = self.decoder1(dec1, enc1)
        # b,48,96,96,96 -> b,out_c,96,96,96
        if self.ds:
            logits = [self.out(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out(out)
        return logits
    
class MambaEncoder_Bot(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], fft_nums = [2,2,0,0],
                 dims=[48, 96, 192, 384], in_shape = [128, 128, 128],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        
        self.depths = depths
        self.fft_nums = fft_nums
        self.dims = dims
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.length = len(self.dims)
        out_indices = [i for i in range(self.length)]
        stem = nn.Sequential(
            #   nn.Conv3d(in_chans, self.dims[0], kernel_size=7, stride=2, padding=3),
            # nn.Conv3d(in_chans, self.dims[0], kernel_size=2, stride=2),
            # nn.Conv3d(self.dims[0], self.dims[0], kernel_size=2, stride=2),
            nn.Conv3d(in_chans, self.dims[0], kernel_size=4, stride=4),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        for i in range(self.length-1):
            downsample_layer = nn.Sequential(
                LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.length):
            stage = nn.Sequential(
                *[Spectral_Layer(dim=self.dims[i], stage=i+1, in_shape=in_shape) 
                  if j<self.fft_nums[i]
                  else MambaLayer(dim=self.dims[i], stage=i+1, in_shape=in_shape) 
                  for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(self.length):
            layer = norm_layer(self.dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(self.dims[i_layer], 4 * self.dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(self.length):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class EMNet_Conv_Enc(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        conv_decoder: bool = False,
        fft_nums: list = [2,2,2,2],  # FFT layer numbers in each stage
        res_block: bool = True,
        ds: bool = False,
        spatial_dims=3,
        in_shpae=[128, 128, 128],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.ds = ds  # Deep supervision
        self.spatial_dims = spatial_dims
        self.mamba_encoder = MambaEncoder_Conv_Enc(depths=self.depths,
                                          dims=self.feat_size,
                                          fft_nums=fft_nums,
                                          drop_path_rate=self.drop_path_rate,
                                          layer_scale_init_value=self.layer_scale_init_value,
                                          in_shape=in_shpae,
                            )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.use_conv = conv_decoder
        if self.use_conv:
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.decoder4 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                in_shape=in_shpae,
                stage=4,
            )
            self.decoder3 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=3,
                in_shape=in_shpae,
            )
            self.decoder2 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=2,
                in_shape=in_shpae,
            )
            self.decoder1 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                # upsample_kernel_size=2,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=False,
                stage=1,
                in_shape=in_shpae,
            )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)  # Final outputs, stage 1
        if self.ds:
            self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
            self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        # b,1,96,96,96-> b,48,48,48,48; b,96,24,24,24; b,192,12,12,12; b,384,6,6,6
        outs = self.mamba_encoder(x_in) 
        enc1 = self.encoder1(x_in)

        #### Normal skip connections
        dec3 = self.decoder4(outs[3], outs[2])
        dec2 = self.decoder3(dec3, outs[1])
        dec1 = self.decoder2(dec2, outs[0])

        # b,48,96,96,96 -> b,48,96,96,96
        out = self.decoder1(dec1, enc1)
        # b,48,96,96,96 -> b,out_c,96,96,96
        if self.ds:
            logits = [self.out(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out(out)
        return logits
    
class MambaEncoder_Conv_Enc(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], fft_nums = [2,2,0,0],
                 dims=[48, 96, 192, 384], in_shape = [128, 128, 128],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        
        self.depths = depths
        self.fft_nums = fft_nums
        self.dims = dims
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.length = len(self.dims)
        out_indices = [i for i in range(self.length)]
        stem = nn.Sequential(
            #   nn.Conv3d(in_chans, self.dims[0], kernel_size=7, stride=2, padding=3),
            # nn.Conv3d(in_chans, self.dims[0], kernel_size=2, stride=2),
            # nn.Conv3d(self.dims[0], self.dims[0], kernel_size=2, stride=2),
            nn.Conv3d(in_chans, self.dims[0], kernel_size=4, stride=4),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        for i in range(self.length-1):
            downsample_layer = nn.Sequential(
                LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.length):
            stage = nn.Sequential(
                *[Spectral_Layer(dim=self.dims[i], stage=i+1, in_shape=in_shape) 
                  if j<self.fft_nums[i]
                  else MambaLayer_Conv(dim=self.dims[i], stage=i+1, in_shape=in_shape) 
                  for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(self.length):
            layer = norm_layer(self.dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(self.dims[i_layer], 4 * self.dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(self.length):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class MambaLayer_Conv(nn.Module):
    def __init__(self, dim, stage=1, d_state = 16, d_conv = 4, expand = 2, pos_embed=True, in_shape=[128, 128, 128]):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
        )
        x = in_shape[0] // 2**(stage+1)
        y = in_shape[1] // 2**(stage+1)
        z = in_shape[2] // 2**(stage+1)
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, x*y*z, self.dim))  # Temp n*n*n/2

        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
        self.res_block = BasicBlockD(conv_op=nn.Conv3d, input_channels=dim, 
                                     output_channels=dim, kernel_size=3, 
                                     stride=1, conv_bias=True, norm_op=nn.InstanceNorm3d,
                                     norm_op_kwargs={'eps': 1e-5, 'affine': True},
                                     nonlin=nn.modules.LeakyReLU,
                                     nonlin_kwargs={'inplace': True},
                                     )
        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim//4)
        self.conv51 = UnetResBlock(3, dim*2, dim, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

    def forward(self, x):
        B, C = x.shape[:2]
 
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        if self.pos_embed is not None:  
            x_flat = x_flat + self.pos_embed    
        x_norm = self.norm(x_flat)
        x_mamba = self.norm1(self.mamba(x_norm))
        x_spatial = self.norm2(self.mlp(x_norm.transpose(-1, -2).reshape(B, C, *img_dims)).reshape(B, C, n_tokens).transpose(-1, -2))
        
        out = x_flat + self.gamma * (x_mamba + x_spatial)
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)

        out = torch.concat([out, self.res_block(x)], dim=1)

        attn = self.conv51(out)
        # out = out + self.conv8(attn)  # 和图上画的不一样
        out = attn + self.conv8(attn)

        return out

if __name__ == "__main__":
    model = EMNet(in_chans=1,
        out_chans=9,
        depths=[3,3,3,3],
        # feat_size=[96, 192, 384, 768],
        feat_size=[48, 96, 192, 384],
        hidden_size=384,
        # feat_size=[32, 64, 128, 256],
        # hidden_size=256,
        fft_nums=[0,0,0,0],
        conv_decoder=False,
        in_shpae=[128, 128, 128],
        ).cuda()
    
    # model = EMNet_Bot(in_chans=1,
    #     out_chans=9,
    #     depths=[3,3,3,3],
    #     # feat_size=[96, 192, 384, 768],
    #     feat_size=[48, 96, 192, 384],
    #     hidden_size=384,
    #     fft_nums=[2,2,0,0],
    #     conv_decoder=False,
    #     in_shpae=[128, 128, 128],
    #     ).cuda()

    # model = EMNet_Conv_Enc(in_chans=1,
    #     out_chans=9,
    #     depths=[3,3,3,3],
    #     # feat_size=[96, 192, 384, 768],
    #     # feat_size=[48, 96, 192, 384],
    #     # hidden_size=384,
    #     feat_size=[32, 64, 128, 256],
    #     hidden_size=256,
    #     fft_nums=[0,0,0,0],
    #     conv_decoder=False,
    #     in_shpae=[128, 128, 128],
    #     ).cuda()

    # t1 = torch.rand(1, 1, 128, 128, 128).cuda()
    # out = model(t1)


    # from thop import profile
    # from ptflops import get_model_complexity_info  # 使用ptflops


    # t1 = torch.rand(1, 1, 128, 128, 128).cuda()

    # flops, params = profile(model, inputs=(t1, ))
    # print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    # print("---|---|---")
    # print(" | %.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))

    # with open('All parameters - Integrated.txt', 'w') as f:
    #     macs, params = get_model_complexity_info(model, (1, 128, 128, 128), as_strings=True, print_per_layer_stat=True, ost=f)
    #     print("%s |%s |%s" % ("Model", macs, params))

    # from torch_flops import TorchFLOPsByFX
    # x = torch.rand(1, 1, 128, 128, 128).cuda()
   
    # # =========
    # print("*" * 40 + " torch_flops " + "*" * 40)
    # flops_counter = TorchFLOPsByFX(model)
    # # flops_counter.graph_model.graph.print_tabular()
    # flops_counter.propagate(x)
    # flops_counter.print_result_table()
    # flops_1 = flops_counter.print_total_flops(show=False)
    # print(f"torch_flops: {flops_1} FLOPs")
    # print("=" * 80)

    # calflops  
    model.eval()
    from calflops import calculate_flops  
    flops, macs, params = calculate_flops(model, input_shape=(1, 1, 128, 128, 128))  
    print(flops)
    print(macs)
    print(params)
