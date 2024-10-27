import torch 
from torch import nn as nn
from transunet3d_model import Generic_TransUNet_max_ppbp
import torch.nn.functional as F
softmax_helper = lambda x: F.softmax(x, 1)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

if __name__ == "__main__":
   x = torch.randn((1, 1, 128, 128, 128))
   norm_op_kwargs = {'eps': 1e-5, 'affine': True}
   dropout_op_kwargs = {'p': 0, 'inplace': True}
   net_nonlin = nn.LeakyReLU
   net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
   model = Generic_TransUNet_max_ppbp(input_channels=1, 
                                    num_classes=9,
                                    patch_size=[128,128,128],
                                    num_pool = 5,
                                    num_conv_per_stage=2,
                                    base_num_features=32, 
                                    conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                    pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                    feat_map_mul_on_downscale=2,
                                    conv_op=nn.Conv3d, 
                                    norm_op=nn.InstanceNorm3d, 
                                    norm_op_kwargs=norm_op_kwargs, 
                                    dropout_op=nn.Dropout3d,
                                    dropout_op_kwargs=dropout_op_kwargs,
                                    nonlin=net_nonlin, 
                                    nonlin_kwargs=net_nonlin_kwargs, 
                                    deep_supervision=True, 
                                    dropout_in_localization=False, 
                                    final_nonlin=lambda x: x,
                                    weightInitializer=InitWeights_He(1e-2), 
                                    is_masked_attn=True,
                                    is_max=True,
                                    upscale_logits=False, 
                                    convolutional_pooling=True, 
                                    convolutional_upsampling= True, 
                                    is_max_cls=True,
                                    is_max_ds=True,)
   out = model(x)
   print(out[0]['pred_masks'].max(),  out[0]['pred_masks'].min(), out[0]['pred_masks'].shape)
   model = Generic_TransUNet_max_ppbp(input_channels=1, 
                                    num_classes=9,
                                    patch_size=[128,128,128],
                                    num_pool = 5,
                                    num_conv_per_stage=2,
                                    base_num_features=32, 
                                    conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                    pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                    feat_map_mul_on_downscale=2,
                                    conv_op=nn.Conv3d, 
                                    norm_op=nn.InstanceNorm3d, 
                                    norm_op_kwargs=norm_op_kwargs, 
                                    dropout_op=nn.Dropout3d,
                                    dropout_op_kwargs=dropout_op_kwargs,
                                    nonlin=net_nonlin, 
                                    nonlin_kwargs=net_nonlin_kwargs, 
                                    deep_supervision=True, 
                                    dropout_in_localization=False, 
                                    final_nonlin=lambda x: x,
                                    weightInitializer=InitWeights_He(1e-2), 
                                    is_masked_attn=True,
                                    is_max=True,
                                    upscale_logits=False, 
                                    convolutional_pooling=True, 
                                    convolutional_upsampling= True, 
                                    is_max_cls=True,
                                    is_max_ds=True,)
   x = torch.randn((1, 1, 96, 96, 96))
   out = model(x)
   print(out[0]['pred_masks'].max(),  out[0]['pred_masks'].min(), out[0]['pred_masks'].shape)
