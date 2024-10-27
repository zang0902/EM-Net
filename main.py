# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
import yaml
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import *

from networks.unetr import UNETR
from monai.networks.nets import resnet101,VNet,UNet,SwinUNETR,SegResNet
from networks.umamba import UMambaBot
from networks.transunet3d_network_architecture.transunet3d_model import Generic_TransUNet_max_ppbp
from networks.vtunet_network_architecture.vtunet import VTUNet
from networks.UXNet_3D.network_backbone import UXNET 

from models.em_net_model import EMNet
from loss_function.deep_supervision import MultipleOutputLoss2

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="./data/synapse_data_clean/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-2, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="sgd", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=3e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=9, type=int, help="number of output channels")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction") #prev 96
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction") #prev 96
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction") #prev 96
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--deep_supervision", action="store_true", help="use deep supervision for segmentation")
parser.add_argument("--dataset_name", default="synapse", type=str, help="model name")
parser.add_argument("--seed", default=3487, type=int, help="seed for initializing training. ")
parser.add_argument("--iter_per_epoch", default=250, type=int, help="iteration per epoch")
parser.add_argument("--train_by_iter", action="store_true", help="train by iter")
parser.add_argument("--crop_nums", default=4, type=int, help="number of samples (crop regions) to take in each list.")
parser.add_argument("--exp_mode", action="store_true", help="experiment and debug mode")

# Config path
parser.add_argument("--data_config", default="./config/dataset/synapse.yaml", type=str, help="path to data config file")
parser.add_argument("--model_config", default="./config/model/fpn_unet.yaml", type=str, help="path to model config file")
parser.add_argument("--trainer_config", default="./config/train/train.yaml", type=str, help="path to train config file")


def main():
    args = parser.parse_args()
    set_configs(args)  # Applying the config files
    args.amp = not args.noamp

    if args.logdir != 'test':
        args.logdir = os.path.join("runs", args.logdir)
    else:
        if "em-net" not in args.model_name:
            args.logdir = os.path.join("runs", args.model_name + '_' + args.dataset_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:  # Ours
            args.logdir = os.path.join("runs", args.model_name + '-' + args.variant + '_' + args.dataset_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"Log directory: {args.logdir}")
    
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)

def set_configs(args):
    with open(os.path.join("configs/datasets", f"{args.data_config}.yaml"), "r") as f:
        data_config = yaml.safe_load(f)
    with open(os.path.join("configs/models", f"{args.model_config}.yaml"), "r") as f:
        model_config = yaml.safe_load(f)
    with open(os.path.join("configs/trainers", f"{args.trainer_config}.yaml"), "r") as f:
        trainer_config = yaml.safe_load(f)
    
    for config in [data_config, model_config, trainer_config]:
        for k, v in config.items():
            setattr(args, k, v)

    args.iter_per_epoch =  250 // (args.crop_nums * args.batch_size // 2) if args.crop_nums * args.batch_size > 1 else 250
    print("Iter per epoch:", args.iter_per_epoch)
    print("batch size:", args.batch_size)

def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    torch.manual_seed(args.seed)
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    # Dataloader
    loader = get_loader(args)  # synapse or btcv
    if args.dataset_name=="flare":
        loader = get_loader_flare(args)
    elif "msd" in args.dataset_name:
        loader = get_loader_msd_pancreas(args)    

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir

    if args.model_name == "em-net":  # Our model and its variants
        model = EMNet(
            in_chans=args.in_channels,
            out_chans=args.out_channels,
            depths=args.depths,
            fft_nums=args.fft_nums,
            feat_size=args.feature_size,
            hidden_size=args.hidden_size,
            ds=args.deep_supervision,
            in_shpae=[args.roi_x, args.roi_y, args.roi_z],
            )
    elif args.model_name == "unet":  # Other comparison models
        model= UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            )
    elif args.model_name == "swin_unetr":
        model = SwinUNETR(img_size=(args.roi_x,args.roi_y,args.roi_z), 
                    in_channels=args.in_channels, out_channels=args.out_channels, feature_size=48)
    elif args.model_name == "segresnet":
        model= SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            dropout_prob=0.2,
            ) 
    elif args.model_name == "umamba_bot":  # For comparison
        from dynamic_network_architectures.building_blocks.residual import BasicBlockD
        model = UMambaBot(input_channels=args.in_channels,
                    deep_supervision=args.deep_supervision,
                    num_classes=args.out_channels,
                    n_stages=6,
                    conv_op = nn.modules.conv.Conv3d,
                    features_per_stage=[32, 64, 128, 256, 320, 320],
                    kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], 
                    n_conv_per_stage=[2, 2, 2, 2, 2, 2],        
                    n_conv_per_stage_decoder=[2,2,2,2,2],
                    conv_bias=True,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={'eps': 1e-05, 'affine': True},
                    nonlin=nn.modules.LeakyReLU,
                    nonlin_kwargs={'inplace': True},
                    block=BasicBlockD,
                    bottleneck_channels = None,
                    stem_channels=None,
                    )
    elif args.model_name == "vtunet":
        model = VTUNet(in_channel=args.in_channels,
                    out_channel=args.out_channels, 
                    input_size =  (128,128,128),
                    path_size = (4,4,4),
                    embed_dim=72,
                    pretrain_ckpt=None,).cuda()
    elif args.model_name == "transunet":
        class InitWeights_He(object):
            def __init__(self, neg_slope=1e-2):
                self.neg_slope = neg_slope

            def __call__(self, module):
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                    module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
                    if module.bias is not None:
                        module.bias = nn.init.constant_(module.bias, 0)

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        model = Generic_TransUNet_max_ppbp(input_channels=args.in_channels, 
                                        num_classes=args.out_channels,
                                        patch_size=[128,128,128],
                                        num_pool = 5,num_conv_per_stage=2,base_num_features=32, 
                                        conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                        pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                        feat_map_mul_on_downscale=2,conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d, 
                                        norm_op_kwargs=norm_op_kwargs, dropout_op=nn.Dropout3d,
                                        dropout_op_kwargs=dropout_op_kwargs,nonlin=net_nonlin, 
                                        nonlin_kwargs=net_nonlin_kwargs, deep_supervision=args.deep_supervision, 
                                        dropout_in_localization=False, final_nonlin=lambda x: x,
                                        weightInitializer=InitWeights_He(1e-2), is_masked_attn=True,
                                        is_max=True,upscale_logits=False, convolutional_pooling=True, 
                                        convolutional_upsampling= True, is_max_cls=True,is_max_ds=True,).cuda()
    elif args.model_name == "unetr":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
    elif args.model_name == "uxnet":
        model = UXNET(in_chans=args.in_channels,
                    out_chans=args.out_channels,
                    depths=[2, 2, 2, 2],feat_size=[48, 96, 192, 384],drop_path_rate=0,
                    layer_scale_init_value=1e-6,spatial_dims=3,).cuda()
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    loss_func = dice_loss
    if args.deep_supervision:
        ################# Here we wrap the loss for deep supervision ############
        # Borrowed from: https://github.com/Amshaker/unetr_plus_plus/, thanks Amshaker
        if args.model_name != "umamba_bot" and args.model_name != 'umamba_enc':  # Normal, fu-mamba
            # we need to know the number of outputs of the network
            net_numpool = len(model.feat_size[:3])  # Except for hidden feature. Todo: defined by args

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])  # 5 stages
            weights = np.array([1 / (2 ** (i+1)) if i != 0 else 1 / (2 ** i) for i in range(net_numpool)])  # 4 stages
        else:
            weights = np.array([1 / (2 ** i) for i in range(5)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        # weights[~mask] = 0
        weights = weights / weights.sum()
        print(weights)
        ds_loss_weights = weights
        # ds_loss_weights = weights.to_list()
        # now wrap the loss
        loss_func = MultipleOutputLoss2(loss_func, ds_loss_weights)
        ################# END ###################

    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
