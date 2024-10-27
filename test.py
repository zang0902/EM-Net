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
import yaml

import numpy as np
import torch
import torch.nn as nn

from trainer import dice
from utils.data_utils import get_loader, get_loader_msd_pancreas
from monai.inferers import sliding_window_inference

from networks.unetr import UNETR
from monai.networks.nets import resnet101,VNet,UNet,SwinUNETR,SegResNet
from networks.umamba import UMambaBot
from networks.transunet3d_network_architecture.transunet3d_model import Generic_TransUNet_max_ppbp
from networks.vtunet_network_architecture.vtunet import VTUNet
from networks.UXNet_3D.network_backbone import UXNET 

from models.em_net_model import EMNet
from loss_function.deep_supervision import MultipleOutputLoss2

from cal_metrics import hd
import pandas as pd
import time
from tqdm import tqdm
from ptflops import get_model_complexity_info  # 使用ptflops


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=32, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=9, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--dataset_name", default="synapse", type=str, help="model name")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--need_flops", action="store_true", help="calculate FLOPs")

# Config path
parser.add_argument("--data_config", default="./config/dataset/synapse.yaml", type=str, help="path to data config file")
parser.add_argument("--model_config", default="./config/model/fpn_unet.yaml", type=str, help="path to model config file")
parser.add_argument("--trainer_config", default="./config/train/train.yaml", type=str, help="path to train config file")


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

def main():
    args = parser.parse_args()
    args.test_mode = True
    set_configs(args=args)
    val_loader = get_loader(args)
    if "msd" in args.dataset_name:
        val_loader = get_loader_msd_pancreas(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

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
    elif args.model_name == "unet": # Other comparison models
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

    model_dict = torch.load(pretrained_pth)['state_dict']
    # print(model_dict['state_dict'])
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    model.to(device)

    if args.dataset_name == 'synapse':
        dice_1_list = []
        dice_2_list = []
        dice_3_list = []
        dice_4_list = []
        dice_5_list = []
        dice_6_list = []
        dice_7_list = []
        dice_8_list = []
        with torch.no_grad():
            dice_list_case = []
            hd_list_case = []
            start_time = time.time()
            for i, batch in enumerate(tqdm(val_loader)):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                # print("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap)
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, args.out_channels):
                    organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)*100
                    organ_HD = hd(val_outputs[0] == i, val_labels[0] == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_HD)
                    if i == 1:
                        dice_1_list.append(organ_Dice)
                    elif i == 2:
                        dice_2_list.append(organ_Dice)
                    elif i == 3:
                        dice_3_list.append(organ_Dice)
                    elif i == 4:
                        dice_4_list.append(organ_Dice)
                    elif i == 5:
                        dice_5_list.append(organ_Dice)
                    elif i == 6:
                        dice_6_list.append(organ_Dice)
                    elif i == 7:
                        dice_7_list.append(organ_Dice)
                    elif i == 8:
                        dice_8_list.append(organ_Dice)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)
                dice_list_case.append(mean_dice)
                hd_list_case.append(mean_hd)
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)),\
                "Overall Mean hd: {}".format(np.mean(hd_list_case)))
            dice_list = np.array([np.mean(dice_1_list), np.mean(dice_2_list), np.mean(dice_3_list), np.mean(dice_4_list),
                        np.mean(dice_5_list), np.mean(dice_6_list), np.mean(dice_7_list), np.mean(dice_8_list),
                        np.mean(dice_list_case), np.mean(hd_list_case)])
            columns_dice = ["spleen", "right kidney", "left kidney", "gallbladder", 
                            "esophagus", "liver", "stomach", "aorta",
                            "mean organ dice", "mean organ hd"
                            ]
            df_dice = pd.DataFrame(dice_list.reshape(1, -1), columns=columns_dice)
            # current_file_name = args.dataset_name+"_"+args.model_name+".csv"
            # np.savetxt(current_file_name, dice_list, delimiter=",")
            if "em-net" in args.model_name:
                save_root = os.path.join("evaluation", args.dataset_name + '_' + args.variant + '-' + args.model_name)
            else:
                save_root = os.path.join("evaluation", args.dataset_name +'_'+ args.model_name)
            os.makedirs(save_root, exist_ok=True)
            df_dice.to_csv(os.path.join(save_root, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_dice.csv"), index=False)
    elif args.dataset_name == 'btcv' or args.dataset_name == 'flare':
        dice_1_list = []
        dice_2_list = []
        dice_3_list = []
        dice_4_list = []
        dice_5_list = []
        dice_6_list = []
        dice_7_list = []
        dice_8_list = []
        dice_9_list = []
        dice_10_list = []
        dice_11_list = []
        dice_12_list = []
        dice_13_list = []
        with torch.no_grad():
            dice_list_case = []
            hd_list_case = []
            for i, batch in enumerate(tqdm(val_loader)):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                # print("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap)
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, args.out_channels):
                    organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)*100
                    organ_HD = hd(val_outputs[0] == i, val_labels[0] == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_HD)
                    if i == 1:
                        dice_1_list.append(organ_Dice)
                    elif i == 2:
                        dice_2_list.append(organ_Dice)
                    elif i == 3:
                        dice_3_list.append(organ_Dice)
                    elif i == 4:
                        dice_4_list.append(organ_Dice)
                    elif i == 5:
                        dice_5_list.append(organ_Dice)
                    elif i == 6:
                        dice_6_list.append(organ_Dice)
                    elif i == 7:
                        dice_7_list.append(organ_Dice)
                    elif i == 8:
                        dice_8_list.append(organ_Dice)
                    elif i == 9:
                        dice_9_list.append(organ_Dice)
                    elif i == 10:
                        dice_10_list.append(organ_Dice)
                    elif i == 11:
                        dice_11_list.append(organ_Dice)
                    elif i == 12:
                        dice_12_list.append(organ_Dice)
                    elif i == 13:
                        dice_13_list.append(organ_Dice)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)
                dice_list_case.append(mean_dice)
                hd_list_case.append(mean_hd)
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)),\
                "Overall Mean hd: {}".format(np.mean(hd_list_case)))
            dice_list = np.array([np.mean(dice_1_list), np.mean(dice_2_list), np.mean(dice_3_list), np.mean(dice_4_list),
                        np.mean(dice_5_list), np.mean(dice_6_list), np.mean(dice_7_list), np.mean(dice_8_list),
                        np.mean(dice_9_list), np.mean(dice_10_list), np.mean(dice_11_list), np.mean(dice_12_list),
                        np.mean(dice_13_list), np.mean(dice_list_case), np.mean(hd_list_case)])
            columns_dice = ["spleen", "right kidney", "left kidney", "gallbladder", 
                            "esophagus", "liver", "stomach", "aorta",
                            "inferior vena cava", "portal vein & splenic vein",
                            "pancreas", "right adrenal gland", "left adrenal gland",
                            "mean organ dice", "mean organ hd"
                            ]
            df_dice = pd.DataFrame(dice_list.reshape(1, -1), columns=columns_dice)
            # current_file_name = args.dataset_name+"_"+args.model_name+".csv"
            # np.savetxt(current_file_name, dice_list, delimiter=",")
            if "em-net" in args.model_name:
                save_root = os.path.join("evaluation", args.dataset_name + '_' + args.variant + '-' + args.model_name)
            else:
                save_root = os.path.join("evaluation", args.dataset_name +'_'+ args.model_name)
            os.makedirs(save_root, exist_ok=True)
            df_dice.to_csv(os.path.join(save_root, f"{os.path.basename(pretrained_dir)}_dice.csv"), index=False)
    elif args.dataset_name == 'msd_pancreas':
        dice_1_list = []
        dice_2_list = []
        with torch.no_grad():
            dice_list_case = []
            hd_list_case = []
            for i, batch in enumerate(tqdm(val_loader)):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                # print("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap)
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, args.out_channels):
                    organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)*100
                    organ_HD = hd(val_outputs[0] == i, val_labels[0] == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_HD)
                    if i == 1:
                        dice_1_list.append(organ_Dice)
                    elif i == 2:
                        dice_2_list.append(organ_Dice)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)
                dice_list_case.append(mean_dice)
                hd_list_case.append(mean_hd)
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)),\
                "Overall Mean hd: {}".format(np.mean(hd_list_case)))
            dice_list = np.array([np.mean(dice_1_list), np.mean(dice_2_list),
                        np.mean(dice_list_case), np.mean(hd_list_case)])
            columns_dice = ["pancreas", "cancer", 
                            "mean organ dice", "mean organ hd"
                            ]
            df_dice = pd.DataFrame(dice_list.reshape(1, -1), columns=columns_dice)
            # current_file_name = args.dataset_name+"_"+args.model_name+".csv"
            # np.savetxt(current_file_name, dice_list, delimiter=",")
            if "em-net" in args.model_name:
                save_root = os.path.join("evaluation", args.dataset_name + '_' + args.variant + '-' + args.model_name)
            else:
                save_root = os.path.join("evaluation", args.dataset_name +'_'+ args.model_name)
            os.makedirs(save_root, exist_ok=True)
            df_dice.to_csv(os.path.join(save_root, f"{os.path.basename(pretrained_dir)}_dice.csv"), index=False)
    elif args.dataset_name == 'msd_spleen':
        # dice_1_list = []
        dice_2_list = []
        with torch.no_grad():
            dice_list_case = []
            hd_list_case = []
            for i, batch in enumerate(tqdm(val_loader)):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                # print("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap)
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, args.out_channels):
                    organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)*100
                    organ_HD = hd(val_outputs[0] == i, val_labels[0] == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_HD)
                    # if i == 1:
                    #     dice_1_list.append(organ_Dice)
                    if i == 2:
                        dice_2_list.append(organ_Dice)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)
                dice_list_case.append(mean_dice)
                hd_list_case.append(mean_hd)
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)),\
                "Overall Mean hd: {}".format(np.mean(hd_list_case)))
            dice_list = np.array([np.mean(dice_2_list),
                        np.mean(dice_list_case), np.mean(hd_list_case)])
            columns_dice = ["spleen", "overall", "hd"]
            df_dice = pd.DataFrame(dice_list.reshape(1, -1), columns=columns_dice)
            # current_file_name = args.dataset_name+"_"+args.model_name+".csv"
            # np.savetxt(current_file_name, dice_list, delimiter=",")
            if "em-net" in args.model_name:
                save_root = os.path.join("evaluation", args.dataset_name + '_' + args.variant + '-' + args.model_name)
            else:
                save_root = os.path.join("evaluation", args.dataset_name +'_'+ args.model_name)
            os.makedirs(save_root, exist_ok=True)
            df_dice.to_csv(os.path.join(save_root, f"{os.path.basename(pretrained_dir)}_dice.csv"), index=False)
    else:
        assert False, "The dataset name should be one of the following: synapse, btcv, msd"

if __name__ == "__main__":
    main()
