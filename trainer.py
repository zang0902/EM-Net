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

import os
import shutil
import time
import yaml

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather
import logging
from monai.data import decollate_batch

torch.backends.cudnn.benchmark = False

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    # start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        start_time = time.time()
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            if args.deep_supervision:
                multi_target = [target]
                for i in range(1, len(logits)):
                    logits[i] = torch.nn.Upsample(scale_factor=2**(i+1))(logits[i])
                    multi_target.append(target)
                loss = loss_func(logits, multi_target)
            else:
                loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        # if args.rank == 0:
        #     print(
        #         "iter time {:.2f}s".format(time.time() - start_time),
        #     )
        # start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def train_epoch_by_iter(model, loader, optimizer, scaler, epoch, loss_func, args, max_iter=250):
    model.train()
    # start_time = time.time()
    run_loss = AverageMeter()
    # for idx, batch_data in enumerate(loader):
    loader_iter = iter(loader)
    for idx in range(max_iter):
        start_time = time.time()
        try:
            batch_data = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_data = next(loader_iter)

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            if args.deep_supervision:
                if args.model_name == "umamba_bot" or args.model_name == 'umamba_enc':
                    multi_target = [target]
                    for i in range(1, len(logits)):
                        logits[i] = torch.nn.Upsample(scale_factor=2**i)(logits[i])
                        multi_target.append(target)
                    loss = loss_func(logits, multi_target)
                else:
                    multi_target = [target]
                    for i in range(1, len(logits)):
                        logits[i] = torch.nn.Upsample(scale_factor=2**(i+1))(logits[i])
                        multi_target.append(target)
                    loss = loss_func(logits, multi_target)
            else:
                loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        # if args.rank == 0:
        #     print(
        #         "iter time {:.2f}s".format(time.time() - start_time),
        #     )
        # start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, logger=None):
    model.eval()
    all_acc_list = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if args.deep_supervision:
                logits = logits[0]
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            else:
                acc_list = acc.detach().cpu().numpy()                
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                all_acc_list.append(avg_acc)

    avg_acc = np.mean(all_acc_list)
    return avg_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)  # Tensorboard
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
        log_name = 'training_log_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        log_file_path = os.path.join(args.logdir, log_name)

        # Create logger
        logger = logging.getLogger('training_logger')
        logger.setLevel(logging.DEBUG)

        # File
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s][line: %(lineno)d] > %(message)s')  # Precise time
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Training settings: \n \
                    Model: {args.model_name} \n \
                    Dataset: {args.dataset_name} \n \
                    Seed: {args.seed} \n \
                    Category Numbers: {args.out_channels} \n \
                    Batch size: {args.batch_size} \n \
                    Learning rate: {args.optim_lr:.2e} \n \
                    Max epochs: {args.max_epochs} \n \
                    Optimizer: {args.optim_name} \n \
                    Regularization: {args.reg_weight} \n \
                    AMP: {not args.noamp} \n \
                    Warmup: {args.warmup_epochs} \n \
                    ROI: {args.roi_x, args.roi_y, args.roi_z} \n \
                    Deep supervision: {args.deep_supervision} \n \
                    Crop nums: {args.crop_nums}")
        
        if "em-net" in args.model_name:  # Ours settting
            logger.info(f"EM-Net settings: \n \
                        hidden_size: {args.hidden_size} \n \
                        feature_size: {args.feature_size} \n \
                        depths: {args.depths} \n \
                        fft_nums: {args.fft_nums} \n \
                        conv_decoder: {args.conv_decoder} \n \
                        ")

        logger.info('Training started')

        with open(f'{args.logdir}/args.yaml', 'w') as f:
            yaml.dump(args.__dict__, f)

    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    if args.exp_mode:
        gpu_mem = torch.cuda.memory_summary(device=None, abbreviated=True)
        print(gpu_mem)
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        # print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        if args.train_by_iter:
            # 2. one epoch = 250 iters
            train_loss = train_epoch_by_iter(
                model, train_loader, optimizer, scaler=scaler, 
                epoch=epoch, loss_func=loss_func, args=args, max_iter=args.iter_per_epoch
            )
        else:
            # 1. True epoch
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
            )
        if args.rank == 0:
            logger.info(
                f"Training: {epoch+1}/{args.max_epochs} loss: {train_loss:.4f}, \
                lr: {optimizer.state_dict()['param_groups'][0]['lr']:.2e}, \
                time: {time.time() - epoch_time:.2f}s"
            )
        if args.rank == 0 and args.exp_mode:
            print("Training speed: {} samples/s".format(args.batch_size * args.world_size * args.crop_nums * args.iter_per_epoch / (time.time() - epoch_time)))
            print(f"batch size: {args.batch_size}")
            print(f"crop nums: {args.crop_nums}")
            print(f"world size: {args.world_size}")
            gpu_mem = torch.cuda.memory_summary(device=None, abbreviated=True)
            print(gpu_mem)
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                logger=logger,
            )
            if args.rank == 0:
                # print(
                #     "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                #     "acc",
                #     val_avg_acc,
                #     "time {:.2f}s".format(time.time() - epoch_time),
                # )
                logger.info(
                    f"Validation: {epoch+1}/{args.max_epochs}, Acc: {val_avg_acc:.4f}, Time: {time.time() - epoch_time:.2f}s"
                )
                if args.exp_mode:
                    print("Inference speed: {:.2f}".format(len(val_loader) / (time.time() - epoch_time)))
                    # print("Loader len: {}".format(len(val_loader)))
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    # print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    logger.info(f"new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    # print("Copying to model.pt new best model!!!!")
                    logger.info("Copying to model.pt new best model!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    # print("Training Finished !, Best Accuracy: ", val_acc_max)
    logger.info(f"Training Finished!, Best Accuracy: {val_acc_max}")

    return val_acc_max
