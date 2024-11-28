import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, colorstr,
                         increment_path, init_seeds, intersect_dicts,
                         labels_to_class_weights, labels_to_image_weights,
                         methods, one_cycle, strip_optimizer)
from utils.loggers import Loggers
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device

LOGGER.setLevel(logging.INFO)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt(known=False):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_known_args()[0] if known else parser.parse_args()

class Trainer:
    def __init__(self, hyp, opt, device, callbacks):
        self.hyp = hyp
        self.opt = opt
        self.device = device
        self.callbacks = callbacks
        
        # Directories
        self.save_dir = Path(opt.save_dir)
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        
        # Model
        self.model = self.create_model(opt.cfg, nc=self.data['nc'], hyp=hyp)
        self.names = self.model.names
        
        # Optimizer
        self.optimizer = self.build_optimizer(opt.optimizer,
                                           model=self.model,
                                           lr=hyp['lr0'],
                                           momentum=hyp['momentum'],
                                           weight_decay=hyp['weight_decay'])
        
        # EMA
        self.ema = ModelEMA(self.model) if RANK in {-1, 0} else None
        
        # Resume
        self.start_epoch = 0
        if opt.resume:
            self.resume_training()
            
    def train(self):
        """Start training process."""
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        maps = np.zeros(self.nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.epochs)
        stopper = EarlyStopping(patience=self.opt.patience)
        
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                   f'Using {self.train_loader.num_workers} dataloader workers\n'
                   f"Logging results to {colorstr('bold', self.save_dir)}\n"
                   f'Starting training for {self.epochs} epochs...')
        
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.train_epoch(epoch)
            
            # Scheduler
            lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
            scheduler.step()
            
            if RANK in {-1, 0}:
                # mAP
                callbacks.run('on_train_epoch_end', epoch=epoch)
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs)
                
                if not self.opt.notest or final_epoch:  # Calculate mAP
                    results, maps = self.validate()
                
                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                
                # Save model
                if (not self.opt.nosave) or final_epoch:
                    self.save_model(epoch, results, fi)
            
        LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        pbar = enumerate(self.train_loader)
        
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(self.train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        mloss = torch.zeros(3, device=self.device)  # mean losses
        self.optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:
            # Forward
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)
                loss, loss_items = self.compute_loss(pred, targets)
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Optimize
            if i % self.accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)
            
            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                   (f'{epoch}/{self.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                
    def validate(self):
        """Run validation."""
        # Switch to evaluation mode
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for batch_i, (imgs, targets, paths, shapes) in enumerate(self.val_loader):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device)
                
                # Forward pass
                pred = self.model(imgs)
                
                # Compute loss
                loss, loss_items = self.compute_loss(pred, targets)
                
                # NMS
                pred = non_max_suppression(pred, 
                                         self.conf_thres,
                                         self.iou_thres,
                                         multi_label=True)
                
                # Metrics
                for si, pred in enumerate(pred):
                    labels = targets[targets[:, 0] == si, 1:]
                    correct = process_batch(pred, labels)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
                    
        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            results = ap_per_class(*stats)
            
        return results

def main(opt):
    # Check device
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # DDP mode
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        
    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'giou': (1, 0.02, 0.2),  # GIoU loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HS
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
        }

        evolve_yaml, evolve_csv = str(self.save_dir / 'hyp_evolve.yaml'), str(self.save_dir / 'evolve.csv')
        
        for _ in range(opt.evolve):  # generations to evolve
            if opt.bucket:
                os.system(f'gsutil cp gs://{opt.bucket}/evolve.txt .')  # download evolve.txt if exists
                
            # Fitness
            gen = 1000  # generations to evolve
            evolve_csv = Path(evolve_csv)
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                yoga = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                
                # Get best row
                fitness = yoga[:, :4].mean(1)
                i = fitness.argmax()
                if parent == 'single' or len(yoga) == 1:
                    # x = yoga[i]  # best single experiment
                    file = self.wdir / 'best_single.pt'
                else:  # weighted combination
                    # x = yoga[fitness.argsort()[-n:]]  # best n mutations
                    file = self.wdir / 'best_weighted.pt'
                    
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7])
                    
            # Evolve
            self.evolve_hyperparameters(meta)
            
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                   'val/obj_loss', 'val/cls_loss')
            print_mutation(self.evolve_yaml, self.evolve_csv, keys)

    def evolve_hyperparameters(self, meta):
        """
        Evolve hyperparameters according to meta settings.
        """
        n = 8  # number of mutations eg 8 with 50% probability of success (.5**8 = 0.004)
        mutation_num = meta['mutation_num']
        
        for _ in range(n):
            noise_scale = meta['noise_scale']
            noise = np.random.normal(0, noise_scale, size=mutation_num)
            hyp = self.hyp.copy()
            
            # Mutate
            for i, k in enumerate(hyp.keys()):
                if meta[k][0]:  # if meta data is available
                    x = (np.clip(float(hyp[k]) + noise[i], meta[k][1], meta[k][2])).astype(np.float32)
                    hyp[k] = x
                    
            # Train mutation
            results = self.train(hyp)
            
            # Write result
            self.write_evolve_results(results, hyp)
            
    def write_evolve_results(self, results, hyp):
        """Write evolution results to evolve.txt."""
        with open('evolve.txt', 'a') as f:
            params = tuple(x for x in hyp.values())
            f.write(('%10.3g,' * len(results) + '%10.3g,' * len(params) + '\n') % (results + params))

    @staticmethod
    def get_optimizer(opt_name: str, model: nn.Module, lr: float = 0.01,
                     momentum: float = 0.937, weight_decay: float = 0.0005):
        """
        Build optimizer based on opt_name.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)
                
        if opt_name == 'Adam':
            optimizer = Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif opt_name == 'AdamW':
            optimizer = AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif opt_name == 'SGD':
            optimizer = SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {opt_name} not implemented.')
            
        optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        
        return optimizer

if __name__ == "__main__":
    opt = parse_opt()
    opt.world_size = int(os.environ.get("WORLD_SIZE", 1))
    opt.global_rank = int(os.environ.get("RANK", -1))
    
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        
    # Train
    if not opt.evolve:
        train = Train(opt.hyp, opt, device)
        train.train()
