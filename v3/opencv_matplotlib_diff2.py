import os
import sys
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 버전문제로 모든 파일을 다운 후 설치 필요
sys.path.insert(0, '/home/kerrykim/PycharmProjects/3.road_obstacle_detection/effdet')
sys.path.insert(0, '/home/kerrykim/PycharmProjects/3.road_obstacle_detection/omegaconf')

# pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

# augmentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# sci-kit learn
from sklearn.model_selection import StratifiedKFold

# etc
import time

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

from dataset import *
from util import *
from main import *

import warnings
warnings.filterwarnings("ignore")

LOGGER = init_logger(CFG.log_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from apex import amp

##
def transform_train():
    return A.Compose([A.RandomSizedCrop(min_max_height=(768, 768), height=1024, width=1024, p=0.5),
            A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))

def transform_val():
    return A.Compose([A.Resize(height=512, width=512, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))


##
def train_one_epoch(loader_train, net, optim, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()

    net.train()
    start = end = time.time()

    for batch, (images, targets) in enumerate(loader_train, 1):
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]

        optim.zero_grad()

        loss, _, _ = net(images, boxes, labels)

        # record loss
        batch_size = images.shape[0]
        losses.update(loss.detach().item(), batch_size)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=CFG.max_grad_norm)  # clipping grad

        optim.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % CFG.print_freq == 0 or batch == len(loader_train):
            print('Epoch {0}: [{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} (avg {loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                .format(
                epoch, batch, len(loader_train), batch_time=batch_time, loss=losses,
                remain=timeSince(start, float(batch) / len(loader_train)),
                grad_norm=grad_norm,
            ))

    return losses.avg


##
def val_one_epoch(loader_val, net, device):
    batch_time = AverageMeter()
    losses = AverageMeter()

    net.eval()
    start = end = time.time()

    for batch, (images, targets) in enumerate(loader_val, 1):
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]

        with torch.no_grad():
            loss, _, _ = net(images, boxes, labels)

        batch_size = images.shape[0]
        losses.update(loss.detach().item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % CFG.print_freq == 0 or batch == (len(loader_val)):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} (avg {loss.avg:.4f}) '
                .format(
                batch, len(loader_val), batch_time=batch_time,
                loss=losses,
                remain=timeSince(start, float(batch) / len(loader_val)),
            ))

    return losses.avg


def collate_fn(batch):
    return tuple(zip(*batch))


##
def fit(df, df_kfold):
    skf = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)

    KFOLD = [(idxT,idxV) for i,(idxT,idxV) in enumerate(skf.split(np.arange(df_kfold.shape[0]), df_kfold['stratify_group']))]

    for fold, (idxT, idxV) in enumerate(KFOLD, 1):
        LOGGER.info(f"Training starts ... KFOLD: {fold}/{CFG.num_fold}")

        train = df.loc[idxT, :].reset_index(drop=True)
        val = df.loc[idxV, :].reset_index(drop=True)

        dataset_train = Dataset(df=train, data_dir=CFG.data_dir, transform=transform_train())
        loader_train = DataLoader(dataset_train, batch_size=CFG.batch_size,
                                  sampler=RandomSampler(dataset_train), pin_memory=False, drop_last=True,
                                  num_workers=CFG.num_workers, collate_fn=collate_fn)

        '''
        # save tain sample 
        images, targets = next(iter(loader_train))

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
        sample = images[0].permute(1, 2, 0).cpu().numpy()

        # effdet is yxyx format
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

        for box in boxes:
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255), 2)

        # ax.set_axis_off()
        sample = sample*255
        cv2.imwrite('b.jpg', sample)
        break
        '''


        dataset_val = Dataset(df=val, data_dir=CFG.data_dir, transform=transform_val())
        loader_val = DataLoader(dataset_val, batch_size=CFG.batch_size,
                                sampler=SequentialSampler(dataset_val), pin_memory=False, drop_last=True,
                                num_workers=CFG.num_workers, collate_fn=collate_fn)

        config = get_efficientdet_config('tf_efficientdet_d5')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('/home/kerrykim/PycharmProjects/3.pothhole_detection/pretrained/efficientdet_d5-ef44aea8.pth')

        # the pretrained weights need to be loaded first,
        # before you modify the head, or you need to modify the state dict to exclude the old head tensors if.
        # you want to do it after.
        config.num_classes = 1
        config.image_size = 512
        net.load_state_dict(checkpoint)
        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

        net = (DetBenchTrain(net, config)).to(device)

        optim = torch.optim.AdamW(net.parameters(), lr=CFG.lr)

        # if args.use_amp:
        #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        scheduler = ReduceLROnPlateau(optim, **CFG.scheduler_params)

        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-1)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=args.warmup_factor, total_epoch=1, after_scheduler=scheduler_cosine)


        # default value
        st_epoch = 0
        best_loss = 1e20

        for epoch in range(st_epoch + 1, CFG.num_epoch + 1):
            # if epoch < 5:
            #     continue
            start_time = time.time()

            # train
            avg_train_loss = train_one_epoch(loader_train, net, optim, epoch, device)

            # val
            avg_val_loss = val_one_epoch(loader_val, net, device)

            scheduler.step(metrics=avg_val_loss)

            # scoring
            elapsed = time.time() - start_time

            LOGGER.info(
                f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

            # save best model
            save_argument = best_loss > avg_val_loss
            best_loss = min(avg_val_loss, best_loss)

            LOGGER.info(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')

            save_model(ckpt_dir=CFG.ckpt_dir, net=net, num_epoch=CFG.num_epoch, fold=fold,
                       epoch=epoch, batch=CFG.batch_size, save_argument=save_argument)












