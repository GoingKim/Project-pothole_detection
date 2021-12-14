import os
import sys
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from dataset import *
from util import *
from main import *
from model import *
from metric import *

import warnings
warnings.filterwarnings("ignore")

LOGGER = init_logger(CFG.log_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



##
def transform_train():
    return A.Compose([A.OneOf([A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))

def transform_val():
    return A.Compose([A.Resize(height=512, width=512, p=1.0), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))


##
def train_one_epoch(loader_train, net, optim, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()

    net.train()
    start = end = time.time()

    for batch, (images, targets, image_ids) in enumerate(loader_train, 1):
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        img_scale = torch.tensor([target['img_scale'].to(device).float() for target in targets])
        img_size = torch.tensor([(512, 512) for _ in targets]).to(device).float()

        target_res = {}
        target_res['bbox'] = boxes
        target_res['cls'] = labels
        target_res['img_scale'] = img_scale
        target_res['img_size'] = img_size

        optim.zero_grad()

        outputs = net(images, target_res)

        loss = outputs['loss']
        loss.backward()

        # record loss
        batch_size = images.shape[0]
        losses.update(loss.detach().item(), batch_size)

        optim.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % CFG.print_freq == 0 or batch == len(loader_train):
            print('Epoch {0}: [{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} (avg {loss.avg:.4f}) '
                .format(
                epoch, batch, len(loader_train), batch_time=batch_time, loss=losses,
                remain=timeSince(start, float(batch) / len(loader_train)),
            ))

    return losses.avg


##
def val_one_epoch(loader_val, net, device):
    batch_time = AverageMeter()
    losses = AverageMeter()

    net.eval()
    start = end = time.time()

    precision_val = []
    for batch, (images, targets, iamge_ids) in enumerate(loader_val, 1):
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        img_scale = torch.tensor([target['img_scale'].to(device).float() for target in targets])
        img_size = torch.tensor([(512, 512) for _ in targets]).to(device).float()

        target_res = {}
        target_res['bbox'] = boxes
        target_res['cls'] = labels
        target_res['img_scale'] = img_scale
        target_res['img_size'] = img_size

        with torch.no_grad():
            outputs = net(images, target_res)
            loss = outputs['loss']    # outputs have (total) loss, class loss and box loss.

            temp = net(images, target_res)["detections"]
            for i in range

            # check IOU
            det = net(images, target_res)["detections"]
            gt_boxes = [target['boxes'][0].detach().cpu().numpy() for target in targets]
            for i in range(images.shape[0]):
                pred_boxes = det[i].detach().cpu().numpy()[:,:4]
                scores = det[i].detach().cpu().numpy()[:,4]

            sorted_idx = np.argsort(scores)[::-1]
            preds_sorted_boxes = pred_boxes[sorted_idx]

        for idx, image in enumerate(images):
            # boxes result is coco metric
            precision = calculate_image_precision(preds_sorted_boxes, gt_boxes, thresholds=iou_thresholds, form='coco')
            precision_val.append(precision)

        batch_size = images.shape[0]
        losses.update(loss.detach().item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % CFG.print_freq == 0 or batch == (len(loader_val)):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} (avg {loss.avg:.4f}) '
                  'AP: {ap:.4f}'
                .format(
                batch, len(loader_val), batch_time=batch_time,
                loss=losses,
                remain=timeSince(start, float(batch) / len(loader_val)),
                ap=np.mean(precision_val)
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

        dataset_train = Dataset(df=train, img_size=CFG.img_size, transform=transform_train())
        loader_train = DataLoader(dataset_train, batch_size=CFG.batch_size,
                                  pin_memory=True, drop_last=True, shuffle=True,
                                  num_workers=CFG.num_workers, collate_fn=collate_fn)

        '''
        # save tain sample 
        images, targets, image_id = next(iter(loader_train))

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
        sample = sample
        cv2.imwrite('transformed_image_with_rec.jpg', sample)
        break
        '''

        dataset_val = Dataset(df=val, img_size=CFG.img_size, transform=transform_val())
        loader_val = DataLoader(dataset_val, batch_size=CFG.batch_size,
                                pin_memory=True, drop_last=True, shuffle=False,
                                num_workers=CFG.num_workers, collate_fn=collate_fn)

        # net = EfficientDetModel(num_classes=1, img_size=512).to(device)
        net = create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l")
        checkpoint = torch.load('./pretrained/tf_efficientnetv2_l-d664b728.pth')
        net.load_state_dict(checkpoint, strict=False)
        net = net.to(device)

        optim = torch.optim.AdamW(net.parameters(), lr=CFG.lr)
        # optim = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False)  # 1e-6

        scheduler = ReduceLROnPlateau(optim, **CFG.scheduler_params)

        # default value
        st_epoch = 0
        best_loss = 1e20

        for epoch in range(st_epoch + 1, CFG.num_epoch + 1):
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












