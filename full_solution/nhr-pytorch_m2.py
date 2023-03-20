#!/usr/bin/env python
# coding: utf-8

import glob, os, pickle, random, shutil, time, gc
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import getpass
import rasterio as rio

from tqdm import tqdm
from pathlib import Path
from radiant_mlhub import Dataset as rmd
from random import choice
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from typing import List, Any, Callable, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import ttach as tta
import albumentations as A
import segmentation_models_pytorch as smp

IMG_WIDTH = 256 
IMG_HEIGHT = 256 
IMG_CHANNELS = 4
BATCH_SIZE = 4
SEED = 2023
is_train = True
n_accumulate = 1
EPOCH = 100
n_folds = 7
lr = 2e-3
num_classes = 1
MONTHS = ['2021_03', '2021_04', '2021_08', '2021_10', '2021_11', '2021_12']
gpus = '0'
SCHEDULER = 'CosineAnnealingWarmRestarts'
decoder = 'UnetPlusPlus'
encoder = 'timm-efficientnet-l2'
OUTPUT_DIR = './7folds_m2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_id = 'nasa_rwanda_field_boundary_competition'
archives = ['source_train', 'source_test', 'labels_train']
train_source_items = f"{dataset_id}/{dataset_id}_source_train"
train_label_items = f"{dataset_id}/{dataset_id}_labels_train"


# os.environ['MLHUB_API_KEY'] = getpass.getpass(prompt="MLHub API Key: ")
# dataset = rmd.fetch(dataset_id)
# dataset.download(output_dir = dataset_id, if_exists='overwrite')

# for archive in archives:
#     full_path = f"{dataset_id}/{dataset_id}_{archive}.tar.gz"
#     shutil.unpack_archive(full_path, dataset_id)

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize(array: np.ndarray):
    """ normalise image to give a meaningful output """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def clean_string(s: str) -> str:
    """
    extract the tile id and timestamp from a source image folder
    e.g extract 'ID_YYYY_MM' from 'nasa_rwanda_field_boundary_competition_source_train_ID_YYYY_MM'
    """
    s = s.replace(f"{dataset_id}_source_", '').split('_')[1:]
    return '_'.join(s)

set_seed(SEED)

train_tiles = [clean_string(s) for s in next(os.walk(train_source_items))[1] if 'source_train' in s]
train_tile_ids = []
for tile in train_tiles:
    train_tile_ids.append(tile.split('_')[0])
train_tile_ids = sorted(set(train_tile_ids))
next(os.walk(train_source_items))[1][0]


# Datasets
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, tiles, label=True, transforms=None):
        self.label = label
        self.tiles = tiles
        self.transforms = transforms

    def __len__(self):
        return len(self.tiles)
    
    def get_image(self, tile_id):
        X = np.empty((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*len(MONTHS)), dtype=np.float32)
        idx = 0
        source = train_source_items if self.label else test_source_items
        txt = 'train' if self.label else 'test'
        months = MONTHS.copy()
        for month in months:
            bd1 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B01.tif")
            bd1_array = bd1.read(1)
            bd2 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B02.tif")
            bd2_array = bd2.read(1)
            bd3 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B03.tif")
            bd3_array = bd3.read(1)
            bd4 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B04.tif")
            bd4_array = bd4.read(1)
            b01_norm = normalize(bd1_array)
            b02_norm = normalize(bd2_array)
            b03_norm = normalize(bd3_array)
            b04_norm = normalize(bd4_array)

            field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
            X[:,:,idx:idx+IMG_CHANNELS] = field
            idx+=IMG_CHANNELS
        return X
    
    def __getitem__(self, index):
        img = self.get_image(self.tiles[index])
        if self.label:
            msk = rio.open(Path.cwd() / f"{train_label_items}/{dataset_id}_labels_train_{self.tiles[index]}/raster_labels.tif").read(1)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img, msk = data['image'], data['mask']
            return torch.tensor(np.transpose(img, (2, 0, 1))), str(self.tiles[index]).zfill(2), torch.tensor(np.transpose(np.expand_dims(msk, axis=2), (2, 0, 1)))
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            return torch.tensor(np.transpose(img, (2, 0, 1))), str(self.tiles[index]).zfill(2)

data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], p=1.0),
    "valid": A.Compose([
    ], p=1.0)
}

BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

def f1_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold)*1.0
    prec = (y_pred*y_true).sum()/(1e-6 + y_pred.sum())
    rec = (y_pred*y_true).sum()/(1e-6 + y_true.sum())
    f1 = 2*prec*rec/(1e-6 + prec + rec)
    return f1

# model
def build_model(encoder, decoder):
    model = smp.UnetPlusPlus(
        encoder_name=encoder, 
        encoder_weights='noisy-student',
        in_channels=IMG_CHANNELS*len(MONTHS),
        classes=num_classes,
        activation=None,
        decoder_attention_type='scse'
    )

    if len(gpus.split(',')) > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model

def load_model(encoder, decoder, path):
    model = build_model(encoder, decoder)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, tiles, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}')
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    val_f1s = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, tiles, masks) in pbar:
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_f1s.append(f1_score(y_true=masks.cpu().detach().numpy(), y_pred=y_pred.cpu().detach().numpy()))
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',)
    val_f1  = np.mean(val_f1s, axis=0)
    
    return epoch_loss, val_f1

def get_loaders(
    train_ids, val_ids,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transforms_fn = None,
    valid_transforms_fn = None,
) -> dict:
    train_dataset = BuildDataset(tiles=train_ids, transforms=train_transforms_fn)
    valid_dataset = BuildDataset(tiles=val_ids, transforms=valid_transforms_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size*2, 
                              num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, valid_loader

def run_training(model, train_loader, valid_loader, device, fold, OUTPUT_DIR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=EPOCH, eta_min=1e-6)
    best_dice      = -np.inf
    best_f1        = -np.inf
    best_epoch     = -1

    for epoch in range(1, EPOCH + 1): 
        print(f'Epoch {epoch}/{EPOCH}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)

        val_loss, val_f1 = valid_one_epoch(model, valid_loader, device=device, epoch=epoch)
        scheduler.step()
        print(f'Valid Loss: {val_loss:0.4f} | Valid F1: {val_f1:0.4f}')
        if val_f1 >= best_f1:
            print(f"Valid F1 Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1    = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'fold{fold}_f1_best.pth'))
    return best_f1

import random
my_seeded_random = random.Random(SEED)

nb_rows = len(train_tile_ids)
index_all = list(range(nb_rows))
my_seeded_random.shuffle(index_all)
fold_size = nb_rows // n_folds

dict_folds = {}
for fold in range(n_folds):
    if fold == 0:
        index_val = index_all[:fold_size]
        index_train = index_all[fold_size:]
    elif fold == (n_folds - 1):
        index_val = index_all[fold_size*(n_folds-1)+1:]
        index_train = index_all[:fold_size*(n_folds-1)+1]
    else:
        index_val = index_all[fold_size*fold:fold_size*(fold+1)]
        index_train = index_all[:fold_size*fold] + index_all[fold_size*(fold+1):]
        
    dict_folds[fold] = (index_train, index_val)
    print(fold, len(index_train), len(index_val))


fold_score={}
fold_score[f'{encoder}_{decoder}'] = []
for fold in range(n_folds):
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)
    (index_train, index_val) = dict_folds[fold]
    fold_train_tile_ids = [train_tile_ids[i] for i in index_train]
    fold_val_tile_ids = [train_tile_ids[i] for i in index_val]
    train_loader, valid_loader = get_loaders(fold_train_tile_ids, fold_val_tile_ids, BATCH_SIZE, 4, data_transforms['train'], data_transforms['valid'])
    comment = f'{decoder}_{encoder}'
    model_path = os.path.join(OUTPUT_DIR, comment)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model = build_model(encoder=encoder, decoder=decoder)
    best_f1 = run_training(model, train_loader, valid_loader, device, fold, model_path)
    fold_score[f'{encoder}_{decoder}'].append({f'{fold}fold' : best_f1})

test_source_items = f"{dataset_id}/{dataset_id}_source_test"
test_tiles = [clean_string(s) for s in next(os.walk(test_source_items))[1] if 'source_test' in s]

test_tile_ids = []
for tile in test_tiles:
    test_tile_ids.append(tile.split('_')[0])
test_tile_ids = sorted(set(test_tile_ids))

test_dataset = BuildDataset(tiles = test_tile_ids, label=False, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

predictions_dictionary = {}
pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test ')
for step, (images, tiles) in pbar:
    model_preds = []
    with torch.no_grad():
        for fold in range(n_folds):
            model = load_model(encoder, decoder, f'{model_path}/fold{fold}_f1_best.pth')
            tta_model = tta.SegmentationTTAWrapper(model, tta.Compose([
                tta.HorizontalFlip(), tta.VerticalFlip()
            ]), merge_mode="mean")
            model_pred = model(images.to(device, dtype=torch.float))
            model_pred = model_pred.sigmoid()
            model_pred = 0.7*model_pred + 0.3*tta_model(images.to(device, dtype=torch.float)).sigmoid()
            model_pred = model_pred.detach().cpu().numpy()
            model_preds.append(model_pred)
    model_pred = np.vstack(model_preds).mean(axis=0).squeeze()
    predictions_dictionary.update([(str(tiles[0]), pd.DataFrame(model_pred))])

dfs = []
for key, value in predictions_dictionary.items():
    ftd = value.unstack().reset_index().rename(columns={'level_0': 'row', 'level_1': 'column', 0: 'label'})
    ftd['tile_row_column'] = f'Tile{key}_' + ftd['row'].astype(str) + '_' + ftd['column'].astype(str)
    ftd = ftd[['tile_row_column', 'label']]
    dfs.append(ftd)

sub = pd.concat(dfs)
sub.to_csv(f"{model_path}/harvest_sample_submission_prob.csv", index=False)

