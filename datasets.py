import os
import glob
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance


def augment(img_pil, boxes):
    if random.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        boxes   = [[c, 1.0 - cx, cy, w, h] for c, cx, cy, w, h in boxes]
    img_pil = ImageEnhance.Color(img_pil).enhance(random.uniform(0.5, 1.5))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.5, 1.5))
    return img_pil, boxes


def collate_fn(batch):
    imgs, boxes = zip(*batch)
    return torch.stack(imgs), list(boxes)


class VOCDataset(Dataset):
    def __init__(self, roots_splits, cls2idx, size=416, augment_data=False, cache=False):
        self.size         = size
        self.augment_data = augment_data
        self.cls2idx      = cls2idx
        self.items        = []

        for voc_root, split in roots_splits:
            split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
            ids = [l.strip() for l in open(split_file) if l.strip()]
            for id_ in ids:
                self.items.append((
                    os.path.join(voc_root, 'JPEGImages',  f'{id_}.jpg'),
                    os.path.join(voc_root, 'Annotations', f'{id_}.xml'),
                ))

        self._cache = {}
        if cache:
            print(f'  caching {len(self.items)} VOC images into RAM...', flush=True)
            for img_path, _ in self.items:
                self._cache[img_path] = Image.open(img_path).convert('RGB')
            print('  done', flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ann_path = self.items[idx]

        img = (self._cache[img_path].copy()
               if img_path in self._cache
               else Image.open(img_path).convert('RGB'))
        img = img.resize((self.size, self.size))

        root = ET.parse(ann_path).getroot()
        W    = float(root.findtext('size/width'))
        H    = float(root.findtext('size/height'))
        boxes = []
        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in self.cls2idx:
                continue
            bb  = obj.find('bndbox')
            x1  = float(bb.findtext('xmin')); y1 = float(bb.findtext('ymin'))
            x2  = float(bb.findtext('xmax')); y2 = float(bb.findtext('ymax'))
            boxes.append([self.cls2idx[name],
                          (x1+x2)/2/W, (y1+y2)/2/H,
                          (x2-x1)/W,   (y2-y1)/H])

        if self.augment_data and boxes:
            img, boxes = augment(img, boxes)

        img   = torch.from_numpy(np.array(img)).float().div(255).permute(2, 0, 1)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return img, boxes


class COCODataset(Dataset):
    def __init__(self, img_dir, lbl_dir, size=416, augment_data=False):
        self.size         = size
        self.augment_data = augment_data
        self.imgs         = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.lbl_dir      = lbl_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB').resize((self.size, self.size))
        lbl = os.path.join(self.lbl_dir, Path(self.imgs[idx]).stem + '.txt')
        boxes = []
        if os.path.exists(lbl):
            for line in open(lbl):
                boxes.append(list(map(float, line.strip().split())))
        if self.augment_data and boxes:
            img, boxes = augment(img, boxes)
        img   = torch.from_numpy(np.array(img)).float().div(255).permute(2, 0, 1)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return img, boxes


def make_loaders(train_ds, val_ds, batch_size, num_workers):
    kw = dict(
        collate_fn              = collate_fn,
        num_workers             = num_workers,
        persistent_workers      = True,
        prefetch_factor         = 4,
        pin_memory              = False,   # no effect on MPS unified memory
        multiprocessing_context = 'fork',
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    return train_dl, val_dl
