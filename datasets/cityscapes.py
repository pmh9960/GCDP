import json
import os
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cityscapesscripts.helpers.labels import trainId2label
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
# autopep8: off
classes = [
    CityscapesClass('unlabeled',             0, 255, 'void',         0, False, True,  (  0,   0,   0)),
    CityscapesClass('ego vehicle',           1, 255, 'void',         0, False, True,  (  0,   0,   0)),
    CityscapesClass('rectification border',  2, 255, 'void',         0, False, True,  (  0,   0,   0)),
    CityscapesClass('out of roi',            3, 255, 'void',         0, False, True,  (  0,   0,   0)),
    CityscapesClass('static',                4, 255, 'void',         0, False, True,  (  0,   0,   0)),
    CityscapesClass('dynamic',               5, 255, 'void',         0, False, True,  (111,  74,   0)),
    CityscapesClass('ground',                6, 255, 'void',         0, False, True,  ( 81,   0,  81)),
    CityscapesClass('road',                  7,   0, 'flat',         1, False, False, (128,  64, 128)),
    CityscapesClass('sidewalk',              8,   1, 'flat',         1, False, False, (244,  35, 232)),
    CityscapesClass('parking',               9, 255, 'flat',         1, False, True,  (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat',         1, False, True,  (230, 150, 140)),
    CityscapesClass('building',             11,   2, 'construction', 2, False, False, ( 70,  70,  70)),
    CityscapesClass('wall',                 12,   3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13,   4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True,  (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True,  (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True,  (150, 120,  90)),
    CityscapesClass('pole',                 17,   5, 'object',       3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object',       3, False, True,  (153, 153, 153)),
    CityscapesClass('traffic light',        19,   6, 'object',       3, False, False, (250, 170,  30)),
    CityscapesClass('traffic sign',         20,   7, 'object',       3, False, False, (220, 220,   0)),
    CityscapesClass('vegetation',           21,   8, 'nature',       4, False, False, (107, 142,  35)),
    CityscapesClass('terrain',              22,   9, 'nature',       4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23,  10, 'sky',          5, False, False, ( 70, 130, 180)),
    CityscapesClass('person',               24,  11, 'human',        6, True,  False, (220,  20,  60)),
    CityscapesClass('rider',                25,  12, 'human',        6, True,  False, (255,   0,   0)),
    CityscapesClass('car',                  26,  13, 'vehicle',      7, True,  False, (  0,   0, 142)),
    CityscapesClass('truck',                27,  14, 'vehicle',      7, True,  False, (  0,   0,  70)),
    CityscapesClass('bus',                  28,  15, 'vehicle',      7, True,  False, (  0,  60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle',      7, True,  True,  (  0,   0,  90)),
    CityscapesClass('trailer',              30, 255, 'vehicle',      7, True,  True,  (  0,   0, 110)),
    CityscapesClass('train',                31,  16, 'vehicle',      7, True,  False, (  0,  80, 100)),
    CityscapesClass('motorcycle',           32,  17, 'vehicle',      7, True,  False, (  0,   0, 230)),
    CityscapesClass('bicycle',              33,  18, 'vehicle',      7, True,  False, (119,  11,  32)),
    CityscapesClass('license plate',        -1,  -1, 'vehicle',      7, False, True,  (  0,   0, 142)),
]
# autopep8: on

map_id_to_id = torch.tensor([x.id for x in classes])
map_id_to_category_id = torch.tensor([x.category_id for x in classes])
map_id_to_train_id = torch.tensor([x.train_id for x in classes])
id_type_to_classes = dict(
    id=dict(num_classes=34,
            map_fn=torch.tensor([x if x not in (-1, ) else 0 for x in map_id_to_id]),
            names=[cls.name for cls in classes][:-1]),
    category_id=dict(num_classes=8,
                     map_fn=map_id_to_category_id,
                     names=[cls.name for cls in classes][:-1]),  # TODO it is wrong
    train_id=dict(num_classes=20,
                  map_fn=torch.tensor([x if x not in (-1, 255) else 19 for x in map_id_to_train_id]),
                  names=[i.name for i in classes if i.train_id != 255][:-1] + ['unlabeled']),
)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5


def unnormalize_and_clamp_to_zero_to_one(img):
    return torch.clamp(unnormalize_to_zero_to_one(img.cpu()), 0, 1)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ToTensorNoNorm():
    def __call__(self, X_i):
        X_i = np.array(X_i)

        if len(X_i.shape) == 2:
            # Add channel dim.
            X_i = X_i[:, :, None]

        return torch.from_numpy(np.array(X_i, copy=False)).permute(2, 0, 1)


def interpolate_3d(x, *args, **kwargs):
    return F.interpolate(x.unsqueeze(0), *args, **kwargs).squeeze(0)


class RandomResize(nn.Module):
    def __init__(self, scale=(0.5, 2.0), mode='nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def get_random_scale(self):
        return random.uniform(*self.scale)

    def forward(self, x):
        random_scale = self.get_random_scale()
        x = interpolate_3d(x, scale_factor=random_scale, mode=self.mode)
        return x


def read_jsonl(jsonl_path):
    import jsonlines
    lines = []
    with jsonlines.open(jsonl_path, 'r') as f:
        for line in f.iter():
            lines.append(line)
    return lines


class CityscapesDataset(Dataset):
    def __init__(
        self,
        root="",
        split='train',
        side_x=64,
        side_y=64,
        shuffle=False,
        caption_list_dir='',
        id_type='train_id',
        augmentation_type='flip',
    ):
        super().__init__()
        self.root = Path(root)
        self.image_dir = os.path.join(self.root, 'leftImg8bit')
        self.label_dir = os.path.join(self.root, 'gtFine')
        self.split = split
        self.metadata = read_jsonl(os.path.join(caption_list_dir, f'{split}_captions.jsonl'))
        self.metadata = sorted(self.metadata, key=lambda line: line['file_name'])

        assert id_type == 'train_id'
        self.map_fn = id_type_to_classes[id_type]['map_fn']
        self.class_names = id_type_to_classes[id_type]['names']
        self.num_classes = id_type_to_classes[id_type]['num_classes']

        # self.text_ctx_len = text_ctx_len
        self.shuffle = shuffle
        self.side_x = side_x
        self.side_y = side_y

        if augmentation_type == 'none':
            self.augmentation = Compose([
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                # ToTensor(),
            ])
        elif augmentation_type == 'flip':
            self.augmentation = Compose([
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        elif 'resizedCrop' in augmentation_type:
            scale = [float(s) for s in augmentation_type.split('_')[1:]]
            assert len(scale) == 2, scale
            self.augmentation = Compose([
                RandomResize(scale=scale, mode='nearest'),
                RandomCrop((1024, 2048)),
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        else:
            raise NotImplementedError(augmentation_type)

        # filenames of images and labels
        self.images = []
        self.labels = []
        for line in self.metadata:
            cityname = line['file_name'].split('_')[0]
            split = 'val' if cityname in ['frankfurt', 'lindau', 'munster'] else 'train'
            img_dir = os.path.join(self.image_dir, split, cityname, line['file_name'])
            lbl_dir = os.path.join(self.label_dir, split, cityname,
                                   line['file_name'].replace('leftImg8bit.png', 'gtFine_labelIds.png'))
            assert os.path.isfile(img_dir), img_dir
            assert os.path.isfile(lbl_dir), lbl_dir
            self.images.append(img_dir)
            self.labels.append(lbl_dir)

    def __len__(self):
        return len(self.images)

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_caption_list_objects(self, idx):
        caption = random.choice(self.metadata[idx]['text'])
        return caption

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def __getitem__(self, idx):
        # load image
        try:
            original_pil_image = Image.open(self.images[idx]).convert("RGB")
            original_pil_target = Image.open(self.labels[idx])
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {self.images[idx]}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Transforms
        image = ToTensor()(original_pil_image)
        label = ToTensorNoNorm()(original_pil_target)
        label = self.map_fn[label.long()]
        img_lbl = self.augmentation(torch.cat([image, label]))

        caption = self.get_caption_list_objects(idx)

        return img_lbl[:3], img_lbl[3:], caption


def indices_segmentation_to_img(indices, colors):
    if indices.size(1) == 1:
        # Remove single channel axis.
        indices = indices[:, 0]
    # for train_id
    indices = indices * (indices != 255) + torch.ones_like(indices) * 19 * (indices == 255)
    rgbs = colors[indices]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs / 255.


def get_colors_from_id_type(id_type):
    num_classes = len(id_type_to_classes[id_type]['map_fn'].unique())
    colors = torch.zeros((num_classes, 3))
    exist_ids = []
    for idx, cls in enumerate(id_type_to_classes[id_type]['map_fn']):
        if cls == 255:
            cls = 19
        if cls not in exist_ids:
            colors[cls] = torch.tensor(classes[idx].color)
            exist_ids.append(cls)
    return colors


def transform_lbl(lbl, id_type='id'):
    colors = get_colors_from_id_type(id_type)
    return indices_segmentation_to_img(lbl, colors)


def transform_img_lbl(x, id_type='id', unnorm=True):
    colors = get_colors_from_id_type(id_type)

    x = x.detach().cpu()
    x = x.unsqueeze(0) if x.dim() == 3 else x

    # b, _, h, w = x.shape
    img = x[:, :3]
    lbl = x[:, 3:].long()
    img = unnormalize_to_zero_to_one(img) if unnorm else img
    saved_img = torch.cat([img, indices_segmentation_to_img(lbl, colors)])  # b * 2, 3, h ,w
    return saved_img


def trainId2label_fn(train_id_map):
    saved_label_id = torch.zeros_like(train_id_map)
    for t_id, label in trainId2label.items():
        if label.ignoreInEval:
            continue
        saved_label_id[train_id_map == t_id] = label.id
    return saved_label_id


def change_19_to_255(id_map):
    id_map[id_map == 19] = 255
    return id_map
