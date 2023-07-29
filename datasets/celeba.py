import os
import os.path as osp
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

CelebaClass = namedtuple('CelebaClass', ['name', 'id', 'color'])
# autopep8: off
classes = [
    CelebaClass('background',  0, (  0,   0,   0)),
    CelebaClass('skin',        1, (204,   0,   0)),
    CelebaClass('nose',        2, ( 76, 153,   0)),
    CelebaClass('eye_g',       3, (204, 204,   0)),
    CelebaClass('l_eye',       4, ( 51,  51, 255)),
    CelebaClass('r_eye',       5, (204,   0, 204)),
    CelebaClass('l_brow',      6, (  0, 255, 255)),
    CelebaClass('r_brow',      7, (255, 204, 204)),
    CelebaClass('l_ear',       8, (102,  51,   0)),
    CelebaClass('r_ear',       9, (255,   0,   0)),
    CelebaClass('mouth',      10, (102, 204,   0)),
    CelebaClass('u_lip',      11, (255, 255,   0)),
    CelebaClass('l_lip',      12, (  0,   0, 153)),
    CelebaClass('hair',       13, (  0,   0, 204)),
    CelebaClass('hat',        14, (255,  51, 153)),
    CelebaClass('ear_r',      15, (  0, 204, 204)),
    CelebaClass('neck_l',     16, (  0,  51,   0)),
    CelebaClass('neck',       17, (255, 153,  51)),
    CelebaClass('cloth',      18, (  0, 204,   0)),
]
# autopep8: on
num_classes = 19
mapping_id = torch.tensor([x.id for x in classes])
colors = torch.tensor([cls.color for cls in classes])


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


class CelebaDataset(Dataset):
    def __init__(
        self,
        root="",
        split='train',
        side_x=128,
        side_y=128,
        caption_list_dir='',
        augmentation_type='flip',
    ):
        super().__init__()
        self.root = Path(root)
        self.image_dir = osp.join(self.root, 'CelebA-HQ-img')
        self.label_dir = osp.join(self.root, 'CelebAMask-HQ-mask-anno', 'preprocessed')
        self.split = split
        self.side_x = side_x
        self.side_y = side_y

        self.caption_list_dir = caption_list_dir
        captions_jsonl = read_jsonl(osp.join(self.caption_list_dir, f'{split}_captions.jsonl'))
        self.caption_dict = {}
        for caption_jsonl in captions_jsonl:
            self.caption_dict[osp.splitext(caption_jsonl['file_name'])[0]] = caption_jsonl['text']

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
                RandomCrop((1024, 1024)),
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        else:
            raise NotImplementedError(augmentation_type)

        # verification
        self.images = sorted([osp.join(self.image_dir, file) for file in os.listdir(self.image_dir)
                              if osp.splitext(file)[0] in self.caption_dict.keys()])
        self.labels = sorted([osp.join(self.label_dir, file) for file in os.listdir(self.label_dir)
                              if osp.splitext(file)[0] in self.caption_dict.keys()])

        assert len(self.images) == len(self.labels), f'{len(self.images)} != {len(self.labels)}'
        for img, lbl in zip(self.images, self.labels):
            assert osp.splitext(osp.basename(img))[0] == osp.splitext(osp.basename(lbl))[0]

    def __len__(self):
        return len(self.images)

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.sequential_sample(ind=ind)

    def get_caption_list_objects(self, idx):
        filename = osp.splitext(osp.basename(self.images[idx]))[0]
        caption = random.choice(self.caption_dict[filename])
        return caption

    def __getitem__(self, idx):
        # load image label
        try:
            original_pil_image = Image.open(self.images[idx]).convert("RGB")
            original_pil_target = Image.open(self.labels[idx])
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {self.images[idx]}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Transforms
        image = Resize((1024, 1024), InterpolationMode.NEAREST)(ToTensor()(original_pil_image))
        label = Resize((1024, 1024), InterpolationMode.NEAREST)(ToTensorNoNorm()(original_pil_target).float())
        img_lbl = self.augmentation(torch.cat([image, label]))

        caption = self.get_caption_list_objects(idx)

        return img_lbl[:3], img_lbl[3:], caption


def transform_lbl(lbl: torch.Tensor, *args, **kwargs):
    lbl = lbl.long()
    if lbl.size(1) == 1:
        # Remove single channel axis.
        lbl = lbl[:, 0]
    rgbs = colors[lbl]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs / 255.
