from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_video_paths_from_folder
import basicsr.data.video_util as utils_video 

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

from pathlib import Path
import random


import numpy as np
import os
import torch


@DATASET_REGISTRY.register()
class PairedVideoDataset(data.Dataset):
    """Video-style dataset for paired video restoration.

    Extended to behave like VideoRecurrentTrainVimeoDataset:
    - Uses meta_info_file
    - Performs random crop
    - Supports flip, rotate augmentation
    - Supports mirror_sequence and pad_sequence
    """

    def __init__(self, opt):
        super(PairedVideoDataset, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.gt_root = Path(opt['dataroot_gt'])
        self.lq_root = Path(opt['dataroot_lq'])

        self.use_hflip = opt.get('use_hflip', False)
        self.use_rot = opt.get('use_rot', False)
        self.mirror_sequence = opt.get('mirror_sequence', False)
        self.pad_sequence = opt.get('pad_sequence', False)
        self.random_reverse = opt.get('random_reverse', False)

        self.scale = opt['scale']
        self.gt_size = opt['gt_size']
        self.num_frame = opt['num_frame']

        self.filename_tmpl = opt.get('filename_tmpl', 'im{}')

        if 'meta_info_file' in opt and opt['meta_info_file'] is not None:
            with open(opt['meta_info_file'], 'r') as f:
                self.keys = [line.strip().split(' ')[0] for line in f]
        else:
            raise ValueError('meta_info_file must be provided for this dataset')

        # define neighbor frame indices (like 1,2,3,4,5,6,7)
        self.neighbor_list = [i + (9 - self.num_frame) // 2 for i in range(self.num_frame)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        clip, seq = key.split('/')

        # Apply random reverse
        neighbor_list = self.neighbor_list.copy()
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        img_lqs, img_gts = [], []
        for neighbor in neighbor_list:
            lq_path = self.lq_root / clip / seq / f'{self.filename_tmpl.format(neighbor)}.png'
            gt_path = self.gt_root / clip / seq / f'{self.filename_tmpl.format(neighbor)}.png'

            lq = imfrombytes(self.file_client.get(str(lq_path), 'lq'), float32=True)
            gt = imfrombytes(self.file_client.get(str(gt_path), 'gt'), float32=True)

            img_lqs.append(lq)
            img_gts.append(gt)

        # Random crop (same as utils_video.paired_random_crop)
        img_gts, img_lqs = utils_video.paired_random_crop(
            img_gts, img_lqs, self.gt_size, self.scale, str(gt_path)
        )

        # Augmentation
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.use_hflip, self.use_rot)

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
        img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        # Mirror or pad
        if self.mirror_sequence:
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)
        if self.pad_sequence:
            img_lqs = torch.cat([img_lqs, img_lqs[-1:].clone()], dim=0)
            img_gts = torch.cat([img_gts, img_gts[-1:].clone()], dim=0)

        return {
            'lq': img_lqs,   # [T, C, H, W]
            'gt': img_gts,   # [T, C, H, W]
            'lq_path': key,
            'gt_path': key
        }

    def __len__(self):
        return len(self.keys)
