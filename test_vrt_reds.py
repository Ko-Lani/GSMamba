# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader

from archs.vrt_arch import VRT as net

from basicsr.utils import utils_image as util
from basicsr.data.videos_test_dataset import VideoRecurrentTestDataset, VideoTestDataset
from basicsr.data.dataset_video_test import VideoTestVimeo90KDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='VRT_REDS4', help='task name, used as the results/ subfolder')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default='/hdd/laniko/Dataset/REDS_dataset/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default='/hdd/laniko/Dataset/REDS_dataset/REDS4/GT',
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[40,128,128],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')

    parser.add_argument('--model_path', type=str, default='pretrained/vrt_reds_x4_official.pth', help='model path')
    
    args = parser.parse_args()
 
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)
    

    if 'vimeo' in args.folder_lq.lower():
        test_set = VideoTestVimeo90KDataset({'name':"VRT_vimeo" ,'dataroot_gt': args.folder_gt, 'dataroot_lq': args.folder_lq,
                                                 'meta_info_file': "/hdd/laniko/Dataset/Vimeo90K/meta_info_Vimeo90K_test_GT.txt",
                                                 'pad_sequence': True, 'num_frame': 7, 'temporal_scale': 1,
                                                 'cache_data': False})
            
    
    else:
        test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                                'sigma':args.sigma, 'num_frame':-1, 'cache_data': False, 'io_backend': {'type': 'disk'}, 'name': 'REDS4', 'cache_data': True})
    
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)
 
    save_dir = f'results/{args.task}'
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    
    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'
 
 
    for idx, batch in enumerate(test_loader):
        

        # if 'vimeo' in args.folder_lq.lower():
        #     lq = batch['L'].to(device)
        #     folder = batch['folder']
        #     gt = batch['H'] if 'H' in batch else None  
        
        # else:
        lq = batch['lq'].to(device)
        folder = batch['folder']
        gt = batch['gt'] if 'gt' in batch else None

        # inference
        with torch.no_grad():
            output = test_video(lq, model, args)
            
            
        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        for i in range(output.shape[1]):

            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8


            if args.save_result:
                if 'vimeo' in args.folder_lq.lower():
                    # seq_ = osp.basename(batch['lq_path'][0]).split('.')[0]
                    seq_ = batch['lq_path'][0].split('/')[6:8]
                    save_path = os.path.join(save_dir, folder[0], *seq_)  # save_dir/folder_name/00001/0266
                    os.makedirs(save_path, exist_ok=True)

                    filename = f"im{i}.png"
                    cv2.imwrite(os.path.join(save_path, filename), img)
                    
                
                else:
                    os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                    cv2.imwrite(f'{save_dir}/{folder[0]}/{i}.png', img)
                    
                    
            if 'vimeo' in args.folder_lq.lower():
                if(i!=3):
                    continue
                
                
 
            # evaluate psnr/ssim
            if gt is not None:
                if 'vimeo' in args.folder_lq.lower():
                    img_gt = gt.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                else:
                    img_gt = gt[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                
                
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = np.squeeze(img_gt)

                test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                if img_gt.ndim == 3:  # RGB image
                    img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                    img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                else:
                    test_results_folder['psnr_y'] = test_results_folder['psnr']
                    test_results_folder['ssim_y'] = test_results_folder['ssim']
 
        if gt is not None:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            print('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                      format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))
 
    # summarize psnr/ssim
    if gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
              format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))
 


def prepare_model_dataset(args):
    ''' prepare model and dataset according to args.task. '''

    ## VRT, official REDS-16frames recipe (matches 002_VRT_videosr_bi_REDS_16frames.pth)
    model = net(upscale=4, img_size=[16,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=6, deformable_groups=24,
                spynet_path='pretrained/spynet_sintel.pth')

    args.scale = 4
    # args.window_size = 8
    args.window_size = [8,8,8]
    args.nonblind_denoising = False 
 
    # download model
    model_path = args.model_path
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
 
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=False)
 
    # download datasets
    if os.path.exists(f'{args.folder_lq}'):
        print(f'using dataset from {args.folder_lq}')
        
 
    return model


def test_video(lq, model, args):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = args.tile[0]
        
        
        ## test the whole
        # num_frame_testing = False
        
        
        
        
        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = args.scale
            num_frame_overlapping = args.tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if args.nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                with torch.no_grad():
                    out_clip = test_clip(lq_clip, model, args)
                
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = args.window_size
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            
            with torch.no_grad():
                output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' test the clip as a whole or as patches. '''

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    
    ##
    size_patch_testing_w = args.tile[2]
    
    # assert size_patch_testing % window_size == 0, 'testing patch size should be a multiple of window_size.'
    

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        
        ##
        stride_w = size_patch_testing_w - overlap_size
        
        
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        # w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        
        w_idx_list = list(range(0, w-size_patch_testing_w, stride_w)) + [max(0, w-size_patch_testing_w)]
        
        
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                # in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing_w]
                
                with torch.no_grad():
                    out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                # E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                # W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing_w)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing_w)*sf].add_(out_patch_mask)
                
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output


if __name__ == '__main__':
    main()