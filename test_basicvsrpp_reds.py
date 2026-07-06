import cv2
import glob
import logging
import os
import os.path as osp
import re
import torch
from archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.data.data_util import read_img_seq
from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, get_time_str, imwrite, tensor2img


def remap_mmedit_spynet_keys(sd):
    """mmediting's SPyNetBasicModule indexes its 5 conv layers 0..4 with a mmcv ConvModule
    '.conv' wrapper; our SpyNet (basicsr) is a flat nn.Sequential with convs at even indices
    (0,2,4,6,8, odd = ReLU). Remap conv_idx -> 2*conv_idx and drop the '.conv' indirection."""
    pat = re.compile(r'basic_module\.(\d+)\.basic_module\.(\d+)\.conv\.(weight|bias)')
    out = {}
    for k, v in sd.items():
        m = pat.match(k)
        out[f'basic_module.{m.group(1)}.basic_module.{int(m.group(2))*2}.{m.group(3)}' if m else k] = v
    return out



def main():
    # -------------------- Configurations -------------------- #
    device = torch.device('cuda')
    save_imgs = True
    test_y_channel = False
    crop_border = 0

    model_path = 'pretrained/basicvsrpp_reds_x4_official.pth'
    spynet_path = 'pretrained/basicvsrpp_spynet_official.pth'

    test_name = f'REDS_30frames_basicvsrpp'

    lr_folder = '/hdd/laniko/Dataset/REDS_dataset/REDS4/sharp_bicubic'
    gt_folder = '/hdd/laniko/Dataset/REDS_dataset/REDS4/GT'

    save_folder = f'results/{test_name}'
    os.makedirs(save_folder, exist_ok=True)

    # logger
    log_file = osp.join(save_folder, f'psnr_ssim_test_{get_time_str()}.log')
    logger = get_root_logger(logger_name='recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {test_name} - {lr_folder}')
    logger.info(f'Model path: {model_path}')

    # set up the models
    model = BasicVSRPlusPlus(mid_channels=64,
                  num_blocks=7,
                  is_low_res_input=True,
                  spynet_path=None)  # spynet is loaded separately below (official ckpt key format differs)

    # main model checkpoint (mmediting release format: state_dict, some keys prefixed "generator.")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("generator.", "", 1) if k.startswith("generator.") else k: v
                  for k, v in state_dict.items()}

    # spynet weights ship separately upstream and are loaded into our own SpyNet, whose
    # BasicModule is a flat nn.Sequential (no mmcv ConvModule ".conv" indirection)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("spynet.")}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print("Main model loaded. Missing keys (should be spynet.*):", missing)

    spynet_ckpt = torch.load(spynet_path, map_location="cpu")
    new_spynet_ckpt = remap_mmedit_spynet_keys(spynet_ckpt)
    missing, unexpected = model.spynet.load_state_dict(new_spynet_ckpt, strict=False)
    print("spynet loaded. Missing keys:", missing)
    print("Unexpected keys:", unexpected)



    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_ssim_l = []
    subfolder_l = sorted(glob.glob(osp.join(lr_folder, '*')))
    subfolder_gt_l = sorted(glob.glob(osp.join(gt_folder, '*')))

    # for each subfolder
    subfolder_names = []
    for subfolder, subfolder_gt in zip(subfolder_l, subfolder_gt_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_names.append(subfolder_name)

        # read lq and gt images
        imgs_lq, imgnames = read_img_seq(subfolder, return_imgname=True)

        # calculate the iter numbers
        length = len(imgs_lq)

        avg_psnr = 0
        avg_ssim = 0
        # inference
        name_idx = 0
        imgs_lq = imgs_lq.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(imgs_lq).squeeze(0)

        # convert to numpy image
        for idx in range(outputs.shape[0]):
            img_name = imgnames[name_idx] + '.png'
            output = tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1))
            # read GT image
            img_gt = cv2.imread(osp.join(subfolder_gt, img_name), cv2.IMREAD_UNCHANGED)
            crt_psnr = psnr_ssim.calculate_psnr(
                output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            crt_ssim = psnr_ssim.calculate_ssim(
            output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            # save
            if save_imgs:
                imwrite(output, osp.join(save_folder, subfolder_name, f'{img_name}'))
            avg_psnr += crt_psnr
            avg_ssim += crt_ssim
            logger.info(f'{subfolder_name}--{img_name} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')
            name_idx += 1

        avg_psnr /= name_idx
        logger.info(f'name_idx:{name_idx}')
        avg_ssim /= name_idx
        avg_psnr_l.append(avg_psnr)
        avg_ssim_l.append(avg_ssim)

    for folder_idx, subfolder_name in enumerate(subfolder_names):
        logger.info(f'Folder {subfolder_name} - Average PSNR: {avg_psnr_l[folder_idx]:.6f} dB. Average SSIM: {avg_ssim_l[folder_idx]:.6f}.')

    logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')
    logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f}  '
    f'for {len(subfolder_names)} clips. ')


if __name__ == '__main__':

    main()