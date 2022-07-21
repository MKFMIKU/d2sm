# Install via: pip install pytorch-msssim
# Install via: pip install git+https://github.com/S-aiueo32/lpips-pytorch.git
import argparse
from os import listdir
from os.path import join

import torch
import numpy as np
from PIL import Image
from lpips_pytorch import lpips
from pytorch_msssim import ssim, ms_ssim
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def is_image_file(f):
    filename_lower = f.lower()
    return any(
        filename_lower.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".tif"]
    )


def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dir1", "--dir1", type=str, default="./imgs/ex_dir0")
    parser.add_argument("-dir2", "--dir2", type=str, default="./imgs/ex_dir0")
    parser.add_argument("-b", "--border", default=32, type=int, metavar="N")
    parser.add_argument("--cuda", action="store_true", help="turn on flag to use GPU")
    parser.add_argument("--benchmark", action="store_true", help="benchmark validation")
    opt = parser.parse_args()

    device = "cuda" if opt.cuda else "cpu"
    shave_border = opt.border

    list_im1 = load_all_image(opt.dir1)
    list_im2 = load_all_image(opt.dir2)
    list_im1.sort()
    list_im2.sort()

    psnrs = []
    ssims = []
    msssims = []
    lpipss = []
    for im1_path, im2_path in tqdm(zip(list_im1, list_im2), total=len(list_im1)):
        data, gt = Image.open(im1_path), Image.open(im2_path)
        data_th, gt_th = (
            to_tensor(data).unsqueeze(0).to(device),
            to_tensor(gt).unsqueeze(0).to(device),
        )
        data_th = torch.clip(data_th, 0., 1.)
        gt_th = torch.clip(gt_th, 0., 1.)

        if shave_border != 0:
            data_th = data_th[
                :, :, shave_border:-shave_border, shave_border:-shave_border
            ]
            gt_th = gt_th[:, :, shave_border:-shave_border, shave_border:-shave_border]

        if opt.benchmark:
            if data_th.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = (
                    torch.FloatTensor(gray_coeffs).to(device).view(1, 3, 1, 1) / 256
                )
                data_th = data_th.mul(convert).sum(dim=1, keepdim=True)
                gt_th = gt_th.mul(convert).sum(dim=1, keepdim=True)

                ih, iw = data_th.shape[2:]
                gt_th = gt_th[:, :, 0:ih, 0:iw]

        if shave_border != 0:
            data_th = data_th[
                :, :, shave_border:-shave_border, shave_border:-shave_border
            ]
            gt_th = gt_th[:, :, shave_border:-shave_border, shave_border:-shave_border]

        if opt.benchmark:
            if data_th.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = (
                    torch.FloatTensor(gray_coeffs).to(device).view(1, 3, 1, 1) / 256
                )
                data_th = data_th.mul(convert).sum(dim=1, keepdim=True)
                gt_th = gt_th.mul(convert).sum(dim=1, keepdim=True)

                ih, iw = data_th.shape[2:]
                gt_th = gt_th[:, :, 0:ih, 0:iw]

        psnrs.append(-10 * np.log10((data_th - gt_th).pow(2).mean().item()))
        ssims.append(ssim(data_th, gt_th, data_range=1.0, size_average=False).item())
        msssims.append(
            ms_ssim(data_th, gt_th, data_range=1.0, size_average=False).item()
        )
        # lpipss.append(
        #     lpips(gt_th * 2 - 1, data_th * 2 - 1, net_type="alex", version="0.1").item()
        # )
        # print(im1_path, im2_path, psnrs[-1])
    print("mean PSNR:", np.mean(psnrs))
    print("mean SSIM:", np.mean(ssims))
    print("mean MSSSIM:", np.mean(msssims))
    # print("mean LPIPS:", np.mean(lpipss))
