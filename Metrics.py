import torch


def SSIM():
    return None

def PSNR(loss):
    psnr = -10. * torch.log10(loss)
    return psnr