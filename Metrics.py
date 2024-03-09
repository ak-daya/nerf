import numpy as np
import cv2

def ssim(img1, img2):
    
    # constants to stabilize the division with weak denominator
    k1 = 0.01
    k2 = 0.03
    # 8-bit dynamic range of pixel-values
    L = 2**8 - 1    
    C1 = (k1*L)**2
    C2 = (k2*L)**2

    # filtering options
    Ksize= (11, 11)
    SigmaX = 1.5
    SigmaY = 1.5

    # work in floating-point precision
    I1 = np.double(img1)
    I2 = np.double(img2)

    # mean
    mu1 = cv2.GaussianBlur(I1, Ksize, SigmaX, SigmaY)
    mu2 = cv2.GaussianBlur(I2, Ksize, SigmaX, SigmaY)

    # variance
    sigma1_2 = cv2.GaussianBlur(I1**2, Ksize, SigmaX, SigmaY) - mu1**2
    sigma2_2 = cv2.GaussianBlur(I2**2, Ksize, SigmaX, SigmaY) - mu2**2
    
    # covariance
    sigma12 = cv2.GaussianBlur(I1*I2, Ksize, SigmaX, SigmaY) - mu1*mu2

    # SSIM index
    map = ((2 * mu1*mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_2 + sigma2_2 + C2))
    ssim = np.mean(map, axis=None)

    return ssim

def psnr(img1, img2):
    # Suppress divide by 0 errors
    np.seterr(divide='ignore', invalid='ignore')

    mse = np.mean(((img1 - img2)**2), axis=None)
    psnr = 10. * np.log10(255.*255. / mse)
    
    if np.isinf(psnr):
        psnr = 0
    
    return psnr

# if __name__ == "__main__":
    # image1 = cv2.imread("img_0.png")
    # image2 = cv2.imread("img_1.png")

    # print("SSIM: ")
    # print(ssim(image1, image1))
    
    # print("PSNR: ")
    # print(psnr(image1, image1))
