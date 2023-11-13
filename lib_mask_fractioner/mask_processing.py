import cv2
import numpy as np


# similar to how StableDiffusionProcessingImg2Img does it
def blur(mask, blur_amount):
    np_mask = np.array(mask)
    kernel_size = 2 * int(2.5 * blur_amount + 0.5) + 1
    return cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), blur_amount)


def mask_interpolation(mask, img_when_0, img_when_1):
    mask = np.stack([mask / 255]*3, axis=-1)
    return img_when_0*(1 - mask) + img_when_1*mask
