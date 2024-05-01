import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_filter(input_file, sigma=1.0):
    # 加载 NIfTI 文件
    image = nib.load(input_file)
    image_data = image.get_fdata()

    # 应用高斯滤波
    filtered_data = gaussian_filter(image_data, sigma=sigma)

    # 创建新的 NIfTI 图像
    filtered_image = nib.Nifti1Image(filtered_data, affine=image.affine, header=image.header)

    return filtered_image
