import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter


def apply_median_filter(input_file, size=3):
    # 加载 NIfTI 文件
    image = nib.load(input_file)
    image_data = image.get_fdata()

    # 应用中值滤波
    filtered_data = median_filter(image_data, size=size)

    # 创建新的 NIfTI 图像
    filtered_image = nib.Nifti1Image(filtered_data, affine=image.affine, header=image.header)
    return filtered_image