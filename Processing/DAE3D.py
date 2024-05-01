import torch
import nibabel as nib
from Models.DAE import DenoisingAutoEncoder

# 加载模型
model = DenoisingAutoEncoder()
model_path = "Checkpoints/DAE.pth"
model.load_state_dict(torch.load(model_path))
model.eval()


def apply_auto_denoise(input_file):
    # 加载 NIfTI 文件
    image = nib.load(input_file)
    image_data = image.get_fdata()
    image_data = torch.Tensor(image_data).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # 影像预处理
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

    # 自动去噪
    with torch.no_grad():
        denoised_data = model(image_data).squeeze()  # Remove batch and channel dimensions

    denoised_data = denoised_data * (
                image.get_data_dtype().max - image.get_data_dtype().min) + image.get_data_dtype().min
    denoised_data = denoised_data.numpy().astype(image.get_data_dtype())

    # 创建新的 NIfTI 图像
    denoised_image = nib.Nifti1Image(denoised_data, image.affine, image.header)
    return denoised_image
