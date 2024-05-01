import cv2
import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from Models.SAM import sam_model_registry

torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "./Checkpoints/MedSAM.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


@torch.no_grad()
def get_embeddings(file_path):
    img_np = io.imread(file_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np

    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        embedding = medsam_model.image_encoder(
            img_1024_tensor
        )  # (1, 256, 64, 64)

    return img_3c, embedding


def blended_mask(image_path, mask, color, alpha=0.3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The specified image path does not exist.")

    colored_overlay = np.zeros_like(image, dtype=np.uint8)
    colored_overlay[mask > 0] = color

    blended_image = cv2.addWeighted(image, 1 - alpha, colored_overlay, alpha, 0)

    return blended_image


def cutout_mask(image_path, mask):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The specified image path does not exist.")

    red_background = np.zeros_like(image, dtype=np.uint8)
    red_background[:, :] = [0, 0, 255]

    cutout_image = np.where(mask[:, :, np.newaxis] > 0, image, red_background)

    return cutout_image



def outline_mask(image_path, mask, outline_color=(0, 255, 0), thickness=2):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The specified image path does not exist.")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, contours, -1, outline_color, thickness)

    return image


medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()


def apply_2D_segmentation(filepath, xmin, ymin, xmax, ymax, step=1):
    img_3c, embedding = get_embeddings(filepath)

    H, W, _ = img_3c.shape
    box_np = np.array([[xmin, ymin, xmax, ymax]])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    sam_mask = medsam_inference(medsam_model, embedding, box_1024, H, W)

    mask_image = blended_mask(filepath, sam_mask, color=colors[step-1])
    cutout_image = cutout_mask(filepath, sam_mask)
    outline_image = outline_mask(filepath, sam_mask, outline_color=(0, 255, 0), thickness=2)

    return mask_image, cutout_image, outline_image


# a = apply_2D_segmentation("../TestFiles/Seg2D_test.jpg", 27.0, 50.0, 423.0, 489.0, step=1)
