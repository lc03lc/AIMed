import nibabel as nib
from mayavi import mlab
import pymesh
import os
import datetime
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from einops import reduce, rearrange, repeat

from Models.SAT.dataset.inference_dataset import Inference_Dataset, inference_collate_fn
from Models.SAT.model.tokenizer import MyTokenizer
from Models.SAT.model.build_model import load_text_encoder, build_segmentation_model

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_id = 0

model_checkpoint = "./Checkpoints/SAT.pth"
tokenizer_path = "./Checkpoints/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
text_encoder_path = "./Checkpoints/SAT_TextEncoder.pth"


def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    # build an gaussian filter with the patch size
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def inference(model, tokenizer, text_encoder, device, testloader, gaussian_filter):
    # inference
    model.eval()
    text_encoder.eval()

    with torch.no_grad():

        # in ddp, only master process display the progress bar
        if int(os.environ["LOCAL_RANK"]) != 0:
            testloader = tqdm(testloader, disable=True)
        else:
            testloader = tqdm(testloader, disable=False)

            # gaussian kernel to accumulate predcition
        windows = compute_gaussian((288, 288, 96)) if gaussian_filter else np.ones((288, 288, 96))

        for batch in testloader:  # one batch for each sample
            # data loading
            batched_patch = batch['batched_patch']
            batched_y1y2x1x2z1z2 = batch['batched_y1y2x1x2z1z2']
            split_prompt = batch['split_prompt']
            split_n1n2 = batch['split_n1n2']
            labels = batch['labels']
            image = batch['image']
            image_path = batch['image_path']

            _, h, w, d = image.shape
            n = split_n1n2[-1][-1]
            prediction = np.zeros((n, h, w, d))
            accumulation = np.zeros((n, h, w, d))

            # for each batch of queries
            for prompts, n1n2 in zip(split_prompt, split_n1n2):
                n1, n2 = n1n2
                input_text = tokenizer.tokenize(prompts)  # (max_queries, max_l)
                input_text['input_ids'] = input_text['input_ids'].to(device=device)
                input_text['attention_mask'] = input_text['attention_mask'].to(device=device)
                queries, _, _ = text_encoder(text1=input_text, text2=None)  # (max_queries, d)

                # for each batch of patches
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patch, batched_y1y2x1x2z1z2):  # [b, c, h, w, d]
                    batched_queries = repeat(queries, 'n d -> b n d', b=patches.shape[0])  # [b, n, d]
                    patches = patches.to(device=device)

                    prediction_patch = model(queries=batched_queries, image_input=patches)
                    prediction_patch = torch.sigmoid(prediction_patch)
                    prediction_patch = prediction_patch.detach().cpu().numpy()  # bnhwd

                    # fill in
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]

                        # accumulation
                        prediction[n1:n2, y1:y2, x1:x2, z1:z2] += prediction_patch[b, :n2 - n1, :y2 - y1, :x2 - x1,
                                                                  :z2 - z1] * windows[:y2 - y1, :x2 - x1, :z2 - z1]
                        accumulation[n1:n2, y1:y2, x1:x2, z1:z2] += windows[:y2 - y1, :x2 - x1, :z2 - z1]

            # avg
            prediction = prediction / accumulation
            prediction = np.where(prediction > 0.5, 1.0, 0.0)

            # save prediction
            save_dir = image_path.split('.')[0]  # xxx/xxx.nii.gz --> xxx/xxx
            np_images = image.numpy()[0, :, :, :]
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            results = np.zeros((h, w, d))  # merge on one channel
            for j, label in enumerate(labels):
                results += prediction[j, :, :, :] * (j + 1)
            return nib.nifti2.Nifti1Image(results, np.eye(4))


def apply_3D_reconstruction(input_file):
    label = ["brain", "head and neck", "abdomen", "lower limb", "upper limb", "thorax", "spine", "pelvis"]
    modality = "CT"

    testset = Inference_Dataset(input_file, label, modality, 1, 16, 1)
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=sampler, batch_size=1, collate_fn=inference_collate_fn, shuffle=False)
    sampler.set_epoch(0)

    # set segmentation model
    model = build_segmentation_model(device, gpu_id)

    # load checkpoint of segmentation model
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # load text encoder
    text_encoder = load_text_encoder(tokenizer_path, text_encoder_path, device, gpu_id)

    # set tokenizer
    tokenizer = MyTokenizer(tokenizer_path)

    segmentation_file =  inference(model, tokenizer, text_encoder, device, testloader, False)

    img = nib.load(segmentation_file)
    data = img.get_fdata()

    mlab.figure(bgcolor=(1, 1, 1))
    mlab.contour3d(data, transparent=True)
    mlab.view(azimuth=90, elevation=90, distance='auto')
    mlab.savefig("temp.obj")
    mlab.close()

    return pymesh.load_mesh("temp.obj")
