import os

import torch
import torch.nn as nn

from .text_tower import Text_Tower
from .SAT import SAT


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_text_encoder(tokenizer_path, text_encoder_checkpoint, device, gpu_id):
    model = Text_Tower(tokenizer_path, 768)

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    checkpoint = torch.load(text_encoder_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def build_segmentation_model(device, gpu_id):
    model = SAT("UNET", [288, 288, 96])

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    return model
