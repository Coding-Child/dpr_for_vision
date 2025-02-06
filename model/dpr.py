import copy

import torch
import torch.nn as nn
from torchvision import models


def dpr_create():
    # model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    # model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    model = nn.Sequential(
        *list(model.children())[:-1],
        nn.Flatten()
    )

    query_encoder = copy.deepcopy(model)
    passage_encoder = copy.deepcopy(model)

    return query_encoder, passage_encoder
