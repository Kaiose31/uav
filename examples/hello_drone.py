from sim.utils import get_img, show_img
from sim.conn import client
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch
import numpy as np


def segmentation():
    client.moveToPositionAsync(5, 5, 5, 2).join()
    image = get_img()
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50()
    model.eval()
    img = torch.from_numpy(np.copy(image))
    preprocess = weights.transforms(antialias=True)
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    show_img(normalized_masks)
