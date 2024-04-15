"""Utility functions for send/fetch data from Airsim"""

from typing import Optional, Dict, Tuple
from sim.conn import client
import airsim
import numpy as np
import cv2
from PIL import Image
from airsim.types import Quaternionr
import math


# Depth image with single channel (no compression)
def get_img(target_size: tuple, display=False) -> Optional[np.ndarray]:
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    img1d = np.array(responses[0].image_data_float, dtype=float)
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d)
    imgf = np.array(image.resize((target_size[0], target_size[1])).convert("L"))
    if display:
        show_img(imgf)
    return imgf.reshape(target_size)


def show_img(image_arr: np.ndarray) -> None:
    cv2.imshow("image", image_arr)
    cv2.waitKey(0)


def to_euler_angles(q: Quaternionr) -> Tuple[float, float, float]:
    roll = math.atan2(2 * (q.w_val * q.x_val + q.y_val * q.z_val), 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val))
    pitch = math.atan2(math.sqrt(1 + 2 * (q.w_val * q.y_val - q.x_val * q.z_val)), math.sqrt(1 - 2 * (q.w_val * q.y_val - q.x_val * q.z_val)))
    yaw = math.atan2(2 * (q.w_val * q.z_val + q.x_val * q.y_val), 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val))
    return roll, pitch, yaw
