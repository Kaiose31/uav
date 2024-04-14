"""Utility functions for send/fetch data from Airsim"""

from typing import Optional, Dict
from sim.conn import client
import airsim
import numpy as np
import cv2
from PIL import Image


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


# Map actions from action_space to Drone Representation in Airsim.
def map_actions(actions: np.ndarray) -> Dict:
    roll, pitch, yaw, throttle = actions
    action_range = [-1, 1]
    return {
        "roll": np.interp(roll, action_range, [-np.pi / 2, np.pi / 2]).round(2),
        "pitch": np.interp(pitch, action_range, [-np.pi / 2, np.pi / 2]).round(2),
        "yaw_rate": np.interp(yaw, action_range, [0, 2 * np.pi]).round(2),
        "throttle": np.interp(throttle, action_range, [0, 1]).round(2),
    }
