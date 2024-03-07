"""Utility functions for send/fetch data from Airsim"""

from typing import Optional
from sim.conn import client

import airsim
import numpy as np
import cv2


def get_img(img_type: airsim.ImageType = airsim.ImageType.Scene, display=False) -> Optional[np.ndarray]:
    response = client.simGetImages([airsim.ImageRequest("0", img_type, False, False)])[0]
    imgf = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    if display:
        show_img(imgf)
    return imgf


def show_img(image_arr: np.ndarray) -> None:
    cv2.imshow("image", image_arr)
    cv2.waitKey(0)
