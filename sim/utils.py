"""Utility functions for send/fetch data from Airsim"""

from typing import Optional, Dict
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


# Map actions from action_space to Drone Representation in Airsim.
def map_actions(actions: np.ndarray) -> Dict:
    roll, pitch, yaw, throttle = actions
    action_range = [-1,1]
    return {
            "roll" : np.interp(roll, action_range, [-np.pi/2, np.pi/2]).round(2),
            "pitch": np.interp(pitch, action_range, [-np.pi/2, np.pi/2]).round(2),
            "yaw_rate" : np.interp(yaw, action_range, [0, 2* np.pi]).round(2),
            "throttle": np.interp(throttle, action_range, [0,1]).round(2),
        } 
