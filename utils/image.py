# pylint: disable=C0326,W0102,C0103,R0913,E1101
"""
Image related functions
"""
import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(image_path: str, gray: bool=False) -> np.ndarray:
    """Returns an image array

    Args:
        image_path (str): Path to image.jpg
        gray (bool): Grayscale flag

    Returns:
        np.ndarray:
          3D numpy array of shape (H, W, 3) or 2D Grayscale Image (H, W)
    """
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image_and_resize(image_path: str,
                          new_WH: Tuple[int, int]=(512, 512),
                          save_dir: str="resize") -> str:
    """Reads an image and resize it

    1) open `image_path` that is image.jpg
    2) resize to `new_WH`
    3) save to save_dir/image.jpg
    4) returns `image_path`

    Args:
        image_path (str): /path/to/image.jpg
        new_WH (tuple): Target width & height to resize
        save_dir (str): Directory name to save a resized image

    Returns:
        image_path (str): same as input `image_path`
    """
    assert os.path.exists(save_dir) is True
    new_path = os.path.join(save_dir, os.path.basename(image_path))
    image = cv2.imread(image_path)
    image = cv2.resize(image, new_WH, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_path, image)

    return image_path


def plot_image(image: np.ndarray, title: Optional[str]=None, **kwargs) -> None:
    """Plot a single image

    Args:
        image (2-D or 3-D array): image as a numpy array (H, W) or (H, W, C)
        title (str, optional): title for a plot
        **kwargs: keyword arguemtns for `plt.imshow`
    """
    shape = image.shape

    if len(shape) == 3:
        plt.imshow(image, **kwargs)
    elif len(shape) == 2:
        plt.imshow(image, **kwargs)
    else:
        raise TypeError(
            "2-D array or 3-D array should be given but {} was given".format(
                shape))

    if title:
        plt.title(title)


def plot_two_images(image_A: np.ndarray,
                    title_A: str,
                    image_B: np.ndarray,
                    title_B: str,
                    figsize: Tuple[int, int]=(15, 15),
                    kwargs_1: dict={},
                    kwargs_2: dict={}) -> None:
    """Plot two images side by side"""
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plot_image(image_A, title=title_A, **kwargs_1)

    plt.subplot(1, 2, 2)
    plot_image(image_B, title=title_B, **kwargs_2)


def draw_bbox(image: np.ndarray,
              left_top: Tuple[int, int],
              right_bot: Tuple[int, int],
              color: Tuple[int, int, int],
              thickness: int,
              **kwargs) -> np.ndarray:
    """Returns an image with the bounding box
    Args:
        image (np.ndarray): Numpy array of shape (H, W, C)
        left_top (Tuple[int, int]): The left top coordinate, (column, row)
        right_bot (Tuple[int, int]): The right bottom coordinate (column, row)
        color (Tuple[int, int, int]): Color (R, G, B)
        thickness (int): Thickness of the box
        **kwargs (dict): kwargs for cv2.rectangle

    Returns:
        np.ndarray: Numpy array of same shape with bounding boxes
    """
    return cv2.rectangle(
        image, left_top, right_bot, color=color, thickness=thickness, **kwargs)
