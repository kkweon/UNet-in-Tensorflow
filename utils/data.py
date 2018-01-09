# pylint: disable=E1101,C0103,C0326,W1202
"""
Data Related Functions

"""
import argparse
import logging
import os
import shutil
import time
from collections import namedtuple
from multiprocessing.pool import Pool
# For Typing Annotation
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from .image import read_image, read_image_and_resize

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

Box = namedtuple("Box", ["left_top", "right_bot"])


def read_flags():
    """Returns global variables"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="resize image and adjusts coordinates")
    parser.add_argument(
        "--src_csv",
        default="data/labels_crowdai.csv",
        help="/path/to/labels.csv")

    parser.add_argument(
        "--data_dir",
        default="object-detection-crowdai",
        help="Directory where training datasets are located")

    parser.add_argument(
        "--save_dir",
        default="data_resize",
        help="path to the directory in which resize image will be saved")

    parser.add_argument(
        "--target_width", default=960, help="new target width (default: 960)")

    parser.add_argument(
        "--target_height", default=640, help="target height (default: 640)")

    parser.add_argument(
        "--target_csv",
        default="labels_resized.csv",
        help="target csv filename")

    return parser.parse_args()


def get_boxes(df: pd.DataFrame) -> List[Box]:
    """Given relevant DATAFRAME return a list of BOX"""
    boxes = []
    for _, items in df.iterrows():

        left_top = items["xmin"], items["ymin"]
        right_bot = items["xmax"], items["ymax"]

        boxes.append(Box(left_top, right_bot))
    return boxes


def create_clean_dir(dirname: str) -> None:
    """Create an empty directory

    Args:
        dirname (str): An empty directory name to create
    """

    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    assert os.path.exists(dirname) is False

    os.mkdir(dirname)

    assert not os.listdir(dirname)


def adjust_bbox(bboxframe: pd.DataFrame,
                src_size: Tuple[int, int],
                dst_size: Tuple[int, int]) -> pd.DataFrame:
    """Returns a new dataframe with adjusted coordinates

      W            W_new
    +----+  ----> +-+
    |    | H      | | H_new
    +----+        +-+
    Args:
        bboxframe (pd.DataFrame): Bounding box infor dataframe
        src_size (Tuple[int, int]): Original image (width, height)
        dst_size (Tuple[int, int]): New image (width, height)

    Returns:
        pd.DataFrame: Its coordinates are adjusted to a new size
    """
    W, H = src_size
    W_new, H_new = dst_size

    bboxframe = bboxframe.copy()

    bboxframe['xmin'] = (bboxframe['xmin'] * W_new / W).astype(np.int16)
    bboxframe['xmax'] = (bboxframe['xmax'] * W_new / W).astype(np.int16)
    bboxframe['ymin'] = (bboxframe['ymin'] * H_new / H).astype(np.int16)
    bboxframe['ymax'] = (bboxframe['ymax'] * H_new / H).astype(np.int16)

    return bboxframe


def get_relevant_frames(image_path: str,
                        dataframe: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe that contains truck image

    Args:
        image_path (str): "path/to/image.jpg"
        dataframe (pd.DataFrame): The base frame to be searched

    Returns:
        pd.DataFrame: A dataframe that contains input images
    """

    cond = dataframe["Frame"] == image_path
    return dataframe[cond].reset_index(drop=True)


def get_mask(image: np.ndarray, bbox_frame: pd.DataFrame) -> np.ndarray:
    """Returns a binary mask

    Args:
        image (3-D array): Numpy array of shape (H, W, C)
        bbox_frame (pd.DataFrame): Dataframe related with the input image
            It contains bounding box coordinates

    Returns:
        2-D array: Mask shape (H, W)
            1 for bounding box area
            0 for background
    """

    H, W, _ = image.shape

    mask = np.zeros((H, W))

    for _, row in bbox_frame.iterrows():
        W_beg, W_end = row['xmin'], row['xmax']
        H_beg, H_end = row['ymin'], row['ymax']

        mask[H_beg:H_end, W_beg:W_end] = 1

    return mask


def create_mask(image_WH: Tuple[int, int],
                image_path: str,
                dataframe: pd.DataFrame) -> np.ndarray:
    """Returns a mask array

    Object = 255
    Else = 0

    Args:
        image_WH (Tuple[int, int]): Image size (width, height)
        image_path (str): /path/to/image.jpg
        dataframe (pd.DataFrame): DataFrame contains bbox information

    Returns:
        2-D array: Mask array

    Examples:
        >>> image_WH = (960, 640)
        >>> image_path = "images/image000.jpg"
        >>> mask = create_mask(image_WH, image_path, dataframe)
    """

    W, H = image_WH
    mask = np.zeros((H, W))

    bbox_frame = get_relevant_frames(image_path, dataframe)

    for _, row in bbox_frame.iterrows():
        W_beg, W_end = row['xmin'], row['xmax']
        H_beg, H_end = row['ymin'], row['ymax']

        mask[H_beg:H_end, W_beg:W_end] = 255

    return mask


def generate_mask_pipeline(image_WH: Tuple[int, int],
                           image_path: str,
                           dataframe: pd.DataFrame,
                           save_dir: str="mask") -> None:
    """Create a mask and save as JPG

    Args:
        image_WH (Tuple[int, int]): (width: int, height: int)
        image_path (str): path/to/image.jpg
        dataframe (pd.DataFrame): labels.csv
        save_dir (str): Save directory
    """
    filename = os.path.basename(image_path)
    full_path = os.path.join(save_dir, filename)
    mask = create_mask(image_WH, image_path, dataframe)

    cv2.imwrite(full_path, mask)


def main(FLAGS):
    """Main Function

    Notes:
        1. Read image and resize to Target Width, Height
        2. Resize bounding box coordinates accordingly
        3. Create masks with the bounding box
             background is 0 and vehicle is 255

    """
    new_WH = (FLAGS.target_width, FLAGS.target_height)
    data = pd.read_csv(FLAGS.src_csv)
    # Only consider car and truck images
    data = data[data["Label"].isin(["Car", "Truck"])].reset_index(drop=True)

    # 123.jpg -> object-detection-crowdai/123.jpg
    data["Frame"] = data["Frame"].map(
        lambda x: os.path.join(FLAGS.data_dir, x))

    # IF dir exists, clean it
    create_clean_dir(FLAGS.save_dir)
    LOGGER.info("Cleaned {} directory".format(FLAGS.save_dir))

    LOGGER.info("Resizing begins")
    start = time.time()
    pool = Pool()
    pool.starmap_async(read_image_and_resize,
                       [(image_path, new_WH, FLAGS.save_dir)
                        for image_path in data["Frame"].unique()])

    pool.close()
    pool.join()
    end = time.time()

    LOGGER.info("Time elapsed: {}".format(end - start))
    LOGGER.info("Resizing ends")

    LOGGER.info("Adjusting dataframe")

    # Read any image file to get the WIDTH and HEIGHT
    image_path = data["Frame"][0]
    image = read_image(image_path)

    H, W, _ = image.shape
    src_size = (W, H)

    labels = adjust_bbox(data, src_size, new_WH)

    # object-.../123.jpg -> data_resize/123.jpg
    labels["Frame"] = labels["Frame"].map(
        lambda x: os.path.join(FLAGS.save_dir, os.path.basename(x)))

    create_clean_dir("mask")
    LOGGER.info("Cleaned {} directory".format("mask"))
    LOGGER.info("Masking begin")
    start = time.time()

    pool = Pool()
    tasks = [(new_WH, image_path, labels, "mask")
             for image_path in labels["Frame"].unique()]
    pool.starmap_async(generate_mask_pipeline, tasks)
    pool.close()
    pool.join()
    end = time.time()
    LOGGER.info("Masking ends. Time elapsed: {}".format(end - start))

    labels["Mask"] = labels["Frame"].map(
        lambda x: os.path.join("mask", os.path.basename(x)))
    labels.to_csv(FLAGS.target_csv, index=False)

    LOGGER.info("Adjustment saved to {}".format(FLAGS.target_csv))


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
