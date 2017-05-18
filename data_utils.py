import os
import shutil
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from keras import backend as K

matplotlib.use('qt5agg')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_image(image, title=None, **kwargs):
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
            "2-D array or 3-D array should be given but {} was given".format(shape))

    if title:
        plt.title(title)


def plot_two_images(image_A, title_A, image_B, title_B, figsize=(15, 15), kwargs_1={}, kwargs_2={}):
    """Plot two images side by side"""
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plot_image(image_A, title=title_A, **kwargs_1)

    plt.subplot(1, 2, 2)
    plot_image(image_B, title=title_B, **kwargs_2)


def create_clean_dir(dirname="resize"):
    """Create an empty directory

    Args:
        dirname (str): An empty directory name to create
    """

    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    assert os.path.exists(dirname) is False

    os.mkdir(dirname)

    assert len(os.listdir(dirname)) == 0


def read_image(image_path):
    """Returns an image array

    Args:
        image_path (str): Path to image.jpg

    Returns:
        3-D array: RGB numpy image array
    """

    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image_and_resize(image_path, new_WH=(512, 512), save_dir="resize"):
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
    new_path = os.path.join(save_dir, os.path.basename(str(image_path)))
    image = cv2.imread(image_path)
    image = cv2.resize(image, new_WH, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_path, image)

    return image_path


def adjust_bbox(bboxframe, src_size, dst_size):
    """Returns a new dataframe with adjusted coordinates

      W            W_new
    +----+  ----> +-+
    |    | H      | | H_new
    +----+        +-+
    Args:
        bboxframe (pd.DataFrame): Bounding box infor dataframe
        src_size (tuple): Original image (width, height)
        dst_size (tuple): New image (width, height)

    Returns:
        pd.DataFrame: Its coordinates are adjusted to a new size
    """
    W, H = src_size
    W_new, H_new = dst_size

    bboxframe = bboxframe.copy()

    bboxframe['xmin'] = (bboxframe['xmin'] * W_new / W).astype(np.int16)
    bboxframe['xmax'] = (bboxframe['xmax'] * H_new / H).astype(np.int16)
    bboxframe['ymin'] = (bboxframe['ymin'] * W_new / W).astype(np.int16)
    bboxframe['ymax'] = (bboxframe['ymax'] * H_new / H).astype(np.int16)

    return bboxframe


def read_flags():
    """Returns global variables"""

    parser = argparse.ArgumentParser(description="resize image and adjusts coordinates")
    parser.add_argument("--src_csv",
                        default="labels.csv",
                        help="/path/to/labels.csv (default: ./labels.csv)")

    parser.add_argument("--save_dir",
                        default="resize",
                        help="path to the directory in which resize image will be saved (default: resize)")

    parser.add_argument("--target_width",
                        default=960,
                        help="new target width (default: 960)")

    parser.add_argument("--target_height",
                        default=640,
                        help="target height (default: 640)")

    parser.add_argument("--target_csv",
                        default="labels_resized.csv",
                        help="target csv filename")

    return parser.parse_args()


def get_relevant_frames(image_path_list, dataframe):
    """Returns a dataframe that contains input image path

    Args:
        image_path_list (1-D array): Each element is a str "path/to/image.jpg"
        dataframe (pd.DataFrame): The base frame to be searched

    Returns:
        pd.DataFrame: A dataframe that contains input images
    """

    return dataframe[dataframe["Frame"].isin(image_path_list)].reset_index(drop=True)


def get_mask(image, bbox_frame):
    """Returns a binary mask

    Args:
        image (3-D array): Numpy array (H, W, C)
        bbox_frame (pd.DataFrame): Dataframe related with the input image
            It contains bounding box coordinates

    Returns:
        2-D array: Mask shape (H, W)
            1 for bounding box area
            0 for background
    """

    H, W, C = image.shape

    mask = np.zeros((H, W))

    for idx, row in bbox_frame.iterrows():
        W_beg, H_beg = row['xmin'], row['xmax']
        W_end, H_end = row['ymin'], row['ymax']

        mask[H_beg:H_end, W_beg:W_end] = 1

    return mask


def create_mask(image_WH, image_path, dataframe):
    """Returns a mask array

    Object = 1
    Else = 0

    Args:
        image_WH (tuple): Numpy array (width, height)
        image_path (str): /path/to/image.jpg
        dataframe (pd.DataFrame): Pandas dataframe

    Returns:
        2-D array: Mask array

    Examples:
        >>> image_WH = (960, 640)
        >>> image_path = "images/image000.jpg"
        >>> mask = create_mask(image_WH, image_path, dataframe)
    """

    W, H = image_WH
    mask = np.zeros((H, W))

    bbox_frame = get_relevant_frames([image_path], dataframe)

    for idx, row in bbox_frame.iterrows():
        W_beg, H_beg = row['xmin'], row['xmax']
        W_end, H_end = row['ymin'], row['ymax']

        mask[H_beg:H_end, W_beg:W_end] = 255

    return mask


def generate_mask_pipeline(image_WH, image_path, dataframe, save_dir="mask"):
    """Create a mask and save as JPG
    
    Args:
        image_WH (tuple): (width: int, height: int)
        image_path (str): path/to/image.jpg
        dataframe (pd.DataFrame): labels.csv
        save_dir (str): Save directory
    """
    filename = os.path.basename(image_path)
    full_path = os.path.join(save_dir, filename)
    mask = create_mask(image_WH, image_path, dataframe)

    cv2.imwrite(full_path, mask)


def get_IOU(y_true, y_pred):
    """Returns the intersection over union (IOU)

    Args:
        y_true (4-D array): (None, H, W, C)
        y_pred (4-D array): (None, H, W, C)

    Returns:
        IOU (float): intersection over unions
            the bigger, the beter

    Notes:
        Dice Coefficient
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    numerator = 2 * K.sum(y_true * y_pred) + 1e-7
    denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7

    return numerator / denominator


def IOU_loss(y_true, y_pred):
    return - get_IOU(y_true, y_pred)


def main(FLAGS):
    save_dir = FLAGS.save_dir
    new_WH = (FLAGS.target_width, FLAGS.target_height)
    new_labels_name = FLAGS.target_csv
    data = pd.read_csv(FLAGS.src_csv)
    data["Frame"] = data["Frame"].map(lambda x: "images/" + x)

    create_clean_dir(save_dir)
    logger.info("Cleaned {} directory".format(save_dir))

    logger.info("Resizing begins")
    start = time.time()
    pool = Pool()
    pool.starmap_async(read_image_and_resize, [(image_path, new_WH, save_dir) for image_path in data["Frame"].unique()])

    pool.close()
    pool.join()
    end = time.time()

    logger.info("Time elapsed: {}".format(end - start))
    logger.info("Resizing ends")

    logger.info("Adjusting dataframe")
    image_path = data["Frame"][0]
    image = read_image(image_path)

    H, W, _ = image.shape
    src_size = (W, H)
    labels = adjust_bbox(data, src_size, new_WH)
    labels["Frame"] = labels["Frame"].map(lambda x: os.path.join(save_dir, os.path.basename(x)))

    create_clean_dir("mask")
    logger.info("Cleaned {} directory".format("mask"))
    logger.info("Masking begin")
    start = time.time()

    pool = Pool()
    tasks = [(new_WH, image_path, labels, "mask")for image_path in labels["Frame"].unique()]
    pool.starmap_async(generate_mask_pipeline, tasks)
    pool.close()
    pool.join()
    end = time.time()
    logger.info("Masking ends. Time elapsed: {}".format(end - start))

    labels["Mask"] = labels["Frame"].map(lambda x: "mask/" + os.path.basename(x))
    labels.to_csv(new_labels_name, index=False)

    logger.info("Adjustment saved to {}".format(new_labels_name))


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
