import pandas as pd
import pytest


@pytest.fixture
def original():
    original = pd.read_csv("data/labels_crowdai.csv")
    original = original[original["Label"].isin(["Car", "Truck"])]
    return original.reset_index(drop=True)


@pytest.fixture
def resized():
    return pd.read_csv("labels_resized.csv")


def test_shape_of_labels(original: pd.DataFrame,
                         resized: pd.DataFrame) -> None:
    """Check the shape of CSV Files"""

    shape_original = original.shape
    shape_resized = resized.shape

    # Same Row
    assert shape_original[0] == shape_resized[0]

    # Mask column was added to the original
    assert shape_original[1] + 1 == shape_resized[1]



def test_bbox_is_smaller(original, resized) -> None:
    """resized bbox should be always smaller than original"""

    assert (resized["xmin"] > original["xmin"]).sum() == 0
    assert (resized["xmax"] > original["xmax"]).sum() == 0
    assert (resized["ymin"] > original["ymin"]).sum() == 0
    assert (resized["ymax"] > original["ymax"]).sum() == 0
