import gzip
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from tqdm import tqdm


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urlretrieve(url, filename, reporthook=t.update_to)


def read_data(dir_path):
    """read data into pandas dataframe"""

    def load_dir_reviews(reviews_path):
        files_list = list(reviews_path.iterdir())
        reviews = []
        for filename in files_list:
            f = open(filename, "r", encoding="utf-8")
            reviews.append(f.read())
        return pd.DataFrame({"text": reviews})

    pos_path = dir_path / "pos"
    neg_path = dir_path / "neg"

    pos_reviews, neg_reviews = load_dir_reviews(pos_path), load_dir_reviews(neg_path)

    pos_reviews["label"] = 1
    neg_reviews["label"] = 0

    merged = pd.concat([pos_reviews, neg_reviews])
    df = merged.sample(frac=1.0)  # shuffle the rows
    df.reset_index(inplace=True)  # don't carry index from previous
    df.drop(columns=["index"], inplace=True)  # drop the column 'index'
    return df
