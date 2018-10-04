import gzip
import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import get_data, read_data

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(str(__name__) + ".log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

data_path = Path(os.getcwd()) / "data" / "aclImdb"
logger.info(data_path)

if not data_path.exists():
    data_url = "http://files.fast.ai/data/aclImdb.tgz"
    get_data(data_url, "data/imdb.tgz")

train_path = data_path / "train"
# load data file as dict object
train = read_data(train_path)

# extract the images (X) and labels (y) from the dict
X_train, y_train = train["text"], train["label"]


lr_clf = Pipeline(
    [("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", LR())]
)
lr_clf.fit(X=X_train, y=y_train)

# save model
joblib.dump(lr_clf, "model.pkl")
