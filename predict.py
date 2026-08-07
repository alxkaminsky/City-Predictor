# basic python imports are permitted
import sys
import csv
import random
import train
# numpy and pandas are also permitted
import numpy as np
import pandas as pd

CITIES = ['Dubai', 'New York City', 'Paris', 'Rio de Janeiro']

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)

    data = pd.read_csv(filename)
    W, vocab, stats = train.fit()
    X, _, _ = train.transform(data, vocab, stats=stats)
    Z = X @ np.transpose(W)
    return [CITIES[i] for i in np.argmax(Z, axis=1)]
