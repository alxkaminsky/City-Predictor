import train
import numpy as np
import pandas as pd

CITIES = ['Dubai', 'New York City', 'Paris', 'Rio de Janeiro']

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    data = pd.read_csv(filename)
    W, vocab, stats = train.fit()
    X, _, _ = train.transform(data, vocab, stats=stats)
    # Impute missing values with the column mean.
    X = np.nan_to_num(X, nan=0.0)
    Z = X @ np.transpose(W)
    return [CITIES[i] for i in np.argmax(Z, axis=1)]
