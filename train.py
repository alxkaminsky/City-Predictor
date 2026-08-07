import pandas as pd
import numpy as np
import re

CITIES = ['Dubai', 'New York City', 'Paris', 'Rio de Janeiro']

STOPWORDS = set((
    "the a an and or of to in on is are was for it its this that there here with you your we our my me but"
    "i he she they them his her at as be by from have has had do does not no all any can come came go "
    "went made make where when what which who how why than then so just like more most some want").split())


def mine_words(quotes, top_n=250):
    """
    Mine top 250 words from quotes
    """
    words = {}
    for q in quotes:
        text = str(q).lower()
        for w in {w for w in re.findall(r'[a-z]{3,}', text) if w not in STOPWORDS}:
            words.setdefault(w, 0)
            words[w] += 1
    return [k for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)[:top_n]]


def softmax(Z):
    # Done for numerical stability
     Z = Z - np.max(Z, axis=1, keepdims=True)
     expZ = np.exp(Z)
     return expZ / np.sum(expZ, axis=1, keepdims=True)

def grad(W, T, X, lam=0.0):
    N = X.shape[0]
    return np.transpose(1/N * (softmax(X @ np.transpose(W)) - T)) @ X + lam * W


def fix_quote(s):
    """
    Repair corrupted quotes
    """
    raw = bytearray()
    for ch in s:
        try:
            raw += ch.encode('cp1252')
        except UnicodeEncodeError:
            raw += ch.encode('latin-1', errors='ignore')
    return raw.decode('utf-8', errors='replace')


def transform(data, vocab, stats=None):
    """
    Build the final feature matrix used in sample.ipynb
    """
    feature_cols = [c for c in data.columns if c not in ('id', 'Quote', 'Label', 'Relatable', 'Company')]
    company = data['Company'].fillna('').astype(str)
    expanded_com = np.array([['Partner' in entry, 'Friends' in entry, 'Siblings' in entry, 'Co-worker' in entry]
                             for entry in company], dtype='float64')
    expanded_rel = np.array([[float(v) if (v := entry.split("=>")[-1].strip()) else np.nan
                              for entry in row.split(',')] for row in data['Relatable'].astype(str)])
    numeric = data[feature_cols].apply(pd.to_numeric, errors='coerce').astype('float64').to_numpy()

    scaled = np.hstack([numeric, expanded_rel])
    if stats is None:
        mean = np.nanmean(scaled, axis=0)
        std = np.nanstd(scaled, axis=0)
        std[std == 0] = 1.0
        stats = (mean, std)
    mean, std = stats
    scaled = (scaled - mean) / std

    quotes = data['Quote'].fillna('').astype(str).map(fix_quote).str.lower()
    words = np.array([[1.0 if w in q else 0.0 for w in vocab] for q in quotes])

    X = np.hstack([scaled, expanded_com, words])
    return X, feature_cols, stats


def one_hot(labels):
    """
    Turn an array of city-name strings into an (N, 4) one-hot target matrix, using the
    fixed CITIES order.
    """
    T = np.zeros((len(labels), len(CITIES)))
    for j, city in enumerate(CITIES):
        T[:, j] = (labels == city)
    return T


def fit(num_iter=1000, alpha=1.0, lam=0.001):
    dataset = pd.read_csv("cleaned_dataset.csv")

    vocab = mine_words(dataset['Quote'])
    X, feature_cols, stats = transform(dataset, vocab)
    t = dataset['Label'].to_numpy()

    keep = ~np.isnan(X).any(axis=1)
    X, t = X[keep], t[keep]
    T = one_hot(t)

    W = np.zeros((len(CITIES), X.shape[1]))

    for i in range(num_iter):
        W = W - alpha * grad(W, T, X, lam)

    return W, vocab, stats

if __name__ == "__main__":
    print(fit())
