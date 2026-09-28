# City Predictor

Predicts which of four cities (Dubai, New York City, Paris, Rio de Janeiro) a survey respondent is describing, from a mix of numeric ratings, multi-select answers, and a free-text quote. The final model is multinomial logistic regression written from scratch in NumPy, reaching **94% accuracy on a held-out test set** (4 balanced classes, so chance is 25%).

## Results

| Model | Features | Test accuracy |
|---|---|---|
| Decision tree (scikit-learn) | numeric only | 83% |
| Logistic regression (scikit-learn) | numeric only | 87% |
| MLP, 1 hidden layer of 128 (scikit-learn) | numeric only | 90% |
| Logistic regression (scikit-learn) | numeric + 250 mined words | 93% |
| **Logistic regression (from scratch, NumPy)** | **numeric + 250 mined words** | **94%** |

Dataset: 1,462 responses, split 70/15/15 into train/validation/test (1,023 / 219 / 220). Every hyperparameter was chosen on the validation set, and the test set was scored once at the end. With 220 test points, expect roughly ±3% sampling noise on that number.

## How it works

**Features.** Numeric survey answers and six "relatability" scores are standardized using training-set statistics. The multi-select "who would you bring" answer becomes four binary flags. The free-text quote is turned into 250 binary word features, with the vocabulary mined from training quotes only (most frequent words of 3+ letters, stopwords removed, substring matching so "skyscraper" also catches "skyscrapers"). A byte-level repair step fixes quotes that were saved with the wrong text encoding.

**Model.** Softmax regression trained by full-batch gradient descent on cross-entropy loss with L2 regularization. The implementation (`train.py`) uses no ML libraries: a numerically stable softmax, an analytic gradient, and a plain update loop.

**Tuning.** Learning rate, regularization strength, iteration count, and vocabulary size were each swept on the validation set. Two findings worth noting:

- Standardizing features mattered more than anything else. On raw features, validation accuracy fell from 92% to 73% as the learning rate rose; standardized, it peaked at 94% with `alpha = 1.0`.
- Without L2 regularization, validation accuracy dropped from 94% to 90% as training ran longer, because the model slowly memorized the sparse word features. A small penalty (`lam = 0.001`) made accuracy stable regardless of training length.

**Error analysis.** The numeric-only models mostly confused New York City with Paris, and Rio de Janeiro with Dubai. The misclassified responses usually had a telling quote ("the big statue of Jesus Christ..."), which is what motivated adding text features. They cut test errors from 21 (MLP) to 13 (final model).

The full experimental story, with plots and confusion matrices, is in [`sample.ipynb`](sample.ipynb).

## Repository layout

```
train.py              from-scratch model: feature pipeline, softmax, gradient, training loop
predict.py            predict_all(csv_path) -> list of city names
sample.ipynb          exploration, model comparison, tuning, and error analysis
data/                 cleaned survey data
```

## Running it

Requires Python 3.9+ with NumPy and pandas (the notebook also uses scikit-learn, matplotlib, and seaborn).

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

```python
from predict import predict_all
predict_all("data/cleaned_dataset.csv")   # any CSV with the same columns
```

`predict_all` retrains on the full dataset each call (training takes a few seconds), then predicts every row of the given file.
