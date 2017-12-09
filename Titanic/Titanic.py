import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

if __name__ == "__main__":
    path = 'train.csv'

    data = pd.read_csv(path, header=None)
    print data



