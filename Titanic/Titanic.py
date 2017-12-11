# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


if __name__ == "__main__":
    path = 'train.csv'


    # STEP 1 : Data Cleaning && Data Preparing

    # pclass Age sibsp parch
    x = pd.read_csv(path, header=None, usecols=[2,4,5,6,7])
    # survived
    y = pd.read_csv(path, header=None, usecols=[1])

    lr = Pipeline([('sc', StandardScaler()),
                   ('clf', LogisticRegression())])

    sex=['male', 'female']
    for i, type in enumerate(sex):
        x.set_value(x[4] == type, 4, i)

    x.fillna(value=25)

    x = x.values.astype(np.int)
    y = y.values.astype(np.int)


    #STEP 2: Model Building
    lr.fit(x, y.ravel())
    y_hat = lr.predict(x)
    y_hat_prob = lr.predict_proba(x)
    np.set_printoptions(suppress=True)


    print 'y_hat = \n', y_hat
    print 'y_hat_prob = \n', y_hat_prob
    print u'准确度：%.2f%%' % (100 * np.mean(y_hat == y.ravel()))




