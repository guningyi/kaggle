# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# compare two list
# if  equality, return 0
def compare(list1, list2):
    if len(list1) == len(list2):
        for i in range(0, len(list1)):
            if (list1[i][0] == list2[i][0] and list1[i][1] == list2[i][1] and list1[i][2] == list2[i][2]):
                #print i, list1[i][0] , list2[i][0]
                pass
            else:
                print 'difference found:\n', i


if __name__ == '__main__':
    if __name__ == "__main__":
        path = 'train.csv'
        test_path = 'test.csv'


        # STEP 1 : Data Cleaning && Data Preparing

        # pclass Sex Age sibsp parch
        x = pd.read_csv(path, header=None, usecols=[2,4,5,6,7])
        # survived
        y = pd.read_csv(path, header=None, usecols=[1])

        lr = Pipeline([('sc', StandardScaler()),
                       ('clf', LogisticRegression())])

        sex=['male', 'female']
        for i, type in enumerate(sex):
            x.set_value(x[4] == type, 4, i)

        # 将Age中缺失的部分简单用25来统一替换,但这样做的效果不好.
        # x.loc[(x[5].isnull())] = 25
        # print x

        x = x.values.astype(np.int)
        y = y.values.astype(np.int)


        #STEP 2: Model Building
        lr.fit(x, y.ravel())

        # predict是强分类器，对于每个样本，它要么输出0，要么输出1
        y_hat = lr.predict(x)
        #print y_hat

        # predict_proba是弱分类器返回的概率。它不像强分类器只给出0或者1.
        # 它给出的概率是 为0的概率0.43348191，1的概率为0.56651809.
        y_hat_prob = lr.predict_proba(x)
        #print y_hat_prob
        np.set_printoptions(suppress=True)


        #print 'y_hat = \n', y_hat
        #print 'y_hat_prob = \n', y_hat_prob
        #print u'准确度：%.2f%%' % (100 * np.mean(y_hat == y.ravel()))

        # read the test set
        test_data = pd.read_csv(test_path)  # 测试文件路径
        test_data['Sex'] = test_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
        #print 'test_data:Sex : \n', test_data['Sex']

        # pclass sex age sibsp parch
        data_for_predict = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

        data_for_predict.loc[(test_data.Age.isnull()), 'Age'] = 25

        #print  data_for_predict
        data_for_predict.values.astype(np.int)
        result1 = lr.predict(data_for_predict)

        passengerId = test_data['PassengerId']
        survived = pd.Series(result1)

        df = pd.DataFrame({'PassengerId':passengerId, 'Survived':survived})

        print df

        df.to_csv('Titanic.csv', index='false')
