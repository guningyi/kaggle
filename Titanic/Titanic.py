# /usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb



def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    #print '%s正确率：%.3f%%' % (tip, acc_rate)
    return acc_rate


# data: raw data path
# isTrainning: the flag for the function to decide whether to trainning the data.
# return value:
def data_process(data, isTrainning):
    data = pd.read_csv(data)  # read the data file

    # process the sex data
    # use 1 to represetate the Male
    # use 0 to represetate the female
    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(np.int)


    # process the Fare, use the median value to complete the missing value
    if len(data.Fare[data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0,3):
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0,3):
            data.loc[(data.Pclass == f+1) & (data.Fare.isnull()), 'Fare'] = fare[f]

    # process the Age value.
    # use the Random Forest Regressor
    # use the flag

    if isTrainning:
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]

        # print age_exist
        # actually, x was the data set for predict the Age.
        # here, Age is the output, so we define it as y. y means it is a label.
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]

        # sub module numbers:1000
        rfr = RandomForestRegressor(n_estimators=1000)

        # trainning
        rfr.fit(x,y)

        # predict
        age_hat = rfr.predict(age_null.values[:, 1:])

        #print age_hat
        # fill the age.
        data.loc[(data.Age.isnull()), 'Age'] = age_hat

    else:
        # if is not trainning, then it will not the Survived data.
        # that's why we need to build the different models.
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]

        # process the age_exist
        # print np.isnan(age_exist[['Fare']]).any()

        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]

        # sub module numbers:1000
        rfr = RandomForestRegressor(n_estimators=1000)


        # trainning
        rfr.fit(x,y)

        # predict
        # use the 'Fare', 'Parch', 'SibSp', 'Pclass' to predict the Age.
        age_hat = rfr.predict(age_null.values[:, 1:])

        # fill the age.
        data.loc[(data.Age.isnull()), 'Age'] = age_hat

    # the data of starting city also missing
    # I think it was very important for the prediction
    # So I should to repair these data
    # put the 'S' to the missing postion of Embarked.
    data.loc[(data.Embarked.isnull()),'Embarked'] = 'S'
    # get the column value contained in 'Embarked'
    embarked_data = pd.get_dummies(data.Embarked)
    # print embarked_data
    # use these column value and 'Embarked_' to make a new string.
    # Embarked_C ,for example
    # Embarked_C will be new column name.
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))


    # put these new coulmns into the data.
    # we are create the new features now.
    data = pd.concat([data, embarked_data], axis=1)

    #print data

    #print data.describe()

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    # 思考：这样做，其实发生了什么？
    # if isTrainning:
    #     x = np.tile(x, (5, 1))
    #     y = np.tile(y, (5, ))
    if isTrainning:
        return x, y
    return x, data['PassengerId']

if __name__ == '__main__':
        x, y = data_process('train.csv', True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

        rfc = RandomForestClassifier(n_estimators=10000)
        rfc.fit(x_train, y_train)

        y_hat = rfc.predict(x_test)

        rfc_rate = show_accuracy(y_hat, y_test, '随机森林 ')

        # XGBoost
        data_train = xgb.DMatrix(x_train, label=y_train)
        data_test = xgb.DMatrix(x_test, label=y_test)

        watch_list = [(data_test, 'eval'), (data_train, 'train')]
        param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}

        bst = xgb.train(param, data_train, num_boost_round=1000, evals=watch_list)
        y_hat = bst.predict(data_test)

        y_hat[y_hat > 0.5] = 1
        y_hat[~(y_hat > 0.5)] = 0
        xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')

        # print 'Logistic回归：%.3f%%' % lr_rate
        print '随机森林：%.3f%%' % rfc_rate
        print 'XGBoost：%.3f%%' % xgb_rate



        #STEP 2 predict the test data

        x, y = data_process('test.csv', False)
        y_hat = rfc.predict(x)

        y_hat[y_hat > 0.5] = 1
        y_hat[~(y_hat > 0.5)] = 0


        #result1 = rfc.predict(x)

        #passengerId = test_data['PassengerId']
        passengerId = y
        survived = pd.Series(y_hat)
        print len(x)
        #
        df = pd.DataFrame({'PassengerId':passengerId, 'Survived':survived})
        #
        # print df
        #
        # # 最后提交的格式必须是和gender_submission.csv一样
        df.iloc[0:418].to_csv('Titanic.csv', index='false')
