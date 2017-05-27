import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def main():
    data = pandas.read_csv('train.csv',index_col = 'PassengerId')
    data_to_let = data.loc[:, ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin']]
    data2 = data_to_let.dropna()
    SexN = data2.Sex.factorize()
    data2['SexN'] = SexN[0]
    X = data2.loc[:, ['Pclass','SexN','Age','SibSp','Parch','Fare','Cabin']]
    y = data2['Survived']
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, y)



    test_data = pandas.read_csv('test.csv', index_col='PassengerId')
    test_data_to_let = test_data.loc[:, ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin']]
    test_data2 =  test_data_to_let.dropna()
    test_SexN = test_data2.Sex.factorize()
    test_data2['SexN'] = test_SexN[0]
    X = test_data2.loc[:, ['Pclass','SexN','Age','SibSp','Parch','Fare','Cabin']]
    res = clf.predict(X)
main()