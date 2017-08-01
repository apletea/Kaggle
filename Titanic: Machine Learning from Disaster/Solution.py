import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

def import_data(str,parametr):
    return pandas.read_csv(str, index_col = parametr)

def test():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    importances = clf.feature_importances_
    print(importances)




def main():
    data = import_data('train.csv','PassengerId')
    data_to_let = data.loc[ : ,['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
    data2 = data_to_let.dropna()
    SexN = data2.Sex.factorize()
    data2.loc[:,'SexN'] = SexN[0]
    X = data2.loc[ :, ['Pclass', 'Fare' , 'Age', 'SexN']]
    y = data2['Survived']
    clf = DecisionTreeClassifier()
    clf.random_state = 241
    clf.fit(X,y)

    test_data = import_data('test.csv','PassengerId')
    test_to_let = test_data.loc[ : ,['Pclass', 'Fare', 'Age', 'Sex']]
    SexN = test_to_let.Sex.factorize()
    test_to_let.loc[:, 'SexN'] = SexN[0]
    X = test_to_let.loc[ :, ['Pclass', 'Fare' , 'Age', 'SexN']]
    print(X)

    res = clf.predict(X)
    print(res)
    print(len(res))
    print(test_data['Fare'])

main()






