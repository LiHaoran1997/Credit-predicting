import warnings

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics._scorer import metric
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import median_absolute_error, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def read_data():
    '''
        读取数据集
    :return: 输入值和标签
    '''
    file_path = 'train.xlsx'   #路径
    data = pd.read_excel(file_path,sheet_name='contest_basic_train')  #读取
    df=data.iloc[:, 1:8]#1-8列
    y = data.iloc[:, 9].values
    # profile = pandas_profiling.ProfileReport(df)
    # profile.to_file("output_file.html")
    return df,y


def try_different_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train,y_train)
    expected = y_test
    predicted = clf.predict(X_test)
    # 评价预测的准确性
    print('解释方差(越接近1越好)：', accuracy_score(expected, predicted))  # 解释方差，越接近1越好
    print('r2_score(越接近1越好)：', r2_score(expected, predicted))  # r2 score 满分1，越接近1越好
    print('平均绝对误差(越小越好)：', mean_absolute_error(expected, predicted))  # 平均绝对误差，越小越好
    print('均方误差(越小越好)：', mean_squared_error(expected, predicted))  # 均方误差，越小越好
    print('中值绝对误差(越小越好)：', median_absolute_error(expected, predicted))  # 中值绝对误差，越小越好
    score = clf.score(X_test, y_test)
    print(score)
    # plt.figure()
    # X1=X_test[:,0]
    # plt.scatter(X1, y_test, c='green', label='true value')
    # plt.scatter(X1, predicted, c='red', label='predict value')
    # plt.title('score: %f' % score)
    # plt.legend()
    # plt.show()

def try_different_model2(clf,X_train, X_test, y_train, y_test):
    # 评价预测的准确性
    warnings.filterwarnings('ignore')
    # predicted = cross_val_predict(clf, X, y, cv=5)
    clf.fit(X_train,y_train)
    #最佳模型
    print("在测试集上准确率：", clf.score(X_test, y_test))
    print("在交叉验证当中最好的结果：", clf.best_score_)
    # # 0.390684110971 并没有使用测试集。(验证集是从训练集中分的)
    print("选择最好的模型是：", clf.best_estimator_)
    expected = y_train
    predicted = clf.predict(X_train)

    # 评价预测的准确性
    # print("-------------------训练集中情况--------------------------------")
    # print(confusion_matrix(expected, predicted))
    # target_names = ['0', '1']
    # print(classification_report(expected, predicted, target_names=target_names))

    print("-------------------测试集中情况--------------------------------")
    expected = y_test
    predicted = clf.predict(X_test)
    print(confusion_matrix(expected, predicted))
    target_names = ['0', '1']
    print(classification_report(expected, predicted, target_names=target_names))
    fpr, tpr, thresholds = roc_curve(expected,predicted)  # 计算fpr,tpr,thresholds
    print("AUC(越接近1越好):"+str(roc_auc_score(expected,predicted)))
    KS_max = 0
    best_thr = 0
    for i in range(len(fpr)):
        if (i == 0):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]
        elif (tpr[i] - fpr[i] > KS_max):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]

    print('最大KS为：', KS_max)
    print('最佳阈值为：', best_thr)

def main():
    X,Y = read_data()
    # print(X)
    # print(Y)
    # scaler = StandardScaler()  # 标准化转换
    # scaler.fit(X)  # 训练标准化对象
    # X = scaler.transform(X)  # 转换数据集
    X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.2);  # 百分之20测试
    # print('\n------使用线性回归-------')
    # clf = linear_model.LinearRegression()  # 线性回归
    # # clf.fit(X,Y)clf
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    # print('系数为:',clf.coef_)
    # #xjx


    print('\n------使用logistic回归-------')
    lr_param = {
        'C': [0.05, 0.1, 0.5, 1],
        'penalty': ['l2'],
    }
    lr = linear_model.LogisticRegression()  # 逻辑回归
    lr_grid = GridSearchCV(lr, lr_param, cv=5, n_jobs=-1)
    try_different_model2(lr_grid,X_train, X_test, y_train, y_test)


    print('\n------使用KNN-------')
    param = {"n_neighbors": [3, 5, 10]}
    clf = GridSearchCV(neighbors.KNeighborsClassifier(),param_grid=param,cv=5)  # 使用KNN
    try_different_model2(clf,X_train, X_test, y_train, y_test)

    print('\n------使用支持向量机-------')
    parameters = {
        'C': [1, 2, 4],
        'gamma': [0.125, 0.25, 0.5, 1, 2, 4]
    }
    clf = GridSearchCV(svm.SVC(),param_grid=parameters,cv=5)  # 支持向量机
    try_different_model2(clf,X_train, X_test, y_train, y_test)

    print('\n-------用决策树-------')
    clf = tree.DecisionTreeClassifier(random_state=80)  # 决策树回归
    params = {'max_depth':range(1,21),'criterion':np.array(['entropy','gini'])}
    grid = GridSearchCV(clf, param_grid=params,cv=5)
    try_different_model2(grid,X_train, X_test, y_train, y_test)
    #
    print('\n-------用随机森林-------')
    clf = ensemble.RandomForestClassifier(n_estimators=8, random_state=5, max_depth=6, min_samples_split=2)  # 随机森林
    param_grid ={'n_estimators':[3,5,8,10,14], 'random_state':[2,3,5,7,9],'max_depth':[5,6,8,9,10,15],'min_samples_split':[2,3,4,5,6]},
    grid = GridSearchCV(clf, param_grid, cv=5)
    try_different_model2(grid,X_train, X_test, y_train, y_test)


    print('\n-------用GBDT-------')
    clf = ensemble.GradientBoostingClassifier()  # GBDT
    param_test1 = {'n_estimators': range(20, 81, 10)}
    grid = GridSearchCV(clf,param_test1, cv=5,  n_jobs=-1)
    try_different_model2(grid,X_train, X_test, y_train, y_test)

    #
    print('\n-------用AdaBoost-------')
    clf = ensemble.AdaBoostClassifier()  # 用AdaBoost          #
    parameter = {'n_estimators': range(20, 81, 10),"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    grid = GridSearchCV(clf,parameter, cv=5,  n_jobs=-1)
    # clf.fit(X,Y)
    try_different_model2(grid,X_train, X_test, y_train, y_test)

    print('\n-------用XgBoost-------')
    parameters = {'n_estimators': [3, 5, 8, 10, 14], 'learning_rate': [0.1, 0.2, 0.25, 0.3, 0.35, 0.4],
                  'max_depth': [5, 10, 15, 20, 25], 'gamma': [6, 9, 12, 13, 15], 'seed': [500, 1000, 1500]}
    clf = xgb.XGBClassifier(n_estimators=8,learning_rate= 0.25, max_depth=20,subsample=1,gamma=13, seed=1000,num_class=1)  # 用AdaBoost
    grid = GridSearchCV(clf,parameters, cv=5,  n_jobs=-1)
    try_different_model2(grid,X_train, X_test, y_train, y_test)

    #
    print('\n-------用LightGBM-------')
    clf = lgb.LGBMClassifier()  # 用loghtGBM
    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }
    grid = GridSearchCV(clf,param_grid, cv=5,  n_jobs=-1)
    try_different_model2(grid,X_train, X_test, y_train, y_test)
if __name__ == '__main__':
    main()

