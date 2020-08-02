import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt

def read_data():
    '''
        读取数据集
    :return: 输入值和标签
    '''
    file_path = 'data.xlsx'
    data = pd.read_excel(file_path,sheet_name='cpu')
    attribute_list = ['name','cpu', 'memory']  # 通过这两个属性来预测
    df=data.iloc[:, 0:8]
    X = df.loc[:, attribute_list].values
    y = df.loc[:, 'latency'].values
    # profile = pandas_profiling.ProfileReport(df)
    # profile.to_file("output_file.html")
    return X,y


def try_different_model(clf, X_train, X_test, y_train, y_test):
    X1=X_train[:,0]
    X_train=X_train[:,1:]
    clf.fit(X_train,y_train)
    expected = y_test
    X2=X_test[:,0]
    X_test=X_test[:,1:]
    predicted = clf.predict(X_test)
    # 评价预测的准确性
    print('解释方差(越接近1越好)：', explained_variance_score(expected, predicted))  # 解释方差，越接近1越好
    print('r2_score(越接近1好)：', r2_score(expected, predicted))  # r2 score 满分1，越接近1越好
    print('平均绝对误差(越小越好)：', mean_absolute_error(expected, predicted))  # 平均绝对误差，越小越好
    print('均方误差(越小越好)：', mean_squared_error(expected, predicted))  # 均方误差，越小越好
    print('中值绝对误差(越小越好)：', median_absolute_error(expected, predicted))  # 中值绝对误差，越小越好
    score = clf.score(X_test, y_test)
    plt.figure()
    x = range(len(X2))
    plt.xticks(x, X2, rotation=45)
    plt.scatter(x, y_test, c='green', label='true value')
    plt.scatter(x, predicted, c='red', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


def main():
    X,Y = read_data()
    # scaler = StandardScaler()  # 标准化转换
    # scaler.fit(X)  # 训练标准化对象
    # X = scaler.transform(X)  # 转换数据集
    X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.2)  # 百分之20测试
    # print('\n------使用线性回归-------')
    # clf = linear_model.LinearRegression()  # 线性回归
    # # clf.fit(X,Y)
    #
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    # print('系数为:',clf.coef_)
    #
    # print('\n------使用岭回归-------')
    # model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
    # # clf.fit(X,Y)
    #
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    # # clf.fit(X,Y)
    #
    # print('系数为:', clf.coef_)
    #
    # print('\n------使用lasso回归-------')
    # clf = linear_model.Lasso()  # 线性回归
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    # print('系数为:', clf.coef_)
    #
    # print('\n------使用神经网络-------')
    # clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    #
    # print('\n------使用KNN-------')
    # clf = neighbors.KNeighborsRegressor()  # 使用KNN
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    #
    # print('\n------使用支持向量机-------')
    # clf = svm.SVR(kernel='linear')  # 支持向量机
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    # print('方程系数为：',clf.coef0)
    #
    #
    # print('\n-------用决策树-------')
    # clf = tree.DecisionTreeRegressor()  # 决策树回归
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    #
    #
    # print('\n-------ExtraTreeRegressor-------')
    # clf = tree.ExtraTreeRegressor()  # ExtraTreeRegressor回归
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)
    #
    # print('\n-------用随机森林-------')
    # clf = ensemble.RandomForestRegressor(n_estimators=20)  # 随机森林回归
    # # clf.fit(X,Y)
    # try_different_model(clf, X_train, X_test, y_train, y_test)

    print('\n-------用GBDT-------')
    clf = ensemble.GradientBoostingRegressor(n_estimators=100)  # GBDT回归
    # clf.fit(X,Y)
    try_different_model(clf, X_train, X_test, y_train, y_test)

    print('\n-------用AdaBoost-------')
    clf = ensemble.AdaBoostRegressor(n_estimators=50)  # 用AdaBoost回归
    # clf.fit(X,Y)
    try_different_model(clf, X_train, X_test, y_train, y_test)

    print('\n-------用Bagging-------')
    clf = ensemble.BaggingRegressor()  # 用Bagging回归
    # clf.fit(X,Y)
    try_different_model(clf, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()

