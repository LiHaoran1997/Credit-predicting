import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sqlalchemy import create_engine
import pymysql

def read_data():
    '''
        读取数据集
    :return: 输入值和标签
    '''
#！！！！！！！！！根据你的数据库设置更改
    # MySQL的用户：root, 密码:1234qwer, 端口：3306,数据库：nacos_config
    # 端口一般都是3306不用管，用户密码就是你mysql的登陆密码，数据库是你那两个名称
    engine = create_engine('mysql+pymysql://root:1234qwer@localhost:3306/nacos_config')

 # ！！！！！！！！！根据数据库表操作
    # 查询语句，选出employee表中的所有数据
    sql = '''
          select * from users;
          '''
    # read_sql_query的两个参数: sql语句， 数据库连接
    df = pd.read_sql_query(sql, engine)
# ！！！！！！！！！自变量，需要更改名称
    attribute_list = ['cpu', 'memory']  # 自变量！！  (自己改一下)
    X = df.loc[:, attribute_list].values
# ！！！！！！！！！因变量，需要更改名称
    y = df.loc[:, 'y'].values

    return X,y

print(read_data())
#处理数据（关键）
X,Y = read_data()#导入数据
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.2)  # 百分之20测试
# print('\n------使用线性回归-------')
clf = linear_model.LinearRegression()  # 线性回归
clf.fit(X_train, y_train)
expected = y_test
predicted = clf.predict(X_test)



# 评价预测的准确性
print('解释方差(越接近1越好)：', explained_variance_score(expected, predicted))  # 解释方差，越接近1越好
print('r2_score(越接近1越好)：', r2_score(expected, predicted))  # r2 score 满分1，越接近1越好
print('平均绝对误差(越小越好)：', mean_absolute_error(expected, predicted))  # 平均绝对误差，越小越好
print('均方误差(越小越好)：', mean_squared_error(expected, predicted))  # 均方误差，越小越好
print('中值绝对误差(越小越好)：', median_absolute_error(expected, predicted))  # 中值绝对误差，越小越好
score = clf.score(X_test, y_test)
print('系数为:',clf.coef_)