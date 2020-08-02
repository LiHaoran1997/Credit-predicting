from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import  pandas as pd

fig = plt.figure()
ax = Axes3D(fig)
file_path = 'data.xlsx'
data = pd.read_excel(file_path,sheet_name='cpu')
attribute_list = ['cpu', 'memory']  # 通过这两个属性来预测
y_data='latency'
df=data.iloc[:, 1:11]
X = df.loc[:, attribute_list].values
X1 = df.loc[:, 'cpu'].values
X2 = df.loc[:, 'memory'].values
y = df.loc[:, y_data].values
ax.set_xlabel('cpu')
ax.set_ylabel('memory')
ax.set_zlabel(y_data)
ax.scatter(X1, X2, y)
plt.show()
