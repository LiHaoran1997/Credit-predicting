import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import  pandas as pd
file_path = 'data.xlsx'
data = pd.read_excel(file_path,sheet_name='cpu')
attribute_list = ['cpu', 'memory']  # 通过这两个属性来预测
df=data.iloc[:, 1:4]
x_data='cpu'
y_data='latency'
X = df.loc[:, attribute_list].values
X1 = df.loc[:, 'cpu'].values
X2 = df.loc[:, 'memory'].values
y = df.loc[:, y_data].values
# profile = pandas_profiling.ProfileReport(df)
# profile.to_file("output_file.html")
sns.scatterplot(x=x_data,y=y_data,data=df)
plt.show()
