# モジュールのインポート

import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline

data = pd.read_csv('house_price.csv')
y = pd.read_csv('y.csv')

from pandas import Series
df = pd.concat([data, y], axis=1)
corr = df.corr()
corr = corr[290:]
index_corr = Series(corr.values.reshape(-1))
# print(index_corr[index_corr>0.7])
#6      0.790982
#82     0.758921
#83     0.700568

data = data.iloc[:, [6, 82, 83]]
data = pd.concat([data, y], axis=1)
# sns.pairplot(data, size=2.0)
# plt.show()

overallqual = data.loc[:, ['OverallQual']].values
allse = data.loc[:, ['AllSF']].values
allflrssf = data.loc[:, ['AllFlrsSF']].values

from sklearn.preprocessing import PolynomialFeatures
quad = PolynomialFeatures(degree=2)    # 2次の多項式基底を生成
overallqual_quad = quad.fit_transform(overallqual) # 生成した基底関数で変数変換を実行

allse_quad = quad.fit_transform(allse) # 生成した基底関数で変数変換を実行

allflrssf_quad = quad.fit_transform(allflrssf) # 生成した基底関数で変数変換を実行

data_quad = np.hstack((overallqual_quad, allse_quad, allflrssf_quad))

from sklearn.cross_validation import train_test_split
data_quad_train, data_quad_test, _, _ = train_test_split(data_quad, y, test_size = 0.3, random_state = 0)

# 線形回帰による学習
model_quad = LinearRegression()
model_quad.fit(data_quad_train, y_train)

def adjusted(score, n_sample, n_features):
    adjusted_score = 1 - (1 - score) * ((n_sample - 1) / (n_sample - n_features - 1))
    return adjusted_score

# 2次関数
print('model_quad')
print('train: %.3f' % adjusted(model_quad.score(data_quad_train, y_train), len(y_train), 6))
#train: 0.797
print('test : %.3f' % adjusted(model_quad.score(data_quad_test, y_test), len(y_test), 6))
#test : 0.730
print('')

from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10)
model_ridge.fit(data_quad_train, y_train)
print(model_ridge.score(data_quad_test, y_test))
#0.748384357755

from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=200)
model_lasso.fit(data_quad_train, y_train)
print(model_lasso.score(data_quad_test, y_test))
#0.748449558239

from sklearn.linear_model import ElasticNet
model_en= ElasticNet(alpha=0.1, l1_ratio=0.9)
model_en.fit(data_quad_train, y_train)
print(model_en.score(data_quad_test, y_test))
#0.748382670579
