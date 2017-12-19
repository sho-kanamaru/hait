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
# print(index_corr[index_corr>0.6])
#6      0.790982
#11     0.673241
#26     0.695147
#33     0.659600
#39     0.640409
#40     0.623431
#53     0.662653
#67     0.615318
#73     0.624572
#81     0.631731
#82     0.758921
#83     0.700568

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
coef = lr.coef_.reshape(-1)
index_coef = Series(coef)
# print(index_coef[index_coef>100000000000])
#155    1.152026e+11
#161    1.799721e+11
#165    1.302850e+11
#243    1.405243e+11
#247    1.263925e+11

#print(index_coef[index_coef<-100000000000])
#114   -1.130689e+11
#118   -1.320249e+11
#181   -1.811188e+11
#185   -1.184221e+11
#200   -1.150144e+11

data1 = data.iloc[:, [6, 26, 82, 83]]
data2 = data.iloc[:, [11, 33, 39, 40, 53, 67, 73, 81, 155, 161, 165, 243, 247, 114, 118, 181, 185, 200]]
# data1 = pd.concat([data1, y], axis=1)
# data2 = pd.concat([data2, y], axis=1)
# sns.pairplot(data1, size=2.0)
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
quad = PolynomialFeatures(degree=2)    # 2次の多項式基底を生成
data1_quad = quad.fit_transform(data1) # 生成した基底関数で変数変換を実行

data_quad = np.hstack((data1_quad, data2))

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data_quad_std = ss.fit_transform(data_quad) # すべての説明変数を変換
y_std = ss.fit_transform(y)

from sklearn.cross_validation import train_test_split
data_quad_train, data_quad_test, y_train, y_test = train_test_split(data_quad_std, y_std, test_size = 0.3, random_state = 0)


# 線形回帰による学習
model_quad = LinearRegression()
model_quad.fit(data_quad_train, y_train)

def adjusted(score, n_sample, n_features):
    adjusted_score = 1 - (1 - score) * ((n_sample - 1) / (n_sample - n_features - 1))
    return adjusted_score

# 2次関数
print('model_quad')
print('train: %.3f' % adjusted(model_quad.score(data_quad_train, y_train), len(y_train), 21))
#train: 0.857
print('test : %.3f' % adjusted(model_quad.score(data_quad_test, y_test), len(y_test), 21))
#test : 0.744
print('')

from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10)
model_ridge.fit(data_quad_train, y_train)
print(model_ridge.score(data_quad_test, y_test))
#0.762914503976

from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0)
model_lasso.fit(data_quad_train, y_train)
print(model_lasso.score(data_quad_test, y_test))
#0.76959557802

from sklearn.linear_model import ElasticNet
model_en= ElasticNet(alpha=0.1, l1_ratio=0.9)
model_en.fit(data_quad_train, y_train)
print(model_en.score(data_quad_test, y_test))
#0.755300539527
