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

# print(index_corr[index_corr<-0.6])

data = data.iloc[:, [6, 11, 26, 33, 39, 40, 53, 67, 73, 81, 82, 83]]
data = pd.concat([data, y], axis=1)
# sns.pairplot(data, size=2.0)
# plt.show()

overallqual = data.loc[:, ['OverallQual']].values
exterqual = data.loc[:, ['ExterQual']].values
grlivarea = data.loc[:, ['GrLivArea']].values
kitchenqual = data.loc[:, ['KitchenQual']].values
garagecars = data.loc[:, ['GarageCars']].values
garagearea = data.loc[:, ['GarageArea']].values
simploverallqual = data.loc[:, ['SimplOverallQual']].values
simplexterqual = data.loc[:, ['SimplExterQual']].values
garagescore = data.loc[:, ['GarageScore']].values
totalbath = data.loc[:, ['TotalBath']].values
allse = data.loc[:, ['AllSF']].values
allflrssf = data.loc[:, ['AllFlrsSF']].values

from sklearn.preprocessing import PolynomialFeatures
quad = PolynomialFeatures(degree=2)    # 2次の多項式基底を生成
overallqual_quad = quad.fit_transform(overallqual) # 生成した基底関数で変数変換を実行

grlivarea_quad = quad.fit_transform(grlivarea) # 生成した基底関数で変数変換を実行

allse_quad = quad.fit_transform(allse) # 生成した基底関数で変数変換を実行

allflrssf_quad = quad.fit_transform(allflrssf) # 生成した基底関数で変数変換を実行

data_quad = np.hstack((overallqual_quad, exterqual, grlivarea_quad, kitchenqual, garagecars, simploverallqual, simplexterqual, allse_quad, allflrssf_quad, garagearea, garagescore, totalbath))

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
print('train: %.3f' % adjusted(model_quad.score(data_quad_train, y_train), len(y_train), 16))
#train: 0.837
print('test : %.3f' % adjusted(model_quad.score(data_quad_test, y_test), len(y_test), 16))
#test : 0.745
print('')

from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10)
model_ridge.fit(data_quad_train, y_train)
print(model_ridge.score(data_quad_test, y_test))
#0.767152166514

from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0)
model_lasso.fit(data_quad_train, y_train)
print(model_lasso.score(data_quad_test, y_test))
#0.757796747608

from sklearn.linear_model import ElasticNet
model_en= ElasticNet(alpha=0.1, l1_ratio=0.9)
model_en.fit(data_quad_train, y_train)
print(model_en.score(data_quad_test, y_test))
#0.754870188119
