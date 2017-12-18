# モジュールのインポート
import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline

data = pd.read_csv('house_price.csv')
y = pd.read_csv('y.csv')

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)

#テストデータへの正答率を参考にして削除する特徴量を決めるため，改めてトレーニングデータ(X_train_std, y_train)を分割する.
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
dimension = X_train2.shape[1]
indices = tuple(range(dimension))

from sklearn.metrics import accuracy_score
from itertools import combinations

lr.fit(X_train2, y_train2)

best_scores = []
best_subsets = []

k_features = 10

while dimension > k_features:
    scores = []
    subsets = []

    for c in combinations(indices, r=dimension-1):
        lr.fit(X_train2.iloc[:, c], y_train2)
        score = lr.score(X_test2.iloc[:, c], y_test2)
        # 選択した特徴量のスコアを格納
        scores.append(score)
        # 選択した特徴量のインデックスを格納
        subsets.append(c)

    best = np.argmax(scores)
    indices = subsets[best]
    best_subsets.append(indices)
    best_scores.append(np.max(scores))
    dimension -= 1

#204番目の組み合わせが最もスコアが高い(89%)
best_scores[np.argmax(best_scores)]
index = best_subsets[np.argmax(best_scores)]

lr.fit(X_train.iloc[:, index], y_train)
print('トレーニングデータの正答率: ', lr.score(X_train.iloc[:, index], y_train))
print('テストデータの正答率: ', lr.score(X_test.iloc[:, index], y_test))
#トレーニングデータの正答率:  0.877990776717
#テストデータの正答率:  0.492281327403

#Ridge回帰モデル
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10)
model_ridge.fit(X_train.iloc[:, index], y_train)
print('トレーニングデータの正答率: ', model_ridge.score(X_train.iloc[:, index], y_train))
print('テストデータの正答率: ', model_ridge.score(X_test.iloc[:, index], y_test))
#トレーニングデータの正答率:  0.845238640917
#テストデータの正答率:  0.771072478292

#LASSOモデル
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=100)
model_lasso.fit(X_train.iloc[:, index], y_train)
print('トレーニングデータの正答率: ', model_lasso.score(X_train.iloc[:, index], y_train))
print('テストデータの正答率: ', model_lasso.score(X_test.iloc[:, index], y_test))
#トレーニングデータの正答率:  0.848611746363
#テストデータの正答率:  0.774935790862

#Elastic Netモデル
from sklearn.linear_model import ElasticNet
model_en= ElasticNet(alpha=0.1, l1_ratio=0.9)
model_en.fit(X_train.iloc[:, index], y_train)
print('トレーニングデータの正答率: ', model_en.score(X_train.iloc[:, index], y_train))
print('テストデータの正答率: ', model_en.score(X_test.iloc[:, index], y_test))
#トレーニングデータの正答率:  0.844991127533
#テストデータの正答率:  0.771002213313


