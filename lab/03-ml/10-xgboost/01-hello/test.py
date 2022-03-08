from sklearn.datasets import load_boston
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

data['PRICE'] = boston.target

X, y = data.iloc[:, :-1], data.iloc[:, -1]
# data_dmatrix = xgb.DMatrix(data=X, label=y)
# print('data_dmatrix = ', data_dmatrix)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(object='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth=5, alpha=10, n_estimators=10)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
#data.info()
# print(data.describe())
#print(data.head())
#print('hello')
