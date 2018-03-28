import numpy as np
from pandas import DataFrame
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone

#使用bagging组合多个算法，定义了类AveragingModels
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights)

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.sum(self.weights * predictions, axis=1)

train_data= np.loadtxt('train_precessing.csv', delimiter=',')
x_train=train_data[:,1:]
y_train=train_data[:,0]

test_data=np.loadtxt('test_precessing.csv',delimiter=',')
x_test=test_data[:,1:]
id_test=test_data[:,0].astype(int)  #默认都是float型

#使用bagging组合多个算法，按照效果赋予不同的权重
xgb=XGBRegressor(n_estimators=700,learning_rate=0.07,max_depth=2,subsample=0.7,colsample_bytree=0.7,n_jobs=-1)
enet=ElasticNet(alpha=0.0035,l1_ratio=0.5)
svm = SVR(C=5,gamma=0.0007)
clf = AveragingModels(models=(xgb,enet,svm),weights=(0.35,0.45,0.2))
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
y_pred=np.exp(y_pred) #之前对saleprice做了log处理，所以这里需要做exp处理
df=DataFrame({'Id':id_test,'SalePrice':y_pred})
df.to_csv('result.csv',index=False)






