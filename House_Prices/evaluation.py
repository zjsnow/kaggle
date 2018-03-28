import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone


data_train = np.loadtxt('train_precessing.csv', delimiter=',') #数据类型默认是float
x=data_train[:,1:] #array下标从0开始
y=data_train[:,0]

#回归问题的评估指标是均方误差，即真实值与预测值的误差平方的平均值
def rmse(model,x,y):
    rmse=np.sqrt(-cross_val_score(model,x,y,scoring='neg_mean_squared_error',cv=5)) #输出的值越小越好
    print("%.5f  +/- %.5f" % (np.mean(rmse), np.std(rmse)))

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

#网格搜索最优的参数
#GBDT算法
# gbr=GradientBoostingRegressor(n_estimators=250,learning_rate=0.1,max_depth=2,min_samples_split=10,min_samples_leaf=7)
#XGBoost算法
# xgb=XGBRegressor(n_estimators=700,learning_rate=0.07,max_depth=2,subsample=0.7,colsample_bytree=0.7,n_jobs=-1)
#lightgbm算法
# lgb=LGBMRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,num_leaves=7,min_child_samples=3,colsample_bytree=0.9,subsample=0.7)
#ElasticNet算法，是将L1 and L2正则化结合起来的线性回归模型
# enet=ElasticNet(alpha=0.0035,l1_ratio=0.5) #alpha用来控制复杂度，越大，则控制过拟合的能力越强，l1_ratio是调整l1正则化和l2正则化比例的，默认0.5
# Lassos算法
# lasso=Lasso(alpha=0.0016)
#Ridge算法
# ridge=Ridge(alpha=45)
# SVM算法
# svm = SVR(C=5,gamma=0.0007)
#采用bagging的方法使用表现最好的ElasticNet做基分类器
# enet=ElasticNet(alpha=0.0035,l1_ratio=0.5)
# bagging=BaggingRegressor(enet,n_estimators=100)
# param_grid={'subsample':[0.6,0.7,0.8,0.9],'colsample_bytree':[0.7,0.8,0.9,1.0]}
# grid_search=GridSearchCV(lgb,param_grid=param_grid,scoring='neg_mean_squared_error',cv=5)
# grid_search.fit(x,y)
# print(np.sqrt(-grid_search.best_score_))
# print(grid_search.best_params_)
# print(grid_search.grid_scores_)

# #交叉验证
# #GBDT算法
gbr=GradientBoostingRegressor(n_estimators=250,learning_rate=0.1,max_depth=2,min_samples_split=10,min_samples_leaf=7)
# rmse(gbr,x,y)#0.11705  +/- 0.00765
# # XGBoost算法
xgb=XGBRegressor(n_estimators=700,learning_rate=0.07,max_depth=2,subsample=0.7,colsample_bytree=0.7,n_jobs=-1)
# rmse(xgb,x,y)#0.11156  +/- 0.00718
# # Lightgbm
# lgb=LGBMRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,num_leaves=7,min_child_samples=3,colsample_bytree=0.9,subsample=0.7)
# rmse(lgb,x,y)#0.11595  +/- 0.00727
# # ElasticNet算法
enet=ElasticNet(alpha=0.0035,l1_ratio=0.5)
# rmse(enet,x,y)#0.11150  +/- 0.00631
# # Lassos算法
# lasso=Lasso(alpha=0.0016)
# rmse(lasso,x,y)#0.11151  +/- 0.00631
# # Ridge算法
# ridge=Ridge(alpha=45)
# rmse(ridge,x,y)#0.11326  +/- 0.00686
# #svm算法
svm = SVR(C=5,gamma=0.0007)
# rmse(svm,x,y) #0.11206  +/- 0.00978

model_aver = AveragingModels(models=(xgb,enet,svm),weights=(0.35,0.45,0.2))
rmse(model_aver,x,y) #0.10835  +/- 0.00796