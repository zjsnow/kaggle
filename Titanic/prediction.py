import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#训练数据
data_train = np.loadtxt('train_standard.csv', delimiter=',') #数据类型默认是float
x_train=data_train[:,1:] #array下标从0开始
y_train=data_train[:,0].astype(int) #由float转为int型

#测试数据
data_test=np.loadtxt("test_standard.csv",delimiter=',')
x_test=data_test[:,1:]
id_test=data_test[:,0].astype(int)

xgboost = XGBClassifier(n_estimators=200, learning_rate=0.2, max_depth=2, mim_child_weight=0.8, gamma=0.009,
                        colsample_bytree=0.7, subsample=0.9)  # accuracy:0.858 +/- 0.031
gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=2, min_samples_split=200,
                                  min_samples_leaf=6)  # 0.847
rf = RandomForestClassifier(n_estimators=500, min_samples_split=4, min_samples_leaf=2, n_jobs=-1)  # 0.83
lr = LogisticRegression(penalty='l2', C=0.1)  # 0.83
svm = SVC(C=10, gamma=0.01)  # 0.83
#采用投票的机制，给每个不同的模型分配权重
clf=VotingClassifier(estimators=[('xgboost',xgboost),('gbdt',gbdt),('rf',rf),('lr',lr),('svm',svm)],voting='hard',weights=[0.50,0.05,0.05,0.2,0.2]) #accuracy:0.860 +/- 0.032
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
df=DataFrame({"PassengerId":id_test,"Survived":y_predict})
df.to_csv("result.csv",index=False)

