import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier

#训练数据
data_train = np.loadtxt('train_standard.csv', delimiter=',') #数据类型默认是float
x_train=data_train[:,1:] #array下标从0开始
y_train=data_train[:,0].astype(int) #由float转为int型

#测试数据
data_test=np.loadtxt("test_standard.csv",delimiter=',')
x_test=data_test[:,1:]
id_test=data_test[:,0].astype(int)

clf=GradientBoostingClassifier(n_estimators=300,learning_rate=0.1,max_depth=2,min_samples_split=200,min_samples_leaf=6)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
df=DataFrame({"PassengerId":id_test,"Survived":y_predict})
df.to_csv("result.csv",index=False)

