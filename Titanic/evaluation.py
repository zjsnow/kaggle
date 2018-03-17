import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import  StratifiedShuffleSplit,GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics

data_train = np.loadtxt('train_standard.csv', delimiter=',') #数据类型默认是float
x=data_train[:,1:] #array下标从0开始
y=data_train[:,0].astype(int) #由float转为int型

# #网格搜索，寻得最优参数组合
#随机森林
# clf=RandomForestClassifier(n_jobs=-1)
# param_grid={'clf__n_estimators':[500],'clf__min_samples_split':[2,4,6],'clf__min_samples_leaf':[1,2,3]} #注意标签加上clf__
# pipeline_rf=Pipeline([('clf',clf)])
# #StratifiedShuffleSplit是分层采样，n_split设置得到的train/test对的个数，test_size=0.2表示train:test=5:1采样，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_rf,param_grid=param_grid,scoring='accuracy',cv=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数组合n_estimators=500,min_samples_split=4,min_samples_leaf=2

#逻辑回归
# clf=LogisticRegression()
# param_grid={'clf__penalty':["l1","l2"],'clf__C':[0.01,0.1,1,10]} #注意标签加上clf__
# pipeline_lr=Pipeline([('clf',clf)])
# #StratifiedShuffleSplit是分层采样，n_split设置得到的train/test对的个数，test_size=0.2表示train:test=5:1采样，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_lr,param_grid=param_grid,scoring='accuracy',cv=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数组合penalty='l2',C=0.1

#svm
# clf=SVC()
# param_grid={'clf__C':[0.01,0.1,1,10],'clf__gamma':[0.001,0.01,0.1,1]} #注意标签加上clf__
# pipeline_svc=Pipeline([('clf',clf)])
# #StratifiedShuffleSplit是分层采样，n_split设置得到的train/test对的个数，test_size=0.2表示train:test=5:1采样，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_svc,param_grid=param_grid,scoring='accuracy',cv=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数组合C=10,gamma=0.01

#KNN
# clf=KNeighborsClassifier(weights='distance')
# param_grid={'clf__n_neighbors':[15,20,25,30,35,40,45,50,55,60]} #注意标签加上clf__
# pipeline_knn=Pipeline([('clf',clf)])
# #StratifiedShuffleSplit是分层采样，n_split设置得到的train/test对的个数，test_size=0.2表示train:test=5:1采样，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_knn,param_grid=param_grid,scoring='accuracy',cv=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数n_neighbors=35

#多层感知机
# clf=MLPClassifier(solver='lbfgs')
# param_grid={'clf__hidden_layer_sizes':[(3,),(4,),(5,)],'clf__alpha':[0.1,1,5,10,]} #注意标签加上clf__
# pipeline_knn=Pipeline([('clf',clf)])
# #StratifiedShuffleSplit是分层采样，n_split设置得到的train/test对的个数，test_size=0.2表示train:test=5:1采样，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_knn,param_grid=param_grid,scoring='accuracy',cv=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数solver='lbfgs',hidden_layer_sizes=5,alpha=1

#Adaboost算法
# clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
# param_grid={'clf__n_estimators':[100,200,300,400,500],'clf__learning_rate':[0.3,0.6,0.8,1]} #注意标签加上clf__
# pipeline_knn=Pipeline([('clf',clf)])
# #StratifiedShuffleSplit是分层采样，n_split设置得到的train/test对的个数，test_size=0.2表示train:test=5:1采样，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_knn,param_grid=param_grid,scoring='accuracy',cv=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数n_estimator=100,learning_rate=0.6

# GBDT算法
# clf=GradientBoostingClassifier(n_estimators=300,learning_rate=0.1,max_depth=2)
# param_grid={'clf__min_samples_split':range(100,201,25),'clf__min_samples_leaf':range(2,11,2)} #注意标签加上clf__
# pipeline_knn=Pipeline([('clf',clf)])
# #StratifiedKFold是分层采样进行的交叉验证，random_state=0是一个固定的数，表示运行多次得到的划分将是固定的
# grid_search=GridSearchCV(pipeline_knn,param_grid=param_grid,scoring='accuracy',cv=StratifiedKFold(n_splits=10,random_state=0))
# grid_search.fit(x,y) #运行网格搜索
# print(grid_search.grid_scores_) #所有参数下的结果
# print(grid_search.best_score_) #获得最好结果的评分
# print(grid_search.best_params_) #最好结果下的参数n_estimators=300,learning_rate=0.1,max_depth=2,min_samples_split=200,min_samples_leaf=6


#交叉验证
accuracy_list=[]
skf=StratifiedKFold(n_splits=10,random_state=0)
for train_index,test_index in skf.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=2, min_samples_split=200,min_samples_leaf=6)  # 0.847
    # clf=RandomForestClassifier(n_estimators=500,min_samples_split=4,min_samples_leaf=2,n_jobs=-1) #0.83
    # clf=LogisticRegression(penalty='l2',C=0.1) #0.83
    # clf=SVC(C=10,gamma=0.01) #0.83
    # clf=KNeighborsClassifier(n_neighbors=35,weights='distance') #0.81
    # clf=GaussianNB() #0.80
    # clf=MLPClassifier(solver='lbfgs',hidden_layer_sizes=5,alpha=5) #0.82
    # clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,learning_rate=0.6) #0.80
    clf.fit(x_train,y_train)
    score=clf.score(x_test,y_test)
    accuracy_list.append(score)
print('accuracy:%.3f +/- %.3f' %(np.mean(accuracy_list),np.std(accuracy_list)))


#画混淆矩阵
# def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt ='d'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=0) #训练集样本划分，得到训练集和测试集
# for train_index,test_index in sss.split(x,y):
#     x_train,y_train=x[train_index],y[train_index]
#     x_test,y_test=x[test_index],y[test_index]
# clf=GradientBoostingClassifier(n_estimators=300,learning_rate=0.1,max_depth=2,min_samples_split=200,min_samples_leaf=6) #0.847
# clf.fit(x_train,y_train)
# y_pred=clf.predict(x_test)
# y_pred_prob=clf.predict_proba(x_test)[:,1] #属于正类的概率

# 混淆矩阵
# matrix=confusion_matrix(y_test,y_pred)
# plt.figure()
# plot_confusion_matrix(matrix,classes=['unsurvived','survived'],title='Confusion matrix')
# plt.show()
#report
# target_names = ['unsurived', 'survived']
# print(classification_report(y_test,y_pred,target_names=target_names))
#auc指标
#y_pred_prob是样本属于正类的概率，pos_label是正类的类别标签，剩下的为负类,返回不同thresholds下的一组fpr,tpr，从而得到roc曲线
# fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred_prob,pos_label=1)
# auc=metrics.auc(fpr,tpr)#计算roc曲线下的面积就死auc的值
# print(auc)