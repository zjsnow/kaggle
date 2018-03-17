import pandas as pd
import string
import re
from pandas import Series,DataFrame
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#填充Age,使用随机森林进行预测
# def set_missing_age(df_train,df_test):
#     age_df_train=df_train[["Age","Pclass","SibSp","Parch","Fare"]] #选取出所有没有缺失值的数值型特征及需要预测的特征
#
#     known_age=age_df_train[df_train.Age.notnull()].as_matrix() #提取出知道age的样本，作为训练集，as_matrix()转为array类型，因为sklearn要求输入的X和Y是array类型
#     unknown_age=age_df_train[df_train.Age.isnull()].as_matrix() #不知道age的样本，作为测试集
#
#     y_train=known_age[:,0]
#     x_train=known_age[:,1:]
#
#     rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1) #n_estimators使用的决策树的个数，通常越多，效果越好，但计算量会很大，而且到达一定数量后，模型提升效果很小。n_jobs=-1表示使用CPU的所有核进行并行计算
#     rfr.fit(x_train,y_train) #训练模型
#     predict_age_train=np.around(rfr.predict(unknown_age[:,1:]),2) #进行预测
#
#     df_train.ix[(df_train.Age.isnull()), 'Age'] = predict_age_train  # 用预测的结果填充缺失值
#
#     #用与训练集一样的模型对测试集中的age缺失值进行填充
#     age_df_test = df_test[["Age", "Pclass", "SibSp", "Parch", "Fare"]]
#     unknown_age_test = age_df_test[df_test.Age.isnull()].as_matrix()
#     predict_age_test = np.around(rfr.predict(unknown_age_test[:, 1:]), 2)
#     df_test.ix[(df_test.Age.isnull()), 'Age'] = predict_age_test
#
#     return df_train,df_test

#cabin是否缺失进行填充,缺失设为no,存在设为yes
def set_missing_cabin(df_train,df_test):
    df_train.ix[(df_train.Cabin.notnull()), 'Cabin'] = 'yes'
    df_train.ix[(df_train.Cabin.isnull()),'Cabin']='no'
    df_test.ix[(df_test.Cabin.notnull()), 'Cabin'] = 'yes'
    df_test.ix[(df_test.Cabin.isnull()),'Cabin']='no'
    return df_train,df_test

#把名字进行转化，只留下称呼mr,mrs,miss,master(小男生),dr（地位较高的人）
def map_func(title):
    if title in ['Mr','Rev','Jonkheer','Major','Don','Capt','Sir','Col']:
        return 'Mr'
    elif title in ['Mrs','the Countess','Dona','Lady']:
        return 'Mrs'
    elif title in ['Miss','Mlle','Mme','Ms']:
        return 'Miss'
    elif title=='Master':
        return 'Master'
    elif title=='Dr':
        return 'Dr'
    else:
        print(title)
        return title

#处理Name，生成Title字段
def change_name(df_train,df_test):
    #title_list训练集和测试集中包含的所有Title的取值种类
    df_train['Title']=df_train.Name.map(lambda x: re.split("[.,]", x)).map(lambda x: x[1].strip()).map(map_func)
    df_test['Title'] = df_test.Name.map(lambda x: re.split("[.,]", x)).map(lambda x: x[1].strip()).map(map_func)

    return df_train,df_test

#提取Ticket的数字部分
def get_ticket_number(ticket):
    number=ticket.split(' ')[-1]
    if number.isnumeric(): #判断字符串中是否只包含数字字符
        return int(number)
    else:
        return np.nan

#对无序的分类的特征Title,sex，cabin,emarked使用one-hot编码转为数值特征，有序特征Pclass已经是数值特征，不需转换
def Numerical(df_train,df_test):
    df_train_encoding = pd.get_dummies(df_train[['Title',"Sex",'Cabin']])
    df_train = pd.concat([df_train, df_train_encoding], axis=1)  # axis=1按列连接

    #测试集进行同样操作
    df_test_encoding = pd.get_dummies(df_test[['Title',"Sex",'Cabin']])
    df_test = pd.concat([df_test, df_test_encoding], axis=1)  # axis=1按列连接

    return df_train,df_test

#标准化
def std(x_train,x_test):
    stdsc=StandardScaler()
    x_train_std=stdsc.fit_transform(x_train)
    x_test_std=stdsc.transform(x_test)
    return x_train_std,x_test_std

data_train=pd.read_csv("train.csv")
data_test=pd.read_csv("test.csv")

#缺失值处理
#测试集中有一个样本的Fare缺失，用Pclass与之相同的样本的均值进行填充
data_test.ix[(data_test.Fare.isnull()),'Fare']=data_test.Fare[data_test.Pclass==3].mean()
#训练集中有两个样本的Embarked缺失，将其赋值为登船人数最多的S口
data_train.ix[(data_train.Embarked.isnull()),'Embarked']='S'
#填充Age
# data_train,data_test=set_missing_age(data_train,data_test)
#填充cabin
data_train,data_test=set_missing_cabin(data_train,data_test)

#提取name属性中的称呼部分，作为新的Title属性
data_train,data_test=change_name(data_train,data_test)

#用不同Title的平均值来为Age填充
mean_ages =[]
mean_ages.append(data_train.Age[data_train.Title=='Mr'].mean())
mean_ages.append(data_train.Age[data_train.Title=='Mrs'].mean())
mean_ages.append(data_train.Age[data_train.Title=='Miss'].mean())
mean_ages.append(data_train.Age[data_train.Title=='Master'].mean())
mean_ages.append(data_train.Age[data_train.Title=='Dr'].mean())
#训练集
data_train.ix[(data_train.Age.isnull())&(data_train.Title=='Mr'),'Age']=mean_ages[0]
data_train.ix[(data_train.Age.isnull())&(data_train.Title=='Mrs'),'Age']=mean_ages[1]
data_train.ix[(data_train.Age.isnull())&(data_train.Title=='Miss'),'Age']=mean_ages[2]
data_train.ix[(data_train.Age.isnull())&(data_train.Title=='Master'),'Age']=mean_ages[3]
data_train.ix[(data_train.Age.isnull())&(data_train.Title=='Dr'),'Age']=mean_ages[4]
#测试集
data_test.ix[(data_test.Age.isnull())&(data_test.Title=='Mr'),'Age']=mean_ages[0]
data_test.ix[(data_test.Age.isnull())&(data_test.Title=='Mrs'),'Age']=mean_ages[1]
data_test.ix[(data_test.Age.isnull())&(data_test.Title=='Miss'),'Age']=mean_ages[2]
data_test.ix[(data_test.Age.isnull())&(data_test.Title=='Master'),'Age']=mean_ages[3]
data_test.ix[(data_test.Age.isnull())&(data_test.Title=='Dr'),'Age']=mean_ages[4]

#提取Ticket中的数字部分，作为新的特征Ticket_number
data_train['Ticket_number']=data_train['Ticket'].map(get_ticket_number)
data_train.ix[(data_train.Ticket_number.isnull()),'Ticket_number']=data_train['Ticket_number'].mean() #没有数字部分的用均值填充
data_test['Ticket_number']=data_test['Ticket'].map(get_ticket_number)

#增加family_size
data_train['Family_size']=data_train['SibSp']+data_train['Parch']
data_test['Family_size']=data_test['SibSp']+data_test['Parch']

#分类特征数值化
data_train,data_test=Numerical(data_train,data_test)

#将需要删除的字段全部删除
data_train.drop(['Title',"Sex", "Cabin", "Embarked",'Ticket','Name','Parch'], axis=1, inplace=True)
data_test.drop(['Title',"Sex", "Cabin", "Embarked",'Ticket','Name','Parch'], axis=1, inplace=True)

# data_train.info()

x_train=data_train.ix[:,'Pclass':'Cabin_yes'].as_matrix()
y_train=data_train.ix[:,'Survived'].as_matrix()
x_test=data_test.ix[:,'Pclass':'Cabin_yes'].as_matrix()
id_test=data_test.ix[:,'PassengerId'].as_matrix()

# x_train_std,x_test_std=std(x_train,x_test) #树模型可以不用进行标准化

# 用随机森林评估各个特征的重要性，去除不重要的特征，但需要注意相关特征的重要性总是一个高一个低，比如进行one-hot编码后的所有Title特征
# feat_labels=data_train.columns[2:]
# forest=RandomForestClassifier(n_estimators=500,random_state=0,n_jobs=-1)
# forest.fit(x_train,y_train)
# importance=forest.feature_importances_
# for i in range(len(feat_labels)):
#     print("%s     %f" %(feat_labels[i],importance[i]))

# 将预处理之后的数据保存到文件中
y_train.shape=(y_train.shape[0],1)
id_test.shape=(id_test.shape[0],1) #对于一列的数据需要表明列数是1
data_train=np.hstack((y_train,x_train)) #hstack按列合并
data_test=np.hstack((id_test,x_test))
np.savetxt("train_standard.csv",data_train,delimiter=',')
np.savetxt("test_standard.csv",data_test,delimiter=',')





