import pandas as pd
import matplotlib.pyplot as plt
import re

data_train=pd.read_csv("train.csv")
data_test=pd.read_csv("test.csv")

# 数据整体情况
data_train.info() #查看数据的缺失情况，类型
print(data_train.describe())#查看数值型数据的均值等指标

#观察Ticket number与输出结果的关系(数字较小，获救可能性大些）
data_train['Ticket_number']=data_train['Ticket'].map(lambda s:s.split(' ')[-1])
# print(data_train['Ticket_label'])

# Pclass与输出的关系，upper的获救率更高
survived_0=data_train.Ticket_number[data_train.Survived==0].value_counts() #计算Ticket_label这一列未获救的人按Pclass作为index的数量，得到Series类型
survived_1=data_train.Ticket_number[data_train.Survived==1].value_counts()

df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0}) #由Series类型组成DatFrame类型
df.plot(kind='bar', color=['g','r'] ,alpha=0.5) #以index即Pclass为横轴进行画图，观察Pclass与获救结果之间的关系
plt.xlabel('Pclass') #设x坐标
plt.ylabel('number of people')
plt.legend() #图标的位置
plt.show()

# 认识数据，进行可视化（观察特征和输出的关系）
# Pclass与输出的关系，upper的获救率更高
survived_0=data_train.Pclass[data_train.Survived==0].value_counts() #计算Pclass这一列未获救的人按Pclass作为index的数量，得到Series类型
survived_1=data_train.Pclass[data_train.Survived==1].value_counts()

df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0}) #由Series类型组成DatFrame类型
df.plot(kind='bar', color=['g','r'] ,alpha=0.5) #以index即Pclass为横轴进行画图，观察Pclass与获救结果之间的关系
plt.xlabel('Pclass') #设x坐标
plt.ylabel('number of people')
plt.legend() #图标的位置

#sex与输出的关系，女性获救率更高
survived_0=data_train.Sex[data_train.Survived==0].value_counts()
survived_1=data_train.Sex[data_train.Survived==1].value_counts()

df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0})
df.plot(kind='bar', color=['g','r'] ,alpha=0.5)
plt.xlabel('Sex')
plt.ylabel('number of people')
plt.legend()

#age与输出的关系(年纪较小的获救可能性大些)
survived_0=data_train.Age[data_train.Survived==0].value_counts()
survived_1=data_train.Age[data_train.Survived==1].value_counts()

df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0,'rate':survived_1/(survived_0+survived_1)})
df.plot(kind='bar', color=['g','r'] ,alpha=0.5)
plt.xlabel('Age')
plt.ylabel('number of people')
plt.legend()

# SibSp与输出的关系(有1-2个SibSp的获救的可能性一半一半，其他情况较低)
survived_0=data_train.SibSp[data_train.Survived==0].value_counts()
survived_1=data_train.SibSp[data_train.Survived==1].value_counts()
df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0})
print(df)

# Parch与输出的关系(有1-3个Parch获救的可能性一半，其他可能性较低)
survived_0=data_train.Parch[data_train.Survived==0].value_counts()
survived_1=data_train.Parch[data_train.Survived==1].value_counts()
df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0})
print(df)

#Fare是与Pclass相关的属性，Fare越多Pclass越upper
print(data_train.Fare[data_train.Pclass==1].mean())
print(data_train.Fare[data_train.Pclass==2].mean())
print(data_train.Fare[data_train.Pclass==3].mean())

# cabin的缺失值很多，分析cabin值有无对输出的影响(有cabin记录的获救可能性更高)
cabin=data_train.Survived[data_train.Cabin.notnull()].value_counts()
nocabin=data_train.Survived[data_train.Cabin.isnull()].value_counts()
df=pd.DataFrame({"nocabin":nocabin,"cabin":cabin}).transpose() #transpose()行列转换
df.plot(kind="bar",color=['r','g'],alpha=0.5)
plt.xlabel('cabin is or not')
plt.ylabel('number of people')
plt.legend()

# Embarked对输出结果的影响（在S登串获救的可能性低于在C口和Q口）
survived_0=data_train.Embarked[data_train.Survived==0].value_counts()
survived_1=data_train.Embarked[data_train.Survived==1].value_counts()

df=pd.DataFrame({"survived":survived_1, "unsurvived":survived_0})
df.plot(kind='bar', color=['g','r'] ,alpha=0.5) #DtaFrame的plot方法是以列名称作为横轴进行绘图
plt.xlabel('Embarked')
plt.ylabel('number of people')
plt.legend()
plt.show()
