import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor
from sklearn.feature_selection import SelectFromModel,VarianceThreshold
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier

#缺失值情况
def missing_condition(features):
    missing_count = features.isnull().sum()[features.isnull().sum() > 0].sort_values(ascending=False)  # sum()按列求和，返回一个index为之前列名的Series,sort_values进行排序，ascending=False表示是降序排列
    missing_percent = missing_count / len(features) #计算缺失值占总样本的比例
    missing_df = pd.concat([missing_count, missing_percent], axis=1, keys=['count', 'percent'])
    print(missing_df)

#处理缺失值
def fill_missing(df):
    # 对有超过80%缺失的特征进行删除,包括'PoolQC','MiscFeature','Alley','Fence'，这些特征与房价的关系不大
    # GarageYrBlt与YearBult相似度很高，且YearBult与房价的线性关系较明显，所以删除GarageYrBlt
    # 'GarageQual','GarageCond'描述的好坏与房价不成正相关，认为这两个特征对房价预测不重要，删除
    # BsmtFinType2和BsmtFinSF2与房价的关系不强，删除
    #SaleType,Exterior1st,Exterior2nd，BsmtHalfBath,BsmtUnfSF与房价关系不强，删除
    # Utilities的取值单一，大部分数据都取同样的值，所以去除该特征
    # 分类特征FireplaceQu,GarageType,GarageFinish,MasVnrType都是有NA这个取值的，如果缺失就默认是不存在
    # 分类特征Electrical用众数填充
    # 数值特征LotFrontage用中位数填充
    # 数值特征MasVnrArea缺失是由于MasVnrType为NA，所以其面积用0来填充
    drop_list = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageQual', 'GarageCond', 'BsmtFinType2',
                 'BsmtFinSF2', 'Utilities','SaleType','Exterior1st','Exterior2nd','BsmtHalfBath','BsmtUnfSF']
    df.drop(drop_list, axis=1, inplace=True)
    cate_NA = ['FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType']
    for col in cate_NA:
        df[col].fillna('NA', inplace=True)
    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    # 缺失值中含有Bsmt的描述地下室，一起处理
    # 因为有NA的特征的缺失值不同，所以只有BsmtCond，BsmtExposure和BsmtQual都同时缺失时，才将其认为是没有地下室的,用NA填充
    NoBmstIndex = (df['BsmtCond'].isnull() & df['BsmtQual'].isnull() & df['BsmtExposure'].isnull())
    Bsmt_col = ['BsmtCond', 'BsmtQual', 'BsmtExposure']
    for col in Bsmt_col:
        df.ix[NoBmstIndex, col] = 'NA'
    # 该三个特征剩余缺失值用众数填充
    for col in Bsmt_col:
        df[col].fillna(df[col].mode()[0], inplace=True)
    # 由于没有basement，所以BsmtFinSF1为0，BsmtFinType1为NA
    df.ix[NoBmstIndex & df.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = 0
    df.ix[NoBmstIndex & df.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NA'
    # 要将没有地下室和地下室未完成的面积区分开，所以如果是未完成的，将其面积设为中位数
    df.ix[df.BsmtFinType1 == 'Unf', 'BsmtFinSF1'] = df.BsmtFinSF1.median()
    #剩下的特征，如果没有basement,全部赋值0
    df.ix[NoBmstIndex & df.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
    df.ix[NoBmstIndex & df.TotalBsmtSF.isnull(), 'TotalBsmtSF'] = 0
    return df

# 调整了SalePrice，GrLivArea，
# TotalBsmtSF有偏移，且有较多为0的值，忽略0值，用中位数填充，进行log变换
def normal_dist(df):
    df['GrLivArea'] = np.log(df['GrLivArea'])
    df.ix[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = df['TotalBsmtSF'].median()
    df['TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
    return df

#将有序的分类特征进行映射，比如级别评选：good,poor之类的，需要映射为有序的数值
def cate_change(df):
    df = df.replace({'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
                     'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
                     'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}})
    return df

#分类特征进行one-hot编码
def one_hot(df):
    categorical_feats = df.columns[df.dtypes == 'object']  # 得到分类特征
    features_encoding = pd.get_dummies(df[categorical_feats])
    df.drop(categorical_feats, axis=1, inplace=True)
    df = pd.concat([df, features_encoding], axis=1)
    return df

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
x_train=train_data.drop(['Id','SalePrice'],axis=1) #只包含训练集的特征
y_train=train_data['SalePrice'] #训练集的输出
x_test=test_data.drop(['Id'],axis=1) #测试集的特征
id_test=test_data['Id'] #测试集的Id

#缺失值处理
x_train=fill_missing(x_train) #训练集调用函数fill_missing进行处理
x_test=fill_missing(x_test) #测试集调用函数fill_missing进行处理
#测试集中有部分缺失情况在训练集中未出现，需要单独处理
#分类特征MSZoning,Functional,KitchenQual用众数填充
cate_mode=['MSZoning','Functional','KitchenQual']
for col in cate_mode:
    x_test[col].fillna(x_test[col].mode()[0],inplace=True)
#GarageCars缺失的样本，通过观察其他GarageX的特征，只有GarageType没有缺失，值为Detchd，所以用GarageType=Detchd的样本的中位数进行填充,是1.0
x_test['GarageCars'].fillna(1.0,inplace=True)

#异常点处理,主要分析了重要的数值特征GrLivArea，TotalBsmtSF
#通过可视化GrLivArea和SalePrice的回归图，会发现有两个离群点,GrLivArea是最高的两个，而SalePrice较低，将这两个点去掉
#TotalBsmtSF有一个离群点，去掉
outlier_Gr=x_train.sort_values(by = 'GrLivArea',ascending = False)[:2].index #按GrLivArea进行倒序排列，取最前面的两个的index
x_train.drop(outlier_Gr,inplace=True)
y_train.drop(outlier_Gr,inplace=True)
outlier_Bs=x_train.sort_values(by = 'TotalBsmtSF',ascending = False)[:1].index #按GrLivArea进行倒序排列，取最前面的两个的index
x_train.drop(outlier_Bs,inplace=True)
y_train.drop(outlier_Bs,inplace=True)

#调整输出和重要的数值特征，使其符合正态分布
y_train = np.log(y_train)
x_train=normal_dist(x_train)
x_test=normal_dist(x_test)

#特征选择
#去除冗余性，相关性强的变量中只保留对房价影响最大的变量
#GarageCars和GarageArea的相关性很高，删除GarageArea
#TotalBsmtSF和1stFlrSF的相关性很高，删除1stFlrSF
#YearBuil和YearRemodAdd的相关性很高，删除YearRemodAdd
#TotRmsAbvGrd和GrLivArea的相关性很高，删除TotRmsAbvGrd
drop_list=['GarageArea','1stFlrSF','YearRemodAdd','TotRmsAbvGrd']
x_train.drop(drop_list,axis=1, inplace=True)
x_test.drop(drop_list,axis=1, inplace=True)

#利用皮尔森稀疏来评估单特征与输出的线性相似度，删除对房价影响较小的特征
drop_fetures=[ 'MiscVal','LowQualFinSF', 'YrSold','3SsnPorch', 'MoSold',  'MSSubClass']
x_train.drop(drop_fetures,axis=1,inplace=True)
x_test.drop(drop_fetures,axis=1,inplace=True)

#分类特征转为数值特征
#有序的分类特征转为数值特征
x_train=cate_change(x_train)
x_test=cate_change(x_test)
#无序的分类特征进行one-hot编码,注意训练集和测试集一起，防止得到的特征维度不同
all_data=pd.concat([x_train,x_test])
all_data=one_hot(all_data)

#划分出训练集和测试集
x_train=all_data[:len(x_train)]
x_test=all_data[len(x_train):]
print(x_train.shape)
print(x_test.shape)

#再次进行特征选择，使用随机森林
# feat_labels=x_train.columns
forest=RandomForestRegressor(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(x_train,y_train)
importance=forest.feature_importances_
# 观察每个特征的重要系数
# importance_dict={}
# for i in range(len(feat_labels)):
#     importance_dict[feat_labels[i]]=importance[i]
# importance_dict=sorted(importance_dict.items(),key=lambda item:item[1])
# print(importance_dict)
model = SelectFromModel(forest, threshold=0.0001,prefit=True) #根据特征的重要系数确定阈值，阈值的大小需要调参
x_train=model.transform(x_train)
x_test=model.transform(x_test)
print(x_train.shape)
print(x_test.shape)

#进行归一化
stds=StandardScaler()
x_train=stds.fit_transform(x_train)
x_test=stds.transform(x_test)

#将进行完预处理的训练集输出到文件中
y_train=y_train.as_matrix()
id_test=id_test.as_matrix()
y_train.shape=(y_train.shape[0],1)
id_test.shape=(id_test.shape[0],1) #对于一列的数据需要表明列数是1
data_train=np.hstack((y_train,x_train)) #hstack按列合并
data_test=np.hstack((id_test,x_test))
np.savetxt("train_precessing.csv",data_train,delimiter=',')
np.savetxt("test_precessing.csv",data_test,delimiter=',')


