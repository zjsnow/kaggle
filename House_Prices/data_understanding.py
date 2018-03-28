import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#缺失值的情况
def missing_condition(features):
    missing_count = features.isnull().sum()[features.isnull().sum() > 0].sort_values(ascending=False)  # sum()按列求和，返回一个index为之前列名的Series,sort_values进行排序，ascending=False表示是降序排列
    missing_percent = missing_count / len(features) #计算缺失值占总样本的比例
    missing_df = pd.concat([missing_count, missing_percent], axis=1, keys=['count', 'percent'])
    print(missing_df)
    print(missing_df.index)

#观察单个的数值特征的分布是否符合正态分布
def features_distribution(train_data,num_feature):
    # train_data.ix[train_data.LotFrontage.isnull(),num_feature] = train_data.LotFrontage.median() #如果有缺失值，需要进行处理后才能画图
    sns.distplot(train_data[num_feature]) #画图时，如果不符合正态分布，可以进行变换
    plt.figure()
    stats.probplot(train_data[num_feature], plot=plt)  # 通过p-p图可以确定正态化程度

# 分类特征画箱线图，观察分类特征和售价之间是否存在明显的线性关系，若存在需要映射成有序数值
def plot_cate_features(train_data,cate_feature_list):
    def my_boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
    df = pd.melt(train_data, id_vars=['SalePrice'], value_vars=cate_feature_list,var_name='features', value_name='value')  # 将所有分类特征名作为变量放入表中
    # col用来创建数据的子集，指定划分数据的依据，得到多个子图，col_wrap指定数值则能使子图的排列不限制于一行，col_wrap=4表示一行四个子图，sharex=False表示子图不共用x轴
    g = sns.FacetGrid(df, col='features', col_wrap=3, sharex=False, sharey=False)
    g.map(my_boxplot, 'value',
              'SalePrice')  # FacetGrid.map()提供绘图功能及绘图所需的横纵轴信息，定义函数my_boxplot而不直接用sns.boxplot是为了规定好顺序

# 数值特征画线性回归图，观察数值特征和售价是否有线性关系，若没有明显的线性关系，转为分类特征处理
def plot_num_features(train_data,num_features_list):
    def my_regplot(x, y, **kwargs):
        sns.regplot(x=x, y=y)  # 画出散点图并进行线性拟合，若设order=2则进行二次曲线拟合
    df = pd.melt(train_data, id_vars=['SalePrice'], value_vars=num_features_list,var_name='features', value_name='value')
    g = sns.FacetGrid(df, col='features', col_wrap=2, sharex=False, sharey=False)
    g.map(my_regplot, "value", "SalePrice")

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
x_train=train_data.drop(['Id','SalePrice'],axis=1) #只包含训练集的特征
y_train=train_data['SalePrice'] #训练集的输出
x_test=test_data.drop(['Id'],axis=1) #测试集的特征
id_test=test_data['Id'] #测试集的Id

categorical_feats=train_data.columns[train_data.dtypes=='object']
for c in categorical_feats:
    train_data[c] = train_data[c].fillna("Missing")  # 将所有分类特征中缺失的值设为"Missing"

#整体情况
# train_data.info() #训练集1460
# test_data.info() #测试集1459 总共2919

#观察训练集中输出house_prices的分布
# f, (ax1,ax2)= plt.subplots(1,2)
# sns.distplot(train_data['SalePrice'],ax=ax1) #存在偏移
# sns.distplot(np.log(train_data['SalePrice']),ax=ax2) #取log后更符合正态变换
# plt.figure()
# stats.probplot(np.log(train_data['SalePrice']), plot=plt) #通过p-p图可以确定正态化程度
# plt.show()

#缺失值的情况
# missing_condition(x_train)
# print(x_train['FireplaceQu'].mode())

plot_cate_features(train_data,['BsmtQual']) #分类特征的箱图
plot_num_features(train_data,['GrLivArea']) #数值特征的回归图
plt.show()

# 观察特征的两两相关性,颜色越深，相关性越高，相关性较高的可以适当删除一些特征
# plt.subplots(figsize=(12,10))
# corr=np.abs(train_data.corr())
# sns.heatmap(corr, cmap='Blues', vmin=0,square=True) #square=True表示每个特征都是正方形
# plt.show()

#观察与SalePrice关系最密切的前10个特征
# corr = np.abs(train_data.corr()) #计算皮尔森相关系数，放回Dataframe类型
# k=10
# cols=corr.nlargest(k,'SalePrice')['SalePrice'].index #nlargest()按照SalePrice选取最大的前10行，取得SalePrice列，再得到与SalePrice最相关的10个属性
# max_corr=train_data[cols].corr()
# sns.heatmap(max_corr, cmap='Blues', square=True,annot=True) #square=True表示每个特征都是正方形
# plt.show()

#输出对SalePrice影响最小的前10个特征，进行删除
# corr=np.abs(train_data.corr())
# print(corr['SalePrice'].sort_values()[:10].index)




