# Titanic

**二元分类**问题。

**一、认识数据**

1.样本的整体情况，样本的类别比例，各项特征的类型，缺失值情况，均值，极值等。

2.用可视化或图表的方式观察特征与输出Y之间的关系，特征之间的关系。

样本类别比例Survived:Unsurvived=1:1.6，没有明显的数据偏倚。

 

**二、预处理** 

**1.缺失值的处理**

①缺失值较多。**直接去除**，以免引入噪声。或者将**是否缺失作为一个特征**（可以在认识数据阶段可视化是否缺失与输出之间的关系），比如缺失颜色：可以考虑r,g,b,缺失这四种情况。

②缺失值较少。采用填充的方法。1）使用**常量填充**：比如均值，中值，众数等，一般是根据label的值，选取同一类型下的该特征的均值填充或者是根据某相关属性的均值填充；2）用**算法拟合**进行填充，即利用训练集中没有缺失值的特征来预测该特征的取值，可以使用树模型，比如：决策树、随机森林、GBDT等。缺点在于如果其他特征与缺失特征无关，则预测无意义，如果预测很准确，则说明该特征没有加入建模的必要，一般情况介于两者之间。

③**忽略**。有些模型比如决策树自身能处理缺失值，但这样会受模型选择的限制。

本例：

①均值填充。测试集中有一个样本的Fare缺失，用Pclass与之相同的样本均值填充。

②众数填充。训练集中有两个样本的Embarked缺失，将其赋值给登船人数最多的S港口。

③缺失有无填充。Cabin属性的缺失过多且缺失与否获救的可能性不同，所以将获救设为‘yes’，未获救设为‘no’，之后进行one-hot编码。

④均值填充。Age按照Name中的Mr,Miss,Mrs,Master,Dr的均值进行填充。

 

**2.挖掘新的特征**

①**属性内部字段的提取**。对于提供的特征属性，可能需要进一步的提取，比如本例中的Name属性，乍一看没有一个相同的值，但表明人身份的称呼，如：Mr,Mrs,Miss,Master,Dr，包含了有用的信息，可以将其提取出来作为新的属性Title，并删除原来的属性Name。同样的还有Ticket属性，虽然既包含字母又包含数字，比较乱，但将数字提取出来可以发现数字较小，获救的可能性较大，所以可以得到Ticket_nuber属性，删除之前的Ticket属性。

②**属性之间的组合**。有些属性通过组合可以获得新的特征，比如本例中的SibSp,Parch属性，相加可得到到Family_size属性，家族的大小可能会影响获救的可能性。

 

**3.分类数特征转为数值型**

①有序的分类特征。采用映射的方法，比如将‘XL‘，’L‘，’M‘映射为3,2,1。

②无序的分类特征。采用one-hot编码，pd.get_dummies方法会对DataFrame中所有字符串类型的列进行独热编码。

本例：

①Pclass是有序的数值特征，不需要进行转换。

②无序的分类特征有：Title,Sex,Cabin,Embarked需呀进行one-hot编码。

 

**4.特征选择**

使用树模型（随机森林）能评估各个特征的重要性，从而去除不相关的特征。基本的原理就是决策树特征选择的原理，基于基尼指数或者信息熵选择特征。但存在几个问题：

①存在偏向性，会偏向具有更多取值种类的特征。

②对存在关联的多个特征，可能只有一个特征的重要度较高，而剩余特征的重要度较低，其实这些特征是同等重要的，因为不纯度已经由选中的那个特征将下来了，其他特征很难降低那么多不纯度。比如：进行one-hot编码后的Title分为Title_Mr,Title_Mrs等多种，其中只要有一个较高的重要度，则这些特征都是重要的。

本例中，通过观察特征的重要度并进行交叉验证，最后删除的特征有Embaked和Parch。

 

**5.特征标准化/归一化**

**归一化**

X=(x-min)/(max-min)

特点：

①可以将特征的取值限制在【0,1】范围内，达到无量纲的作用。

②改变原始数据的分布。使各个特征维度对目标函数的影响权重是一致的。

③最大值和最小值容易受异常点影响，鲁棒性差，只适合精确的小数据场景。

**标准化**

X=x-u/s

特点：

①标准化后的数据均值为0，标准差为1，也能达到去量纲的效果。

②不改变原始数据的分布。保持各个特征维度对目标函数的影响权重。

③受异常点的影响没有归一化那么大，适合嘈杂的大数据环境。

**需要进行标准化/归一化的算法：**

**逻辑回归**（求解过程中有梯度下降法，进行标准化处理能使其更加快速的收敛），**KNN**（有距离计算公式，进行标准能防止受数值较大的特征的过度影响），**svm**（防止模型的参数被分布范围较大或较小的数据控制）。

**不需要进行标准化/归一化的算法：**

**概率模型**（比如朴素贝叶斯，因为不关心变量的值，而只关心变量的分布及变量之间的条件概率）和**决策树**。

本例中使用的GBDT算法，弱分类器是决策树，所以不用进行标准化。

 

**三、算法评估**

**1.K折交叉验证**

**方法**：首先随机的将原先的训练集数据切分成K个互不相交的大小相同的子集，利用K-1个子集的数据作为训练集训练模型，剩下的一个子集作为验证集测试模型，这一过程对可能的K种选择重复进行，对K次得到的评估指标取平均值作为模型的评估结果。这种交叉验证的方式能够避免抽样的随机性所导致的对模型评估的不准确。

进行交叉验证时若样本类别不平衡，需要进行**分层抽样**，即在每一份的子集中保持原始。数据集的类别比例。

交叉验证主要是在模型调参中使用。

 

**2.网格搜索**

给模型的每个参数设定一组可能的值，通过**穷举**各种参数组合，使用交叉验证确定最优的一组。Skelarn中有实现。

 

**3.评估指标**

![img](file:///C:/Users/zjs_m/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

常用的就是**accuracy,precision,recall, F-measure**，在信息检索领域，precision叫查准率，recall叫查全率。

**ROC曲线**和**AUC指标**常被用来评价一个二元分类器的优劣。

ROC曲线横轴是FPR，纵轴是TPR。FPR表示将负例错分为正例的概率，TPR表示将正例正确分类的概率，所以点[0,1]是完美的分类器，点[1,0]是最差的分类器，ROC曲线越接近**左上角**，该分类器的性能越好。直线y=x上的点表示一个采用随机猜测策略的分类器的结果。

由于ROC曲线出现相交时，难以区分哪个分类器更优，所以引入AUC进行评估，它是**ROC曲线下的面积**，AUC值越大分类器效果越好。

使用AUC进行评估的原因是，当数据集中出现**类不平衡现象**及测试集中的正负**样本随时间变化**时，AUC指标也能进行很好的评估。

 

**四、结果**

目前在kaggle上取得的最好结果是0.77511。

 

**五．代码说明**

代码：

data_understanding.py：认识数据，对数据进行观察。

pretreatment.py：预处理

evaluation.py：评估算法

prediction.py：构建模型，进行预测

数据：

train.csv,test.csv：原始的训练集和测试集

train_standard.csv,test_standard.csv：预处理之后的训练集和测试集

result.csv：预测的结果
