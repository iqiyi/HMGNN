## Heterogeneous Multiple Mini-Graph Neural Network for Invitation Anti-cheating

### code overview
```
.
├── HMGNN.py
├── establish_super_nodes.py
├── hparams.py
├── utils.py
├── data_utils
│   ├── __init__.py
│   └── data_loader.py
├── model_graph
│   ├── __init__.py
│   ├── inits.py
│   ├── layers.py
│   ├── metrics.py
│   └── models.py
├── data
│   ├── edges_mat.npy
│   ├── features.npy
│   └── labels.npy
├── requirements.txt
└── README.md
```

-  GAT_models 和 GAT_utils 是用于GAT建模的组件。 
-  data_utils中包括了从MySql数据库中读取数据，包括特征、标注、关系图等
-  date_utils.py 用来获取日期
-  hparam.py 是参数设置
-  utils.py 用于数据预处理等
-  requirements.txt 包含特定依赖，代码基于Tensorflow完成

### 数据读写

主要流程：先从MySql读入特征(`feature_utils.py`), 标注(`label_utils.py`)，以及关系图(`edge_utils.py`)，然后在`data_loader.py`中根据超参设置合并这些信息。`data_loader.py`中包含了具体的合并步骤。

- `feature_utils.py`：包括从MySql中读取特征信息，删除不必要的特征列，对每个uid选择最新的特征，和将特征dataframe转化成(uid + feature + date)的形式。
- `label_utils.py`：包括从MySql中读取标注信息、删除不必要的列（只保留(uid+label+date)）列，和对每个uid选择最大的label作为最终的label。
- `edge_utils.py`: 包括从MySql中读取边信息和合并重复边，其中邀请关系的边权是按照邀请次数叠加，其余关系图边权均为1.
- `mysql_helper.py`和`mysql_tabris_helper.py`：是对MySql读写数据的辅助函数。
- `data_loader.py`：包括如下9个步骤。
    - 调用`feature_utils.py`、`label_utils.py`和`edge_utils.py`读取特征、标注和边信息。
    - 根据训练时间区间筛选需要的数据信息，时间区间定义在`hparam.py`中。
    - 将feature表和label表左合并得到feature_label表，并且对没有label的feature将其label设为-1.
    - 根据邀请关系图的大小，筛掉feature uid中较小关系图的uid，关系图大小阈值定要在FLAGS.filter_graph_size，默认是5.
    - 根据feature中的uid，过滤无用的边：只有边的至少一个端点在feature uid中，才保留这条边。
    - 合并所有的边的uid，得到相应的id，即对uid进行id编码。
    - 对feature_label表中的uid替换成id表示，对在边中但不在feature_label中的uid，设置其特征为全0特征+末尾补1.
    - 将边中的uid编码为id。
    - 生成sparse的邻接矩阵。

Notes:

- **data_loader.py** 和 **data_loader_v2.py** 的区别在于：v2会返回原始的特征向量和邻接矩阵，因为后续需要构建超点。
- 在调用data_loader.load_data()时，会根据FLAGS.predict_date区分是随机划分测试集还是在指定的日期（FLAGS.predict_date）上测试。
- 划分训练集：验证集：测试集的比例是6:2:2.

    

### 模型说明

### 参数说明

参数的定义主要在`hparam.py`中，主要包括：
- 特征维数feature_dim=174
- 迭代次数epochs=300
- 学习率learning_rate=0.0005
- 是否使用attention机制attention=True
- 是否使用原始特征向量的嵌入：residual=True
- 是否平衡黑白样本balance=False，如果设置为True，则对黑(或者白)样本下采样，保持两类样本个数一样
- 是否重新量化邻接矩阵的权重reweight_adj=False，如果设置为True，则会重新量化邻接矩阵的权重。在上线代码中并没有使用这个参数。
- 新生成的邻接矩阵的权重系数beta=0.5, 这里主要是在建模时的尝试，在上线代码中并没有使用这个参数。

#### 上线模型
安装好依赖后，运行`python HMMG.py`即运行代码。


#### 对比模型

对比模型包括非图方法(`LR.py`, `XGBoost.py`, `LGBM.py`）和基于图的方法(`GCN.py`, `GAT.py`)。