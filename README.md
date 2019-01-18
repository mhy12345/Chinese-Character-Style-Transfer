# Chinese-Character-Style-Transfer

## 工程结构

* `./raw_data` : 原始数据集
* `./dataset` : 模型运行加载的数据集
* `./data` : 数据加载模块
* `./model` : 模型
* `./utils` : 数据预处理以及一些全局包
* `./checkpoints` : 训练时数据保存
* `./save_models` : 每个版本的模型保存
* `./train.py` : 使用训练集训练模型
* `./test.py` : 使用测试集运行模型
* `./exp.py` : 一些试验，诸如渐变style向量等等

## 工程搭建步骤

* 在根目录新建连接`./dataset`，`./checkpoints/main`
* 在根目录新建超链接`./raw_data`，指向原始的数据集
* 运行`./utils/picture_transform.py`，即可将 `../raw_data/image_2939x200x64x64_stand.npy`加载到`./dataset`目录
* 运行`./train.py`即可开始训练

## 数据加载器

现在支持两种加载方式`CrossDataset`和`PairedDataset`。

`CrossDataset`加载器支持每次加载“N个风格图片，N个文字图片，1个目标图片”。

`PairedDataset`加载器支持每次随机两种风格，分别返回该风格的N个图片。


## 模型

现在有两个模型`CrossModel`和`CrossModelV`，其都是基于`CrossDataset`加载方式的。其中`CrossModelV`将内容压缩为了向量，而结果跟屎一样。`CrossModel`则是让文字图片基于风格向量自行变化形成的。
