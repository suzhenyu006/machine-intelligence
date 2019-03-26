服务于图片分类的机器学习项目

使用Keras框架，OpenCV，基于TensorFlow后端

目前已经将一些CNN模型开发到我们自己的工程里面，使用者根据自己的想使用的模型，在run.sh中传入即可


>目前支持的神经网络模型为以下几种:
>- LeNet
>- AlexNet  
>- VGGNet
>- ZFNet
>- GoogLeNet
>- ResNet_18/34/50/101/152
>- DenseNet_161


# 1.依赖:
keras >=2.1.5
tensorflow >=1.8
opencv-python >= 3.4.0.12


# 2.工程结构树:
├─data             
│  ├─test          
│  └─train         
│      ├─00000     
│      ├─00001     
│      ├─00002     
│      ├─00003     
│      ├─00004     
│      ├─00005     
│      ├─00006     
│      ├─00007     
│      ├─00008     
│      └─00009     
├─log              
└─src              
    │  config.py
    │  train.py
    │  utils.py
    │  predict.py
    │
    ├─models
    │  │  AlexNet.py
    │  │  DenseNet.py
    │  │  GooLeNet.py
    │  │  LeNet.py
    │  │  resnet.py
    │  │  VGGNet.py
    │  │  ZFNet.py 


# 3.怎么用：

按照上面工程树data下的train文件夹下的目录结构，区分类型放置你的训练集
目前工程树结构是区分了10个类型

运行train.py，开始进行训练，配置信息在config.py中


预测数据：

将被预测图片放到data文件夹下的test文件夹中

运行：
>python predict.py
