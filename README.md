服务于分帧后台的机器学习项目

目前已经将一些CNN模型开发到我们自己的工程里面，使用者根据自己的想使用的模型，在run.sh中传入即可

目前此服务已经对接在爱奇艺深度学习云中：
http://jarvis.cloud.qiyi.domain/group/


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
目前已经对接深度学习云，推荐大家直接在爱奇艺云平台开展你的训练：

使用jarvis训练数据需要做一些前期配置，请参考：
http://jarvis.gitlab.qiyi.domain/jarvis/training/step-by-step.html

远程ssh启动任务的命令行： ./jarvis create-gitlab-job <GitLab 项目 SSH 地址>

预测数据：

运行：
>python predict.py
