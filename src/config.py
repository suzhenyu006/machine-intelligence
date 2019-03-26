# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:15
# @Author  : Suzhenyu
# @File    : AlexNet.py
# @Email   : suzhenyu@qiyi.com

class DefaultConfigs(object):
	"""docstring for DefaultConfig"""
	train_data_path = ["../data/train/"]     #训练数据所在路径
	test_data_path = "../data/test/"       #要识别的图像存储路径
	weights_path = "../log/"               #模型保存路径
	normal_size = 270 				        #图像输入网络之前需要被resize的大小
	channels = 3                           #RGB通道数
	epochs = 50                            #训练的epoch次数
	batch_size = 64                         #训练的batch数
	classes = 2                            #要识别的类数
	data_augmentation = True               #是否使用keras的数据增强模块
	model_name = "ResNet_50"                #选择所要使用的网络结构名称
	hot_pic_path = "../data/img_grad_cam/"				#热图保存地址

config = DefaultConfigs()