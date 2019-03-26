# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:15
# @Author  : Suzhenyu
# @File    : AlexNet.py
# @Email   : suzhenyu@qiyi.com
import h5py
from keras.preprocessing.image import img_to_array
from config import config
from utils import build_model, conv_output, vis_conv, output_heatmap, vis_heatmap
import numpy as np
import cv2
import os


def predict(config):
    global image1
    global data
    global images
    weights_path = config.weights_path + config.model_name + "_model.h5"
    data_path = config.test_data_path
    data_list = os.listdir(data_path)
    data_list.remove(".DS_Store")
    print(data_list)
    print(data_path)
    data1 = []
    data = {}
    images = []
    for file in data_list:
        file_name = data_path + file
        image = cv2.imread(file_name)
        image = cv2.resize(image, (config.normal_size,config.normal_size))
        images.append(image)
        image1 = img_to_array(image)
        data1.append(image1)
    data1 = np.array(data1, dtype="float") / 255.0
    model = build_model(config)
    model.load_weights(weights_path)
    pred = model.predict(data1)
    f = h5py.File(weights_path)
    for layer, g in f.items():
        for key, value in g.attrs.items():
            data[key] = value
    res = data["layer_names"]
    for i in res:
        print(i)

    result = conv_output(model, "conv2d_1", data1)
    vis_conv(result, 8, "conv2d_1", 'conv')

    result = conv_output(model , "conv2d_10" , data1)
    vis_conv(result, 8, "conv2d_10", 'conv')

    result = conv_output(model , "conv2d_20" , data1)
    vis_conv(result, 8, "conv2d_20", 'conv')

    result = conv_output(model , "conv2d_30" , data1)
    vis_conv(result, 8, "conv2d_30", 'conv')

    result = conv_output(model , "conv2d_40" , data1)
    vis_conv(result, 8, "conv2d_40", 'conv')

    result = conv_output(model , "conv2d_50" , data1)
    vis_conv(result, 8, "conv2d_50", 'conv')

    print(pred)

    heatmap = output_heatmap(model, "batch_normalization_49", data1)
    vis_heatmap(images[0], "batch_normalization_49" , heatmap)

predict(config)