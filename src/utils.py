# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:15
# @Author  : Suzhenyu
# @File    : AlexNet.py
# @Email   : suzhenyu@qiyi.com
from itertools import chain
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from netmodel import AlexNet,resnet
from netmodel.LeNet import LeNet
from netmodel.VGGNet import VGG_16
from netmodel.ZFNet import ZF_Net
from netmodel.GooLeNet import GoogLeNet
from netmodel.DenseNet import DenseNet_161
import cv2
import os
import numpy as np
import glob
import tensorflow as tf
import os.path as osp
from keras import backend as K, Model
import matplotlib.pyplot as plt
from config import config

np.random.seed(42)

def get_file(category_path):
    return glob.glob(category_path + "/" + "*.jpg")

def load_data(FLAGS):
    labels = []                                                         #存放标签信息
    images_data = []
    data_paths = []                                                     #存放图片信息
    print("loading dataset......")
    data_paths.append(FLAGS.data_dir)                                   #也即是train文件夹
    category_paths = [list(map(lambda x: data_path + x, os.listdir(data_path))) for data_path in data_paths]
    category_paths = category_paths[0]
    files = list(chain.from_iterable(list(map(get_file,category_paths))))
    np.random.shuffle(files)
    for file_name in files:
        if file_name.find("DS_Store") < 0:
            if file_name.find("Readme") < 0:
                print file_name
                label = int(file_name.split('/')[6][-2:]) #提取类别信息
                labels.append(label)
                image = cv2.imread(file_name)   #使用opencv读取图像
                image = cv2.resize(image,(FLAGS.normal_size,FLAGS.normal_size))
                image = img_to_array(image)     #将图像转换成array形式
                images_data.append(image)
    #缩放图像数据
    images_data = np.array(images_data,dtype="float") / 255.0
    labels = np.array(labels)
    #将label转换成np.array格式
    labels = to_categorical(labels , num_classes=FLAGS.classes)
    return images_data,labels

def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

def build_model(FLAGS):
    #根据选择的网络模型构建
    if FLAGS.model_name == "AlexNet":
        model = AlexNet.AlexNet(FLAGS)
    elif FLAGS.model_name == "ResNet_18":
        model = resnet.ResnetBuilder.build_resnet_18(FLAGS)
    elif FLAGS.model_name == "ResNet_34":
        model = resnet.ResnetBuilder.build_resnet_34(FLAGS)
    elif FLAGS.model_name == "ResNet_50":
        model = resnet.ResnetBuilder.build_resnet_50(FLAGS)
    elif FLAGS.model_name == "ResNet_101":
        model = resnet.ResnetBuilder.build_resnet_101(FLAGS)
    elif FLAGS.model_name == "ResNet_152":
        model = resnet.ResnetBuilder.build_resnet_152(FLAGS)
    elif FLAGS.model_name == "LeNet":
        model = LeNet().build(FLAGS)
    elif FLAGS.model_name == "VGGNet":
        model = VGG_16(FLAGS)
    elif FLAGS.model_name == "ZFNet":
        model = ZF_Net(FLAGS)
    elif FLAGS.model_name == "GoogLeNet":
        model = GoogLeNet(FLAGS)
    elif FLAGS.model_name == "DenseNet_161":
        model = DenseNet_161(FLAGS)
    else:
        print("The model you have selected doesn't exists!")
    return model


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]

def vis_conv(images, n, name, t):
    size = 64
    margin = 5
    if t == 'filter':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin, 3))
    if t == 'conv':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin))
    for i in range(n):
        for j in range(n):
            if t == 'filter':
                filter_img = images[i + (j * n)]
            if t == 'conv':
                filter_img = images[..., i + (j * n)]
            filter_img = cv2.resize(filter_img, (size, size))

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            if t == 'filter':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
            if t == 'conv':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img

    plt.imshow(results)
    plt.savefig(config.hot_pic_path + name + ".png".format(t, name), dpi=600)
    # plt.show()


def output_heatmap(model, last_conv_layer, img):
    """Get the heatmap for image.
    Args:
            model: keras model.
            last_conv_layer: name of last conv layer in the model.
            img: processed input image.
    Returns:
            heatmap: heatmap.
    """
    # predict the image class
    preds = model.predict(img)
    # find the class index
    index = np.argmax(preds[0])
    # This is the entry in the prediction vector
    target_output = model.output[:, index]
    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

     # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads
    # given the input picture
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def vis_heatmap(img, name , heatmap):
    """visualize heatmap.
    Args:
           img: original image.
           heatmap：heatmap.
    """
    print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(221)
    plt.imshow(cv2.resize(img, (224, 224)))
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(heatmap)
    plt.axis('off')

    plt.subplot(212)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(config.hot_pic_path + name + "_heatMap.png", dpi=600)

plt.show()