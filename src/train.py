# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:15
# @Author  : Suzhenyu
# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:15
# @Author  : Suzhenyu
# @File    : AlexNet.py
# @Email   : suzhenyu@qiyi.com
import os
from keras.models import load_model
from keras import backend as K
from utils import build_model, load_data, h5_to_pb
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os.path as osp
import argparse
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

FLAGS = None

def train(FLAGS,train_x,train_y,dev_x,dev_y):
    model = build_model(FLAGS)
    lr_reducer = ReduceLROnPlateau(factor=0.005, cooldown=0, patience=5, min_lr=0.5e-6,verbose=1)      #设置学习率衰减
    early_stopper = EarlyStopping(min_delta=0.001, patience=10,verbose=1)                                     #设置早停参数
    checkpoint = ModelCheckpoint(FLAGS.model_dir + FLAGS.model_name + "_model.h5",
                                 monitor="val_acc", verbose=1,
                                 save_best_only=True, save_weights_only=False,mode="max")            #保存训练过程中，在验证集上效果最好的模型
    #使用数据增强
    if FLAGS.data_augmentation:
        print("using data augmentation method")
        data_aug = ImageDataGenerator(
            rotation_range=90,              #图像旋转的角度
            width_shift_range=0.2,          #左右平移参数
            height_shift_range=0.2,         #上下平移参数
            zoom_range=0.3,                 #随机放大或者缩小
            horizontal_flip=True,           #随机翻转
        )
        data_aug.fit(train_x)
        model.fit_generator(
            data_aug.flow(train_x,train_y,batch_size=FLAGS.batch_size),
            steps_per_epoch=train_x.shape[0] // FLAGS.batch_size,
            validation_data=(dev_x,dev_y),
            shuffle=True,
            epochs=FLAGS.epochs,verbose=1,max_queue_size=100,
            callbacks=[lr_reducer,early_stopper,checkpoint]
        )
    else:
        print("don't use data augmentation method")
        model.fit(train_x,train_y,batch_size = FLAGS.batch_size,
                  nb_epoch=FLAGS.epochs,
                  validation_data=(dev_x, dev_y),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, checkpoint]
                  )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Directory for storing input data')
    parser.add_argument('--output_dir', type=str,
                        help='Directory for storing output data')
    parser.add_argument('--model_dir', type=str,
                        help='Directory for storing model')
    parser.add_argument('--normal_size', type=int,
                        help='Directory for storing model')
    parser.add_argument('--channels', type=int,
                        help='Directory for storing model')
    parser.add_argument('--epochs', type=int,
                        help='Directory for storing model')
    parser.add_argument('--batch_size', type=int,
                        help='Directory for storing model')
    parser.add_argument('--classes', type=int,
                        help='Directory for storing model')
    parser.add_argument('--data_augmentation', type=bool,
                        help='Directory for storing model')
    parser.add_argument('--model_name', type=str,
                        help='Directory for storing model')
    FLAGS, unparsed = parser.parse_known_args()
    print FLAGS.data_dir
    print FLAGS.model_dir
    images_data, labels = load_data(FLAGS)#加载训练集图片
    train_x,dev_x,train_y,dev_y = train_test_split(images_data,labels,test_size=0.25) #随机切分数据集，分为训练和验证集
    train(FLAGS,train_x,train_y,dev_x,dev_y)
    input_path = FLAGS.model_dir.split("/")[0] + "/" + FLAGS.model_dir.split("/")[1] + "/" + FLAGS.model_dir.split("/")[2] + "/" + FLAGS.model_dir.split("/")[3] + "/" + FLAGS.model_dir.split("/")[4]
    weight_file = FLAGS.model_dir.split("/")[5] + FLAGS.model_name + '_model.h5'
    weight_file_path = osp.join(input_path, weight_file)
    output_graph_name = weight_file[:-3] + '.pb'
    output_dir = osp.join(input_path, "trans_model")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    h5_model = load_model(weight_file_path)
    h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
    print('model saved')