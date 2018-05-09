
# coding: utf-8

# # 采用TensorFlow 来搭建CNN进行手写数字预测

# In[1]:

#整体思路 用神经网络CNN提取特征后 用SVM进行分裂，效果的确比全连接层要好一些

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from sklearn import  svm
import sklearn


# <font color = red>首先对数据进行预处理<br>
#     1、将数据图像标准化<br>
#     2、将数据打包成为mini-batch<br>

# In[2]:




class get_mini_batch(object):
    """
    """
    def __init__(self,data,batch_size,image_height,image_width,channel=1):
        '''
        data:DataFrame数据类型，为读入的数据，第一列为标签，之后为像素数
        batch_size:每一个batch的图片数
        image_height:图片的高度
        image_width:图片的宽度
        channel:图片的通道数量
        '''
        #提取出标签，和图片数据
        #x_train ,y_train 均为矩阵
        
        y_train = data.label.values.astype(np.int32)
       #将像素归一化为【0:1】
        x_train = data.drop('label',axis=1).values.astype(np.float32)
        x_train = x_train/x_train.max()
        
        #将y转换为 42000个数据  28*28 1通道的图像
        x_train = x_train.reshape(-1,image_height,image_width,channel)
        
        self.x_train = x_train
        self.y_train = y_train
        self.mini_batch_size = batch_size
        self.index_in_epoch = 0
        self.current_epoch =0.0
        self.select_array = np.array([])
    def next_batch(self):
        '''
        return: x_train_data_batch , y_train_data_batch
        '''
        start = self.index_in_epoch
        self.index_in_epoch += self.mini_batch_size
        self.current_epoch += self.mini_batch_size/(len(self.x_train))
        
        #将选择数组扩充到与训练样本一样的长度
        if not len(self.select_array) == len(self.x_train):
            self.select_array = np.arange(len(self.x_train))
        
        #r若是第一次取batch，则打乱顺序取
        if  start == 0:
            np.random.shuffle(self.select_array)
        
        #若到了数据尾部,则打乱重新开始选择
        if self.index_in_epoch>self.x_train.shape[0]:
            start = 0
            np.random.shuffle(self.select_array)
            self.index_in_epoch = self.mini_batch_size
        end = self.index_in_epoch
        
        #至此已经选出mini-batch所以要对image进行标准化
        x_tr = self.x_train[self.select_array[start:end]]
        y_tr = self.y_train[self.select_array[start:end]]

        
        return x_tr,y_tr


# #  建立TensorFlow CNN模型

# In[3]:


def inference(images , batch_size ,n_classes ,dropout1_ratio,dropout2_ratio):
    '''
    参数说明:
    images：输入图像batch 4D-tensor shape =[batchsize,image_width,image_height,channel]
    batch_size : 每一批图像数量
    n_classes: 输出类别数
    
    返回值:
     output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # CONV1  16个 3*3 大小的卷积核
    with tf.variable_scope('conv1') as scope:
        #创建16 3*3 大小的卷积核,shape = [filter_height, filter_width, 
        #                                   in_channels, out_channels]
        weights = tf.get_variable('weights',
                                   shape= [3,3,1,16],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases  = tf.get_variable('biases',
                                   shape= [16],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
        conv   = tf.nn.conv2d(images,weights,[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1  = tf.nn.relu(pre_activation ,name =scope.name)
    # pool1 and norm1 池化层1和标准化层1
    with tf.variable_scope('pooling1_lrn') as scope:
        
        pool1 = tf.nn.max_pool(conv1,ksize =[1,2,2,1],strides=[1,2,2,1],
                              padding='SAME',name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,
                         beta=0.75,name='norm1')
    #CONV2  同conv1
    with tf.variable_scope('conv2') as scope:
        
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases  = tf.get_variable('biases',
                                   shape=[16],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
        
        conv = tf.nn.conv2d(norm1,weights,[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name = scope.name)
    
    #pooling2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,1,1,1],
                               padding='SAME',name='pool2')
        
    #full-connect1
    with tf.variable_scope('fc1')  as scope:
        #将池化层输出转化为每一行为一个样本
        reshape = tf.reshape(pool2,shape = [batch_size,-1])
        dim = reshape.get_shape()[1].value
        
        weights = tf.get_variable('weights',
                                   shape = [dim,512],
                                   dtype = tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases  = tf.get_variable('biases',
                                   shape = [512],
                                   dtype = tf.float32,
                                   initializer= tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)


    #加入一个dropout1

    with tf.variable_scope('dropout1') as scope :

        keep_prob1 = tf.constant(dropout1_ratio,dtype=tf.float32)
        dropout1 = tf.nn.dropout(fc1,keep_prob=keep_prob1)


    #full-connect2
    with tf.variable_scope('fc2') as scope:
        
        weights = tf.get_variable('weights',
                                   shape=[512,1024],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases  = tf.get_variable('biases',
                                  shape=[1024],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(dropout1,weights)+biases,name=scope.name)


    #dropout2
    with tf.variable_scope('dropout2') as scope :

        keep_prob2 = tf.constant(dropout2_ratio, dtype=tf.float32)
        dropout2 = tf.nn.dropout(fc2,keep_prob=keep_prob2)
        
    #soft-max output
    with tf.variable_scope('soft_max') as scope:
        
        weights =tf.get_variable('weights',
                                 shape = [1024,n_classes],
                                 dtype =tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(dropout2,weights),biases,name=scope.name)
        
    
    return softmax_linear,fc2



#获得特诊提取batch

class get_feature_batch(object):
    """
    """

    def __init__(self, data, batch_size, image_height, image_width, channel=1):
        '''
        data:DataFrame数据类型，为读入的数据，第一列为标签，之后为像素数
        batch_size:每一个batch的图片数
        image_height:图片的高度
        image_width:图片的宽度
        channel:图片的通道数量
        '''
        # 提取出标签，和图片数据
        # x_train ,y_train 均为矩阵
        x_train = data.values.astype(np.float32)
        #y_train = data.label.values.astype(np.int32)
        # 将像素归一化为【0:1】
        #x_train = data.drop('label', axis=1).values.astype(np.float32)
        x_train = x_train / x_train.max()

        # 将y转换为 42000个数据  28*28 1通道的图像
        x_train = x_train.reshape(-1, image_height, image_width, channel)

        self.x_train = x_train
        #self.y_train = y_train
        self.mini_batch_size = batch_size
        self.index_in_epoch = 0
        self.current_epoch = 0.0
        self.select_array = np.array([])
        #总样本个数
        self.num_image = self.x_train.shape[0]

    def next_batch(self):
        '''
        return: x_train_data_batch , y_train_data_batch
        '''
        start = self.index_in_epoch
        self.index_in_epoch += self.mini_batch_size
        end = self.index_in_epoch

        x_tr = self.x_train[start:end]
        #y_tr = self.y_train[start:end]

        #return x_tr,y_tr
        return x_tr

    def data_is_over(self):
        if self.index_in_epoch >= self.num_image:
            return True
        return False


#获得通过神经网络得到的特征，并保存为CSV文件
def get_features():

    train_data = pd.read_csv('data/test.csv')
    logs_train_dir = 'logs/train/'
    feature_batch = 4000

    feature_batch_generator = get_feature_batch(train_data,feature_batch,28,28,1)

    with tf.Graph().as_default() :

        N_CLASSES= 10
        x_train = tf.placeholder(dtype=tf.float32,shape=[feature_batch,28,28,1])

        _ ,outputfeature = inference(x_train, feature_batch, N_CLASSES, 1, 1)

        saver = tf.train.Saver()

        with tf.Session() as sess:


            #导入之前训练好的模型
            print("读取保存节点，导入参数模型....")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("载入成功,global_step is %s" % global_step)


            else:
                print('载入失败')

            print("开始提取特征")


            select_feature = np.array([])

            while (feature_batch_generator.data_is_over()!=True):
                #image_data,image_label = feature_batch_generator.next_batch()
                image_data= feature_batch_generator.next_batch()
                feature = sess.run(outputfeature, feed_dict={x_train:image_data})
                select_feature = np.append(select_feature,feature)
                print('完成4000样本特征提取')

            #将其转化为特征
            select_feature = select_feature.reshape(-1,1024)
            #processed_data = np.insert(select_feature,0,train_data['label'],axis=1)
            #data_frame = pd.DataFrame(processed_data)
            data_frame = pd.DataFrame(select_feature)

            data_frame.to_csv('data/test_data_processed.csv',index=False,sep=',')
            print("完成总特征提取")





# run_training()
# predict_the_test_image()

get_features()