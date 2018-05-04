
# coding: utf-8

# # 采用TensorFlow 来搭建CNN进行手写数字预测

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os


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


def inference(images , batch_size ,n_classes):
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
                                   shape = [dim,64],
                                   dtype = tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases  = tf.get_variable('biases',
                                   shape = [64],
                                   dtype = tf.float32,
                                   initializer= tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
     
    #full-connect2
    with tf.variable_scope('fc2') as scope:
        
        weights = tf.get_variable('weights',
                                   shape=[64,64],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases  = tf.get_variable('biases',
                                  shape=[64],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,weights)+biases,name=scope.name)
        
        
    #soft-max output
    with tf.variable_scope('soft_max') as scope:
        
        weights =tf.get_variable('weights',
                                 shape = [64,n_classes],
                                 dtype =tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2,weights),biases,name=scope.name)
        
    
    return softmax_linear
        


# In[4]:


def losses(logits,labels):
    '''
    参数说明:
    logits:logits tensor ,float ,[batch_size,n_classes]
    labels:label  tensor ,tf.int32,[batch_size]
    
    返回值:
    loss tensor float type
     
    '''
    
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss
     


# In[5]:


def trainning(loss,learning_rate):
    
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op =optimizer.minimize(loss,global_step=global_step)
    return train_op


# In[6]:


def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy


# #  对模型进行训练

# In[7]:


#每个minibatch的图片数量
BATCH_SIZE = 50
#图片的长和宽
IMG_H = 28
IMG_W = 28
n_classes = 10
MAX_STEP  = 50000
learning_rate = 0.0001


# In[8]:


def run_training():
    train_data = pd.read_csv('data/train.csv')
    logs_train_dir = 'logs/train/'
    
    
    x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,IMG_W,IMG_H,1])
    y_= tf.placeholder(tf.int64,shape=[BATCH_SIZE])
    
    
    batch_generater = get_mini_batch(train_data,BATCH_SIZE,IMG_H,IMG_W,channel=1)
    logits  = inference(x,BATCH_SIZE,n_classes)
    loss    = losses(logits,y_)
    acc     = evaluation(logits,y_)
    train_op = trainning(loss,learning_rate=learning_rate)
    

    with tf.Session()  as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # summary_op =tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
        print('Start Trainning')
        try:
            for step in np.arange(MAX_STEP):
                tra_image ,tra_label = batch_generater.next_batch()
                ll, _ , tra_loss ,tra_acc =sess.run([logits,train_op,loss,acc],
                                                feed_dict={x:tra_image,y_:tra_label})
                if (batch_generater.current_epoch - int(batch_generater.current_epoch))==0:
                    print('Epoch %d , train loss = %.2f , train accuracy=%.2f%%'%(int(batch_generater.current_epoch),tra_loss,tra_acc*100))

                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    # with tf.Graph().as_default():
                    #     summary_str = sess.run(summary_op)
                    #     train_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step+1)==MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done trainning -- epoch limit reached')


class get_test_image(object):
    def __init__(self, data, batch_size=4000, image_height=28, image_width=28, channel=1):
        '''
        data:DataFrame数据类型，为读入的数据，第一列为标签，之后为像素数
        batch_size:每一个batch的图片数
        image_height:图片的高度
        image_width:图片的宽度
        channel:图片的通道数量
        '''
        # 提取出标签，和图片数据
        # x_train 为矩阵


        x_train = data.values.astype(np.float32)
        x_train = x_train / x_train.max()

        # 将y转换为 42000个数据  28*28 1通道的图像
        x_train = x_train.reshape(-1, image_height, image_width, channel)

        self.x_train = x_train
        self.mini_batch_size = batch_size
        self.index = 0
        self.num = x_train.shape[0]

    def next_batch(self):
        '''
        return: x_train_data_batch , y_train_data_batch
        '''
        start =self.index
        end   =self.index+self.mini_batch_size
        x_tr = self.x_train[start:end]
        self.index += self.mini_batch_size
        return x_tr
    def data_is_over(self):
        if self.index==self.num:
            return True
        else:
            return False



# In[9]:

def predict_the_test_image():

    test_data = pd.read_csv('data/test.csv')
    logs_train_dir = 'logs/train/'
    test_image_generator = get_test_image(test_data)

    with tf.Graph().as_default():
        BATCH_SIZE = test_image_generator.mini_batch_size
        N_CLASSES  = 10



        x_test = tf.placeholder(tf.float32,shape=[BATCH_SIZE,28,28,1])

        logits = inference(x_test,BATCH_SIZE,N_CLASSES)

        logits = tf.nn.softmax(logits)

        saver =tf.train.Saver()

        with tf.Session() as sess:

            print("读取保存节点，导入参数模型....")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("载入成功,global_step is %s" % global_step)


            else:
                print('载入失败')

            predict_label = []
            num_image = test_image_generator.num

            while (test_image_generator.data_is_over()==False):
                test_image = test_image_generator.next_batch()
                prediction = sess.run(logits,feed_dict={x_test:test_image})
                max_index = np.argmax(prediction,axis=1)
                max_index.reshape([1, -1])
                predict_label.extend(max_index)
                print('完成4000个样本预测')


            print("完成全部预测")
            submission = pd.DataFrame({'ImageId':list(range(1,num_image+1)),'Label':predict_label})
            submission.to_csv('data/cnn_sub.csv',index=False,sep=',')


run_training()
predict_the_test_image()