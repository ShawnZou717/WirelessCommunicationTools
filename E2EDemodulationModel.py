# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import tensorflow as tf
import random as biubiubiu
import struct
import os

# 创建tensorflow权重变量
def weight_variable(shape):
    initial_state = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial_state)

# 创建tensorflow偏置变量
def bias_variable(shape):
    initial_state = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_state)

# 卷积操作
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

# 最大值池化（下采样）
def max_pool_2v2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], \
                          strides = [1,2,1,1], padding='SAME')

# 最邻近采样（上采样）
def Upsampling(x, interpolation='nearest'):
    return tf.keras.layers.UpSampling2D(\
        size=(2, 2), data_format="channels_last", \
        interpolation=interpolation)(x)


# 卷积块类
class netblock:
    def __init__(self):
        self.next_block = None
        
        self.sampling_method = None
        self.nonlinear_function = None
        self.weight = None
        self.bias = None
    
    # 设置卷积块卷积层数、权重变量与偏置变量
    def set_conv_variable(self, shape_list):
        conv_num = len(shape_list)
        self.weight = [weight_variable(shape_list[i]) for i in range(conv_num)]
        self.bias = [bias_variable(shape_list[i][3:4]) for i in range(conv_num)]
    
    # 设置激活函数
    def set_nonlinaer_function(self, func):
        self.nonlinear_function = func
    
    # 设置采样方法
    def set_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method
    
    # 开始计算
    def run(self, x):
        h = x
        for weight, bias in zip(self.weight, self.bias):
            h = self.nonlinear_function(conv2d(h, self.weight) + self.bias)
        
        # 可以定义无采样的卷积层
        if self.sampling_method is not None:
            sampled_h = self.sampleing_method(h)
        else:
            sampled_h = h
        
        # 本卷积块计算结束，调用下一个卷积块进行计算
        if self.next_block is not None:
            output = self.next_block.run(sampled_h)
        else:
            output = sampled_h

        return output


# 解码编码器类
class autocoder:
    def __init__(self):
        self.head_block = None
        self.end_block = None
        self.block_num = 0
    
    # 添加卷积块
    def set_block(self, block):
        if self.head_block is None:
            self.head_block = block
        else:
            self.end_block.next_block = block
        self.end_block = block
        self.block_num += 1
    
    # 获取编码器解码器卷积块数目
    def get_block_num(self):
        return self.block_num
    
    # 启动编码器解码器
    def run(self, x):
        h = self.head_block.run(x)
        return h
    

# 被数据集管理类管理的数据类，每条数据包含调制类型、信噪比
# 滚降系数、信号数据、标签5个属性
class data_elem:
    def __init__(self, modulation_type, snr, rolloff, seq, label):
        self.modulation_type = modulation_type
        self.snr = snr
        self.rolloff = rolloff
        # 信号数据长度为4096
        self.seq = seq
        # 标签数据长度为4(如果使用符号数更大的调制方式，则需要在load_data中修改label长度)
        self.label = label


# 数据集管理类
class data_manager:
    def __init__(self):
        self.epoch = None
        self.epoch_finished = 0
        self.raw_data = None
    
    # 读取数据集
    def init(self):
        self.raw_data, self.epoch = load_data()
    
    # 获取一个新的epoch
    def get_one_epoch(self):
        biubiubiu.shuffle(self.epoch)
    
    # 从epoch中抓取一个batch
    def pop_one_batch(self, batch_size):
        data = list()
        label = list()
        for i in range(batch_size):
            data_ = self.epoch.pop()
            data.append(data_.seq)
            label.append(data_.label)
        return 


# 读取二进制数据集，并解码(以文本形式存储数据集更大一些，所以存储时采用二进制数据存储)
def load_data():
    data_bar = list()
    data_list = list()
    for root, dirs, files in os.walk("D:\\[0]MyFiles\\FilesCache\\DataSet"):
        for filename in files:
            filepath = os.path.join(root, filename)
            # 0为调制方式，1为数据条数，2为信噪比，3为滚降系数
            file_contributes = filename.split("_")
            modulation_type = file_contributes[0]
            snr = file_contributes[2]
            rolloff = float(file_contributes[3][1:-4])
            sequence_num = int(file_contributes[1].split("bars")[0])
            
            with open(filepath, 'rb') as f:
                fcontent = f.read()
                # unpack信号数据与标签数据
                data_bar += struct.unpack(sequence_num*4096*'d', fcontent[0:sequence_num*4096*8])
                label = struct.unpack(sequence_num*4*'c', fcontent[sequence_num*4096*8:])
                
                [data_list.append(data_elem(modulation_type, snr, rolloff, \
                data_bar[i*4096:(i+1)*4096], label[i*4:(i+1)*4])) for i in range(sequence_num)]
    
    # 返回读取的原始数据与经过排列的数据集
    return data_bar, data_list



def main():
    # 设置encoder
    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,1,32], [3,1,32,32]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(max_pool_2v2)
    
    encoder = autocoder()
    encoder.set_block(block_tmp)
    
    for i in range(6):
        block_tmp = netblock()
        block_tmp.set_conv_variable([[3,1,32,32], [3,1,32,32]])
        block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
        block_tmp.set_sampling_method(max_pool_2v2)
        
        encoder.set_block(block_tmp)
    
    # 设置decoder
    decoder = autocoder()
    
    for i in range(5):
        block_tmp = netblock()
        # 由于解码器采用上采样，所以卷积层卷积核个数与深度需要随之改变
        block_tmp.set_conv_variable([[3,1,32*2**i,32*2**i], [3,1,32*2**i,32*2**i]])
        block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
        block_tmp.set_sampling_method(Upsampling)
        
        decoder.set_block(block_tmp)
    

    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,1024,256], [3,1,256,64]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(None)
    
    decoder.set_block(block_tmp)
    
    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,64,16], [3,1,16,4]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(None)
    
    decoder.set_block(block_tmp)
    
    # 设置模型输入输出，并定义精确度与损失函数
    received_signal = tf.placeholder(tf.float64, [None, 4096, 1])
    true_label = tf.placeholder(tf.float64, [None, 1024, 4])
    keep_prob = tf.placeholder(tf.float64)
    
    decoded_signal_pdf = decoder.run(encoder.run(received_signal))
    decoded_signal_pdf = tf.nn.softmax(decoded_signal_pdf, axis = 1)
    
    # 损失函数采用交叉熵函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(true_label * tf.log(decoded_signal_pdf), [1, 2]))
    
    learning_stride = 1e-2
    train_op = tf.train.AdamOptimizer(learning_stride).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(true_label, 2), tf.argmax(decoded_signal_pdf, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    
    # 训练与测试参数设定
    tf.global_variables_initializer().run()
    data_set_size = 50000
    epoch_num = 100
    batch_size = 100
    batch_num = data_set_size / batch_size
    
    
    # 初始化数据集管理器
    training_data_manager = data_manager()
    testing_data_manager = data_manager()
    
    training_data_manager.init()
    
    
    # 开始训练
    for i in range(epoch_num):
        training_data_manager.get_one_epoch()
        for j in range(batch_num):
            batch_content = training_data_manager.pop_one_batch(batch_size) 
            
            train_op.run(feed_dict={received_signal: batch_content[0], true_label: batch_content[1], keep_prob: 1.0})
    
    
if __name__ == "__main__":
    main()







































        