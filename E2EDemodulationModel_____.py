# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import tensorflow as tf
import random as biubiubiu
import struct
import os

# disbale version 2 tensor
tf.compat.v1.disable_v2_behavior()


def weight_variable(shape):
    initial_state = tf.random.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial_state)


def bias_variable(shape):
    initial_state = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_state)


def conv2d(x, w):
    return tf.nn.conv2d(input=x, filters=w, strides=[1,1,1,1], padding='SAME')


def max_pool_2v2(x):
    return tf.nn.max_pool2d(input=x, ksize=[1,2,2,1], \
                          strides = [1,2,1,1], padding='SAME')


def Upsampling(x, interpolation='nearest'):
    return tf.keras.layers.UpSampling2D(\
        size=(2, 2), data_format="channels_last", \
        interpolation=interpolation)(x)



class netblock:
    def __init__(self):
        self.next_block = None
        
        self.sampling_method = None
        self.nonlinear_function = None
        self.weight = None
        self.bias = None
    
    
    def set_conv_variable(self, shape_list):
        conv_num = len(shape_list)
        self.weight = [weight_variable(shape_list[i]) for i in range(conv_num)]
        self.bias = [bias_variable(shape_list[i][3:4]) for i in range(conv_num)]
    
    
    def set_nonlinaer_function(self, func):
        self.nonlinear_function = func
    
    
    def set_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method
    
    
    def run(self, x):
        h = x
        for weight, bias in zip(self.weight, self.bias):
            h = self.nonlinear_function(conv2d(h, self.weight) + self.bias)
        
        
        if self.sampling_method is not None:
            sampled_h = self.sampleing_method(h)
        else:
            sampled_h = h
        
        
        if self.next_block is not None:
            output = self.next_block.run(sampled_h)
        else:
            output = sampled_h

        return output



class autocoder:
    def __init__(self):
        self.head_block = None
        self.end_block = None
        self.block_num = 0
    
    
    def set_block(self, block):
        if self.head_block is None:
            self.head_block = block
        else:
            self.end_block.next_block = block
        self.end_block = block
        self.block_num += 1
    
    def get_block_num(self):
        return self.block_num
    
    def run(self, x):
        h = self.head_block.run(x)
        return h
    

class data_elem:
    def __init__(self, modulation_type, snr, rolloff, seq, label):
        self.modulation_type = modulation_type
        self.snr = snr
        self.rolloff = rolloff
        self.seq = seq
        self.label = label


class data_manager:
    def __init__(self):
        self.epoch = None
        self.epoch_finished = 0
        self.raw_data = None
    
    def init(self):
        self.raw_data, self.epoch = load_data()
    
    def get_one_epoch(self):
        biubiubiu.shuffle(self.epoch)
    
    def pop_one_batch(self, batch_size):
        data = list()
        label = list()
        for i in range(batch_size):
            data_ = self.epoch.pop()
            data.append(data_.seq)
            label.append(data_.label)
        return 


def load_data():
    data_bar = list()
    data_list = list()
    for root, dirs, files in os.walk("D:\\[0]MyFiles\\FilesCache\\DataSet"):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_contributes = filename.split("_")
            modulation_type = file_contributes[0]
            snr = file_contributes[2]
            rolloff = float(file_contributes[3][1:-4])
            sequence_num = int(file_contributes[1].split("bars")[0])
            
            with open(filepath, 'rb') as f:
                fcontent = f.read()
                data_bar += struct.unpack(sequence_num*4096*'d', fcontent[0:sequence_num*4096*8])
                label = struct.unpack(sequence_num*4*1024*'c', fcontent[sequence_num*4096*8:])
                
                [data_list.append(data_elem(modulation_type, snr, rolloff, \
                data_bar[i*4096:(i+1)*4096], label[i*4:(i+1)*4])) for i in range(sequence_num)]
    
    return data_bar, data_list



def main():
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
    
    decoder = autocoder()
    
    for i in range(5):
        block_tmp = netblock()
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
    
    tf.compat.v1.disable_eager_execution()
    received_signal = tf.compat.v1.placeholder(tf.float64, [None, 4096, 1])
    true_label = tf.compat.v1.placeholder(tf.float64, [None, 1024, 4])
    keep_prob = tf.compat.v1.placeholder(tf.float64)
    
    decoded_signal_pdf = decoder.run(encoder.run(received_signal))
    decoded_signal_pdf = tf.nn.softmax(decoded_signal_pdf, axis = 1)
    
    cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=true_label * tf.math.log(decoded_signal_pdf), axis=[1, 2]))
    
    learning_stride = 1e-2
    train_op = tf.compat.v1.train.AdamOptimizer(learning_stride).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(input=true_label, axis=2), tf.argmax(input=decoded_signal_pdf, axis=2))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float64))
    
    tf.compat.v1.global_variables_initializer().run()
    data_set_size = 50000
    epoch_num = 100
    batch_size = 100
    batch_num = data_set_size / batch_size
    
    
    training_data_manager = data_manager()
    testing_data_manager = data_manager()
    
    training_data_manager.init()
    
    
    for i in range(epoch_num):
        training_data_manager.get_one_epoch()
        for j in range(batch_num):
            batch_content = training_data_manager.pop_one_batch(batch_size) 
            print("running.")
            train_op.run(feed_dict={received_signal: batch_content[0], true_label: batch_content[1], keep_prob: 1.0})
            print("one running finished.")
    
    
if __name__ == "__main__":
    main()







































        
