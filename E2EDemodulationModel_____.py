# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import tensorflow as tf
import random as biubiubiu
import matplotlib.pyplot as plt
import struct
import time
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
    h = tf.nn.conv2d(input=x, filters=w, strides=[1,1,1,1], padding='SAME')

    # add global BN after each conv layer
    mean, variance = tf.nn.moments(h, axes = [0, 1, 2])
    return tf.nn.batch_normalization(x=h, mean=mean, variance=variance, \
        offset = None, scale = None, variance_epsilon = 1e-7)


def max_pool_2v2(x):
    return tf.nn.max_pool2d(input=x, ksize=[1,2,2,1], \
                          strides = [1,2,1,1], padding='SAME')


def Upsampling(x, interpolation='bilinear'):
    #shape_of_x = tf.shape(x)
    #x = tf.reshape(x, [-1, shape_of_x[1], shape_of_x[3], 1])
    #sampled_h =  tf.keras.layers.UpSampling2D(\
    #    size=(2, 2), data_format="channels_last", \
    #    interpolation=interpolation)(x)
    #shape_of_h = tf.shape(sampled_h)
    #return tf.reshape(sampled_h, [-1, shape_of_h[2], 1, shape_of_h[2]])

    return tf.keras.layers.UpSampling2D(\
        size=(2, 1), data_format="channels_last", \
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
            h = self.nonlinear_function(conv2d(h, weight) + bias)
        
        
        if self.sampling_method is not None:
            sampled_h = self.sampling_method(h)
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
        self.label = [float(item) for item in label]


class data_manager:
    def __init__(self, data_set_for):
        self.data_set_for = data_set_for
        self.epoch = None
        self.epoch_finished = 0
        self.raw_data = None
    
    def init(self):
        self.raw_data = load_data(self.data_set_for)
    
    def data_normalization(self):
        for i in range(len(self.raw_data)):
            data = self.raw_data[i]
            data_max = max(data.seq)
            data_min = min(data.seq)
            for ii in range(len(data.seq)):
                data.seq[ii] = (data.seq[ii] - data_min) / (data_max - data_min)

    def get_one_epoch(self):
        self.epoch = list(self.raw_data)
        biubiubiu.shuffle(self.epoch)

    def get_epoch_size(self):
        return len(self.raw_data)

    def get_batch_num(self, batch_size):
        return int(len(self.raw_data) / batch_size)
    
    def pop_one_batch(self, batch_size):
        data = list()
        label = list()
        for i in range(batch_size):
            data_ = self.epoch.pop()
            data.append(data_.seq)
            label.append(data_.label)
        return [data, label]


def load_data(data_set_for):
    data_bar = list()
    data_list = list()
    for root, dirs, files in os.walk("D:\\[0]MyFiles\\FilesCache\\DataSet"):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_contributes = filename.split("_")
            modulation_type = file_contributes[0]
            snr = file_contributes[2]

            if snr not in ["3dB", "4dB"] and data_set_for == "training":
                continue
            elif snr in ["3dB", "4dB"] and data_set_for == "testing":
                continue

            rolloff = float(file_contributes[3][1:-4])
            sequence_num = int(file_contributes[1].split("bars")[0])
            
            with open(filepath, 'rb') as f:
                fcontent = f.read()
                data_bar += struct.unpack(sequence_num*4096*'d', fcontent[0:sequence_num*4096*8])
                label = struct.unpack(sequence_num*4*1024*'c', fcontent[sequence_num*4096*8:])
                
                [data_list.append(data_elem(modulation_type, snr, rolloff, \
                data_bar[i*4096:(i+1)*4096], label[i*4096:(i+1)*4096])) for i in range(sequence_num)]
    
    return data_list



def main():
    # construct encoder
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
    
    # construct decoder
    decoder = autocoder()
    
    for i in range(5):
        block_tmp = netblock()
        block_tmp.set_conv_variable([[3,1,32,32], [3,1,32,32]])
        block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
        block_tmp.set_sampling_method(Upsampling)
        
        decoder.set_block(block_tmp)
    

    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,32,32], [3,1,32,32]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(None)
    
    decoder.set_block(block_tmp)
    
    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,32,32], [3,1,32,32]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(None)
    
    decoder.set_block(block_tmp)
    
    # contruct placehold to feed input and get predicted label
    tf.compat.v1.disable_eager_execution()
    received_signal = tf.compat.v1.placeholder(tf.float32, [None, 4096])
    x_input = tf.reshape(received_signal, [-1, 4096, 1, 1])
    true_label = tf.compat.v1.placeholder(tf.float32, [None, 4096])
    y_input = tf.reshape(true_label, [-1, 1024, 4])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    
    # run encoder and decoder, compute cross entropy
    encoder_result = encoder.run(x_input)
    decoded_signal_pdf = decoder.run(encoder_result)
    weight = weight_variable([3, 1, 32, 4])
    bias = bias_variable([4])
    decoded_signal_pdf_after = tf.nn.softmax(conv2d(decoded_signal_pdf, weight) + bias, axis = 3)
    decoded_signal_pdf_last = tf.reshape(decoded_signal_pdf_after, [-1, 1024, 4])

    cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y_input * tf.math.log(decoded_signal_pdf_last), axis=[2]))
    
    # setting trainging step and loss function
    global_step = tf.Variable(0, trainable=False)
    learning_stride = tf.compat.v1.train.exponential_decay(1e-3, global_step, 200, 0.5, staircase = True)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_stride).minimize(cross_entropy)
    
    # computing accuracy when feed testing data
    correct_prediction = tf.equal(tf.argmax(input=y_input, axis=2), tf.argmax(input=decoded_signal_pdf_last, axis=2))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float64))
    
    # initialize tensorflow
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()
    
    # load data for training  
    training_data_manager = data_manager("training")
    training_data_manager.init()
    training_data_manager.data_normalization()
    print("\ntraning data loaded. Totally %d bars.\n" % training_data_manager.get_epoch_size())

    # load data for testing
    testing_data_manager = data_manager("testing")
    testing_data_manager.init()
    testing_data_manager.data_normalization()
    print("\ntesting data loaded. Totally %d bars.\n" % testing_data_manager.get_epoch_size())

    # getting data set size and set training parametres here
    data_set_size = training_data_manager.get_epoch_size()
    epoch_num = 100
    batch_size = 16
    print("The total data size to train is %d. Training %d epochs with batch size of %d" % (data_set_size, epoch_num, batch_size))
    batch_num = training_data_manager.get_batch_num(batch_size)
    
    # plot setting
    plt.ion()
    plt.figure(1)
    t_list = list()
    accuracy_list = list()
    loss_value_list = list()

    # start trainging
    for i in range(epoch_num):
        training_data_manager.get_one_epoch()
        testing_data_manager.get_one_epoch()
        for j in range(batch_num):
            batch_content = training_data_manager.pop_one_batch(batch_size)
            start_time = time.time()
            print("[%s] Start training..."%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            train_op.run(feed_dict={received_signal: batch_content[0], true_label: batch_content[1], keep_prob: 0.5})
            end_time = time.time()
            print("[%s] %d th training finished. Totally cost %d seconds" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i*epoch_num + j + 1, end_time - start_time))

            #if (i * batch_num + j + 1) % 10 == 0:
            #    loss_value = cross_entropy.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1], keep_prob: 0.5})
            #    t_list.append(i * batch_num + j + 1)
            #    loss_value_list.append(loss_value)
            #    plt.plot(t_list, loss_value_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
            #    plt.pause(0.1)

            if (i * batch_num + j + 1) % 10 == 0:
                batch_content = testing_data_manager.pop_one_batch(batch_size)
                train_accuracy = accuracy.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1], keep_prob: 1.0})

                t_list.append(i * batch_num + j + 1)
                accuracy_list.append(train_accuracy)
                
                plt.plot(t_list, accuracy_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
                plt.pause(0.1)
    
    plt.ioff()
    plt.show()
    
if __name__ == "__main__":
    main()







































        
