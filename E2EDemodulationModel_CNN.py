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

from tensorflow.python.ops.gen_data_flow_ops import padding_fifo_queue

# disbale version 2 tensor
tf.compat.v1.disable_v2_behavior()


def weight_variable(shape):
    initial_state = tf.random.uniform(shape, minval= -1, maxval = 1)
    return tf.Variable(initial_state)


def bias_variable(shape):
    initial_state = tf.random.uniform(shape, minval = -1, maxval = 1)
    #initial_state = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_state)


def conv2d(x, w):
    h = tf.nn.conv2d(input=x, filters=w, strides=[1,1,1,1], padding='SAME')
    # return h

    # add global BN after each conv layer
    mean, variance = tf.nn.moments(h, axes = [0, 1, 2])
    return tf.nn.batch_normalization(x=h, mean=mean, variance=variance, \
        offset = tf.Variable(tf.random.uniform([1])), scale = tf.Variable(tf.random.uniform([1])), variance_epsilon = 1e-7)

def conv2d_4stride(x, w):
    h = tf.nn.conv2d(input=x, filters=w, strides=[1,4,1,1], padding='SAME')
    # return h

    # add global BN after each conv layer
    mean, variance = tf.nn.moments(h, axes = [0, 1, 2])
    return tf.nn.batch_normalization(x=h, mean=mean, variance=variance, \
        offset = tf.Variable(tf.random.uniform([1])), scale = tf.Variable(tf.random.uniform([1])), variance_epsilon = 1e-7)


def conv2d_sec(x, w):
    h = tf.nn.conv2d(input=x, filters=w, strides=[1,4,1,1], padding='SAME')
    # return h

    # add global BN after each conv layer
    mean, variance = tf.nn.moments(h, axes = [0, 1, 2])
    return tf.nn.batch_normalization(x=h, mean=mean, variance=variance, \
        offset = tf.Variable(tf.random.uniform([1])), scale = tf.Variable(tf.random.uniform([1])), variance_epsilon = 1e-7)


def max_pooling(x):
    return tf.nn.max_pool2d(input=x, ksize=[1,4,1,1], \
                          strides = [1,2,1,1], padding='SAME')


def Upsampling(x, concat_tensor = None):
    #shape_of_x = tf.shape(x)
    #x = tf.reshape(x, [-1, shape_of_x[1], shape_of_x[3], 1])
    #sampled_h =  tf.keras.layers.UpSampling2D(\
    #    size=(2, 2), data_format="channels_last", \
    #    interpolation=interpolation)(x)
    #shape_of_h = tf.shape(sampled_h)
    #return tf.reshape(sampled_h, [-1, shape_of_h[2], 1, shape_of_h[2]])

    if concat_tensor is not None:
        x = tf.concat([x, concat_tensor], 3)

    return tf.keras.layers.UpSampling2D(\
        size=(2, 1), data_format="channels_last", \
        interpolation='bilinear')(x)


def Upsampling_unconv(x, concat_tensor = None):
    #shape_of_x = tf.shape(x)
    #x = tf.reshape(x, [-1, shape_of_x[1], shape_of_x[3], 1])
    #sampled_h =  tf.keras.layers.UpSampling2D(\
    #    size=(2, 2), data_format="channels_last", \
    #    interpolation=interpolation)(x)
    #shape_of_h = tf.shape(sampled_h)
    #return tf.reshape(sampled_h, [-1, shape_of_h[2], 1, shape_of_h[2]])
    if concat_tensor is not None:
        x = tf.concat([x, concat_tensor], 3)

    output_shape = [tf.shape(x)[0], tf.shape(x)[1]*2, tf.shape(x)[2], tf.shape(x)[3]]
    batch, h, w, channels = x.get_shape().as_list()

    filter_w = weight_variable([3, 1, channels, channels])
    h = tf.nn.conv2d_transpose(input = x, filters = filter_w, output_shape = output_shape, padding = "SAME", strides = [1,2,1,1])
    return h


class netblock:
    def __init__(self):
        self.next_block = None
        
        self.sampling_method = None
        self.nonlinear_function = None
        self.weight = None
        self.bias = None
    
    
    # input shape_list should be a list of shape, for example shape_list = [[1,1,1,1], [2,2,1,32]]
    def set_conv_variable(self, shape_list):
        conv_num = len(shape_list)
        self.weight = [weight_variable(shape_list[i]) for i in range(conv_num)]
        self.bias = [bias_variable(shape_list[i][3:4]) for i in range(conv_num)]
    
    
    def set_nonlinaer_function(self, func):
        self.nonlinear_function = func
    
    
    def set_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method
    
    
    def run(self, x, concat_tensor = None):
        h = x
        for weight, bias in zip(self.weight, self.bias):
            h = self.nonlinear_function(conv2d(h, weight) + bias)
        
        
        if self.sampling_method is not None:
            if concat_tensor is None:
                sampled_h = self.sampling_method(h)
            else:
                sampled_h = self.sampling_method(h, concat_tensor)
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

    def get_one_epoch(self, snr = None, modulation_type = None):
        if snr is None and modulation_type is None:
            self.epoch = list(self.raw_data)
            biubiubiu.shuffle(self.epoch)
        elif snr is not None and modulation_type is not None:
            self.epoch = list()
            for data_item in self.raw_data:
                if data_item.snr == snr and data_item.modulation_type == modulation_type:
                    self.epoch.append(data_item)
            #biubiubiu.shuffle(self.epoch)
        else:
            sys.exit("SNR or Modulation_type not defined.")

    def get_epoch_size(self):
        return len(self.raw_data)

    def get_batch_num(self, batch_size):
        return int(len(self.epoch) / batch_size)
    
    def pop_one_batch(self, batch_size):
        data = list()
        label = list()
        for i in range(batch_size):
            data_ = self.epoch.pop()
            data.append(data_.seq)
            label.append(data_.label)
        return [data, label]

def load_data(data_set_for):
    data_list = list()
    for root, dirs, files in os.walk("D:\\[0]MyFiles\\FilesCache\\DataSet"):
        for filename in files:
            data_bar = list()
            filepath = os.path.join(root, filename)
            file_contributes = filename.split("_")
            modulation_type = file_contributes[0]
            snr = file_contributes[2]

            if snr not in ["3dB", "4dB"] and data_set_for == "training":
                continue
            #elif snr in ["3dB", "4dB"] and data_set_for == "testing":
            #    continue

            rolloff = float(file_contributes[3][1:-4])
            sequence_num = int(file_contributes[1].split("bars")[0])
            
            with open(filepath, 'rb') as f:   
                fcontent = f.read()
                data_bar += struct.unpack(sequence_num*4096*'d', fcontent[0:sequence_num*4096*8])
                label = struct.unpack(sequence_num*4*1024*'c', fcontent[sequence_num*4096*8:])
                
                [data_list.append(data_elem(modulation_type, snr, rolloff, \
                data_bar[i*4096:(i+1)*4096], label[i*4096:(i+1)*4096])) for i in range(sequence_num)]
    
    return data_list


def save_list(bar, file_name):
    if not isinstance(bar, list):
        with open(file_name, 'a') as f:
                f.write(str(bar) + "\n")
    else:
        for item in bar:
            save_list(item, file_name)

        


def main():
    # construct encoder
    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,1,32], [3,1,32,32]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(max_pooling)
    
    encoder = list()
    encoder.append(block_tmp)
    
    for i in range(6):
        block_tmp = netblock()
        block_tmp.set_conv_variable([[3,1,32,32], [3,1,32,32]])
        block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
        block_tmp.set_sampling_method(max_pooling)
        
        encoder.append(block_tmp)
    
    # construct decoder
    decoder = list()
    
    for i in range(5):
        block_tmp = netblock()
        block_tmp.set_conv_variable([[3,1,32*(i+1),32*(i+1)], [3,1,32*(i+1),32*(i+1)]])
        block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
        #block_tmp.set_sampling_method(Upsampling)
        block_tmp.set_sampling_method(Upsampling_unconv)
        
        decoder.append(block_tmp)
    

    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,192,192], [3,1,192,192]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(Upsampling_unconv)
    
    decoder.append(block_tmp)
    
    block_tmp = netblock()
    block_tmp.set_conv_variable([[3,1,224,224], [3,1,224,224]])
    block_tmp.set_nonlinaer_function(tf.nn.leaky_relu)
    block_tmp.set_sampling_method(Upsampling_unconv)
    
    decoder.append(block_tmp)
    
    # contruct placehold to feed input and get predicted label
    tf.compat.v1.disable_eager_execution()
    received_signal = tf.compat.v1.placeholder(tf.float32, [None, 4096], name = "input_tensor")
    x_input = tf.reshape(received_signal, [-1, 4096, 1, 1])
    true_label = tf.compat.v1.placeholder(tf.float32, [None, 4096], name = "input_true_label")
    y_input = tf.reshape(true_label, [-1, 1024, 1, 4])
    #keep_prob = tf.compat.v1.placeholder(tf.float32)
    
    # run encoder and decoder, compute cross entropy
    encoder_result = list()
    h = x_input
    for block_item in encoder:
        h = block_item.run(h)
        encoder_result.append(h)

    decoder_result = list()
    for i in range(len(decoder)):
        block_item = decoder[i]
        #if i < 5:
        #    h = block_item.run(h, encoder_result[len(encoder) - 1 - i])
        #elif i < len(encoder) - 1:
        #    h = block_item.run(h)
        #    h = tf.concat([h, encoder_result[len(encoder) - 1 - i]], 3)
        #else:
        #    h = block_item.run(h)
        h = block_item.run(h, encoder_result[len(encoder) - 1 - i])

        decoder_result.append(h)

    decoded_signal_pdf = h
    weight = weight_variable([4, 1, 256, 4])
    bias = bias_variable([4])
    #decoded_signal_pdf_last = tf.nn.softmax(conv2d(decoded_signal_pdf, weight) + bias, axis = 3)
    decoded_signal_pdf_last = tf.nn.softmax(conv2d_4stride(decoded_signal_pdf, weight) + bias, axis = 3)

    #decoded_signal_pdf_last = tf.reshape(decoded_signal_pdf_after, [-1, 1024, 4])
    #decoded_signal_pdf_last = decoded_signal_pdf_after
    cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y_input * tf.math.log(decoded_signal_pdf_last) / tf.math.log(10.), axis=[3]))
    
    
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
    batch_size = 64
    print("The total data size to train is %d. Training %d epochs with batch size of %d" % (data_set_size, epoch_num, batch_size))
    training_data_manager.get_one_epoch()
    batch_num = training_data_manager.get_batch_num(batch_size)
    

    # setting trainging step and loss function
    global_step = tf.Variable(0, trainable=False)
    learning_stride = tf.compat.v1.train.exponential_decay(1e-3, global_step, batch_num*10, 0.5, staircase = True)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_stride).minimize(cross_entropy, global_step=global_step)
    #train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_stride).minimize(cross_entropy, global_step=global_step)   
    
    # computing accuracy when feed testing data
    #correct_prediction = tf.equal(tf.argmax(input=y_input, axis=2), tf.argmax(input=decoded_signal_pdf_last, axis=2))
    correct_prediction = tf.math.equal(tf.argmax(input=y_input, axis=3), tf.argmax(input=decoded_signal_pdf_last, axis=3))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float64))


    # plot setting
    plt.ion()
    figure_full = plt.figure(num = 1, figsize = [12,4])
    ax1 = figure_full.add_subplot(3, 1, 1)
    ax2 = figure_full.add_subplot(3, 1, 2)
    ax3 = figure_full.add_subplot(3, 1, 3)
    t_list = list()
    accuracy_list = list()
    loss_value_list = list()
    learning_list = list()


    saver = tf.compat.v1.train.Saver()

    # initialize tensorflow
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()

        # start trainging
        for i in range(epoch_num):
            training_data_manager.get_one_epoch()
            batch_num = training_data_manager.get_batch_num(batch_size)
            start_time = time.time()
            print("[%s] Start training..."%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            for j in range(batch_num):
                batch_content = training_data_manager.pop_one_batch(batch_size)
                train_op.run(feed_dict={received_signal: batch_content[0], true_label: batch_content[1]})


                if j == batch_num - 1:
                    loss_value = cross_entropy.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1]})
                    t_list.append(i + 1)
                    loss_value_list.append(loss_value)
                    ax1.plot(t_list, loss_value_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
                    ax1.set_xlabel("Training epoch")
                    ax1.set_ylabel("Loss function")
                    plt.pause(0.1)


                    testing_data_manager.get_one_epoch()
                    batch_content = testing_data_manager.pop_one_batch(64)
                    train_accuracy = accuracy.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1]})
                    accuracy_list.append(train_accuracy)
                
                    ax2.plot(t_list, accuracy_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
                    ax2.set_xlabel("Training epoch")
                    ax2.set_ylabel("Accuracy")
                    plt.pause(0.1)

                    learning_rate = learning_stride.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1]})
                    learning_list.append(learning_rate.tolist())
                    ax3.plot(t_list, learning_list, c='k',ls='-.', marker='*', mec='r',mfc='w')
                    ax3.set_xlabel("Training epoch")
                    ax3.set_ylabel("Learning stride")
                    plt.pause(0.1)

                    #lalala = decoded_signal_pdf_last.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1]})

                    #lalala = lalala.tolist()
                    #save_list(lalala, "predicted.dat")
                    #lalala = batch_content[1]
                    #save_list(lalala, "true.dat")
                    #pass

            end_time = time.time()
            print("[%s] %d th training finished. Totally cost %d seconds" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, end_time - start_time))


        #saver.save(sess, 'E2EDemodulatorModel\\model.ckpt')
        saver.save(sess, 'QPSK_E2EDemodulatorModel_unconv111\\model.ckpt')
        plt.ioff()
        plt.show()



def main2():
    saver = tf.compat.v1.train.import_meta_graph("QPSK_E2EDemodulatorModel_unconv\\model.ckpt.meta")
    graph = tf.compat.v1.get_default_graph()
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]

    received_signal = graph.get_tensor_by_name("input_tensor:0")
    true_label = graph.get_tensor_by_name("input_true_label:0")
    
    y_input = tf.reshape(true_label, [-1, 1024, 1, 4])
    predicted_pdf = graph.get_tensor_by_name("Softmax:0")

    correct_prediction = tf.math.equal(tf.argmax(input=y_input, axis=3), tf.argmax(input=predicted_pdf, axis=3))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float64))

    testing_data_manager = data_manager("testing")
    testing_data_manager.init()
    testing_data_manager.data_normalization()
    print("\ntesting data loaded. Totally %d bars.\n" % testing_data_manager.get_epoch_size())

    data_set_size = testing_data_manager.get_epoch_size()
    epoch_num = 1
    batch_size = 100
    print("The total data size to train is %d. Training %d epochs with batch size of %d" % (data_set_size, epoch_num, batch_size))
    

    plt.ion()
    figure_full = plt.figure(num = 1, figsize = [4,4])
    ax1 = figure_full.add_subplot(1, 1, 1)

    # initialize tensorflow
    snr_list = ["-2dB", "-1dB", "0dB", "1dB", "2dB", "3dB", "4dB", "5dB", "6dB", "7dB", "8dB"]
    snr_list.reverse()
    #snr_list = ["3dB", "4dB"]
    with tf.compat.v1.Session() as sess:
        #tf.compat.v1.global_variables_initializer().run()
        saver.restore(sess, tf.train.latest_checkpoint("QPSK_E2EDemodulatorModel_unconv\\"))
        # start trainging
        for snr_value in snr_list:
            t_list = list()
            accuracy_list = list()
            for i in range(epoch_num):
                testing_data_manager.get_one_epoch(snr = snr_value, modulation_type = "QPSK")

                #testing_data_manager.get_one_epoch(snr = None, modulation_type = None)
                batch_num = testing_data_manager.get_batch_num(batch_size)
                start_time = time.time()
                print("[%s] Start training..."%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                for j in range(batch_num):
                    batch_content = testing_data_manager.pop_one_batch(batch_size)

                    t_list.append(j + 1)

                    train_accuracy = accuracy.eval(feed_dict={received_signal: batch_content[0], true_label: batch_content[1]})
                    accuracy_list.append(train_accuracy)
                
                    ax1.plot(t_list, accuracy_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
                    ax1.set_xlabel("Testing Time")
                    ax1.set_ylabel("Accuracy")
                    plt.pause(0.1)


                end_time = time.time()
            print("[%s] %d th training finished. Totally cost %d seconds" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, end_time - start_time))
            print("Accuracy of %s is %f."%(snr_value, sum(accuracy_list)/len(accuracy_list)))

        plt.ioff()
        plt.show()

if __name__ == "__main__":
    #main()
    main2()







































        
