# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import random as biubiubiu
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import LSTM as lstm
import ExceptionDealingModule
import SignalSimulatorTools as sst


_code_dict = {1:"Illegal input. ",\
              100:"undefined error. "}
_log = ExceptionDealingModule.log("E2EDemodulationModel_LSTM", _code_dict)


max_symbol_len = 10


def create_autocoder(num_cells, input_size, cell_size):
    lstm_layer = lstm.LSTM_layer(num_cells = num_cells)
    lstm_layer.set_size(input_size = input_size, output_size=cell_size)
    return lstm_layer


# it should be noticed here that the default run method is specifically constructed for updated_LSTM_cell_1
def encoder_run(layer, xt, time_step, ct, ht, set_zero_rate):
    hidden_state_list = []
    
    for lstm_cell in layer.lstm_cells:
        lstm_cell.set_initial_hidden_cell_state(ct, ht)


    ct0, ht0, yt = layer.lstm_cells[0].run(xt, time_step, True, set_zero_rate)
    hidden_state_list.append([ct0[:, -1, :], ht0[:, -1, :]])


    for lstm_cell in layer.lstm_cells[1:]:
        ct, ht, yt = lstm_cell.run(yt, time_step, True, set_zero_rate)
        hidden_state_list.append([ct[:, -1, :], ht[:, -1, :]])

    return hidden_state_list



def decoder_run(layer, GO, hidden_state_list, time_step, set_zero_rate):
    symbol_num_this_modulation = sst._bits_per_symbol[modulation_type]
    final_size = 2**symbol_num_this_modulation + 2
    cell_num = layer.get_cell_num()
    cell_state = []
    cell_state.append([hidden_state_list[0][0], hidden_state_list[0][1]])
    if cell_num > 1:
        [cell_state.append([hidden_state_list[i][0], hidden_state_list[i][1]]) for i in range(1, cell_num)]

    yt = GO
    yt_all = None
    for i in range(time_step):
        for ii in range(cell_num):
            ct, ht = cell_state[ii]
            cell_state[ii][0], cell_state[ii][1], yt = layer.lstm_cells[ii].run_at_t(yt, ct, ht, set_zero_rate)

        output_size = yt.get_shape().as_list()[1]
        weight = lstm.weight_variable([output_size, final_size])
        if yt_all is None:
            yt_all = tf.reshape(tf.nn.softmax(tf.linalg.matmul(yt, weight), axis=1), [tf.shape(yt)[0], 1, final_size])
        else:
            yt_all = tf.concat([yt_all, tf.reshape(tf.nn.softmax(tf.linalg.matmul(yt, weight), axis=1), [tf.shape(yt)[0], 1, final_size])], axis = 1)

    return yt_all


def signal_generator(symbols_num, carrier_frequency, filter_span, modulation_type, oversampling_factor, roll_ratio = None, snr = None):
    transer = sst.transmitter()
    transer.set_carrier_frequency(carrier_frequency)
    transer.set_filter_span(filter_span)
    transer.set_modulation_type(modulation_type)
    transer.set_oversamping_factor(oversampling_factor)
    
    if roll_ratio is None:
        roll_ratio = 0.4*biubiubiu.random() + 0.1
    transer.set_roll_ratio(roll_ratio)
    if snr is None:
        snr = biubiubiu.randint(2, 5)
    transer.set_snr(snr)
    transer.init_setting()

    return transer


def padding(batch_signal, symbol_list):
    max_encoder_time_step = max_symbol_len * oversampling_factor * 2
    # decoder padding signal shoule be add one since there are EOS symbol at the end.
    max_decoder_time_step = max_symbol_len + 1

    symbol_num_this_modulation = sst._bits_per_symbol[modulation_type]

    for i in range(len(batch_signal)):
        signal = batch_signal[i]
        if len(signal) < max_encoder_time_step:
            for ii in range(max_encoder_time_step - len(signal)):
                signal.append(0)

    for i in range(len(symbol_list)):
        symbol = symbol_list[i]
        if max_decoder_time_step - len(symbol) > 1:
            for ii in range(max_decoder_time_step - len(symbol) - 1):
                symbol.append(2**symbol_num_this_modulation)
        symbol.append(2**symbol_num_this_modulation + 1)

    return batch_signal, symbol_list


def labelize(symbol_list):
    max_index = symbol_list[0][-1]
    symbol_num = len(symbol_list[0])
    batch_size = len(symbol_list)

    res = [[[0. for iii in range(max_index+1)] for ii in range(symbol_num)] for i in range(batch_size)]

    for i in range(batch_size):
        for ii in range(symbol_num):
            index = symbol_list[i][ii]
            res[i][ii][index] = 1.

    return res

def get_one_batch(batch_size):
    batch_signal = list()
    symbol_list = list()
    for _ in range(batch_size):
        symbols_num = biubiubiu.randint(5, max_symbol_len)
        symbols_num = 5
        snr = 20
        roll_ratio = 0.5
        transer = signal_generator(symbols_num = symbols_num, \
                            carrier_frequency = carrier_frequency, \
                            filter_span = filter_span, \
                            modulation_type = modulation_type, \
                            oversampling_factor = oversampling_factor, \
                            roll_ratio = roll_ratio, \
                            snr = snr)
        batch_signal.append(transer.generate_signal_by_symbol_num(symbols_num = symbols_num))
        symbol_list.append(transer.get_symbols())

    batch_signal, symbol_list = padding(batch_signal, symbol_list)
    label = labelize(symbol_list)


    #for i, signal_item in enumerate(batch_signal):
    #    batch_signal[i] = rollover(signal_item)

    return batch_signal, label


def rollover(signal_item):
    res_signal = []

    for i in range(max_encoder_time_step):
        index = int(input_size_set/2)*i
        res_signal.extend(signal_item[index:index+input_size_set])

    return res_signal



if __name__ == "__main__":
    # signal setting
    carrier_frequency = 2500000
    filter_span = 16
    modulation_type = "QPSK" 
    oversampling_factor = 4
    symbol_num_this_modulation = 2**sst._bits_per_symbol[modulation_type]

    input_size_set = 4
    #max_encoder_time_step = int((max_symbol_len * oversampling_factor * 2) / input_size_set * 2) - 1
    max_encoder_time_step = int((max_symbol_len * oversampling_factor * 2) / input_size_set)
    # decoder padding signal shoule be add one since there are EOS symbol at the end.
    max_decoder_time_step = max_symbol_len + 1

    ## training setting
    #batch_size = 32
    #batch_step = 128
    #epoch_time = 100

    # network construction
    hidden_size = 128
    tf.compat.v1.disable_eager_execution()
    encoder = create_autocoder(2, input_size_set, hidden_size)
    # adding PADDING and EOS symbol
    decoder = create_autocoder(2, hidden_size, hidden_size)
    # setting placeholder for input and label
    x_ = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, input_size_set * max_encoder_time_step])
    x = tf.reshape(x_, [-1, max_encoder_time_step, input_size_set])
    y = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, max_decoder_time_step, symbol_num_this_modulation + 2])
    set_zero_rate = tf.compat.v1.placeholder(dtype = tf.float32)
    ct0 = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, hidden_size])
    ht0 = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, hidden_size])
    # set GO vector for decoder
    GO = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, hidden_size])
    
    # construct graph with encoder and decoder
    hidden_state_list = encoder_run(encoder, x, max_encoder_time_step, ct0, ht0, set_zero_rate)
    yt = decoder_run(decoder, GO, hidden_state_list, max_decoder_time_step, set_zero_rate)
    # set loss function
    # training setting
    batch_size = 128
    batch_num = 64
    epoch_time = 200
    cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y * tf.math.log(yt) / tf.math.log(10.), axis=[2]))
    global_step = tf.Variable(0, trainable=False)
    learning_stride = tf.compat.v1.train.exponential_decay(1e-2, global_step, batch_num*10, 0.5, staircase = True)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_stride).minimize(cross_entropy, global_step=global_step)
    
    correct_prediction = tf.math.equal(tf.argmax(input=y, axis=2), tf.argmax(input=yt, axis=2))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float64))

    training_set = list()
    for ii in range(batch_num):
        batch_signal, label = get_one_batch(batch_size)
        for signal_item, label_item in zip(batch_signal, label):
            training_set.append([signal_item, label_item])


    plt.ion()
    figure_full = plt.figure(num = 1, figsize = [8,16])
    ax1 = figure_full.add_subplot(3, 1, 1)
    ax2 = figure_full.add_subplot(3, 1, 2)
    ax3 = figure_full.add_subplot(3, 1, 3)
    zeros_init = [[0. for i in range(hidden_size)] for ii in range(batch_size)]
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()


        # plot setting
        t_list = list()
        accuracy_list = list()
        loss_value_list = list()
        learning_list = list()

        for i in range(epoch_time):

            biubiubiu.shuffle(training_set)
            print("[%s] Start %dth training..."%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i))
            for ii in range(batch_num):
                # generating training signal batch
                batch_signal = []
                label = []
                for item in training_set[ii*batch_size:(ii+1)*batch_size]:
                    batch_signal.append(item[0])
                    label.append(item[1])
                #batch_signal, label = get_one_batch(batch_size)
                train_op.run(feed_dict={x_: batch_signal, y: label, \
                    ct0: zeros_init, ht0: zeros_init, GO: zeros_init, \
                                        set_zero_rate: 0.1})
            print("[%s] END %dth training..."%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i))
            
            loss_value = cross_entropy.eval(feed_dict={x_: batch_signal, y: label, \
                    ct0: zeros_init, ht0: zeros_init, GO: zeros_init, \
                                        set_zero_rate: 0.})
            t_list.append(i + 1)
            loss_value_list.append(loss_value)
            ax1.plot(t_list, loss_value_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
            ax1.set_xlabel("Training epoch")
            ax1.set_ylabel("Loss function")
            plt.pause(0.1)


            batch_signal, label = get_one_batch(batch_size)
            train_accuracy = accuracy.eval(feed_dict={x_: batch_signal, y: label, \
                    ct0: zeros_init, ht0: zeros_init, GO: zeros_init, \
                                        set_zero_rate: 0.})
            accuracy_list.append(train_accuracy)
                
            ax2.plot(t_list, accuracy_list,c='k',ls='-.', marker='*', mec='r',mfc='w')
            ax2.set_xlabel("Training epoch")
            ax2.set_ylabel("Accuracy")
            plt.pause(0.1)

            learning_rate = learning_stride.eval(feed_dict={x_: batch_signal, y: label, \
                    ct0: zeros_init, ht0: zeros_init, GO: zeros_init, \
                                        set_zero_rate: 0.})
            learning_list.append(learning_rate.tolist())
            ax3.plot(t_list, learning_list, c='k',ls='-.', marker='*', mec='r',mfc='w')
            ax3.set_xlabel("Training epoch")
            ax3.set_ylabel("Learning stride")
            plt.pause(0.1)
        plt.ioff() 
        plt.savefig("TEMPPIG.png")