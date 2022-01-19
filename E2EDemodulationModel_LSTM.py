# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import random as biubiubiu

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
def encoder_run(layer, xt, time_step):
    ct0, ht0, yt = layer.lstm_cells[0].run(xt, time_step, True)
        
    for lstm_cell in layer.lstm_cells[1:]:
        ct, ht, yt = lstm_cell.run(yt, time_step, True)

    return ct0, ht0, ct, ht



def decoder_run(layer, GO, hidden_state_list, time_step):
    ct, ht, yt = list(), list(), list()

    ct, ht, yt = layer.lstm_cells[0].run_at_t(GO, hidden_state_list[0][0], hidden_state_list[0][1])
        
    for i in range(1, len(layer.lstm_cells)):
        lstm_cell = layer.lstm_cells[i]
        ct, ht, yt = lstm_cell.run_at_t(yt, hidden_state_list[i][0], hidden_state_list[i][1])

    for i in range(1, time_step, 1):
        ct, ht, yt = layer.lstm_cells[0].run_at_t(yt)
        
        for lstm_cell in layer.lstm_cells[1:]:
            ct, ht, yt = lstm_cell.run_at_t(yt)

    return yt


def signal_generator(symbols_num, carrier_frequency, filter_span, modulation_type, oversampling_factor, roll_ratio = None, snr = None):
    transer = sst.transmitter()
    transer.set_carrier_frequency(carrier_frequency)
    transer.set_filter_span(filter_span)
    transer.set_modulation_type(modulation_type)
    transer.set_oversamping_factor(oversampling_factor)
    
    if roll_ratio is not None:
        transer.set_roll_ratio(roll_ratio)
    if snr is not None:
        transer.set_roll_ratio(snr)
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
        symbols_num = biubiubiu.randint(5,max_symbol_len)
        snr = biubiubiu.randint(2,5)
        transer = signal_generator(symbols_num = symbols_num, \
                            carrier_frequency = carrier_frequency, \
                            filter_span = filter_span, \
                            modulation_type = modulation_type, \
                            oversampling_factor = oversampling_factor, \
                            snr = snr)
        batch_signal.append(transer.generate_signal_by_symbol_num(symbols_num = symbols_num))
        symbol_list.append(transer.get_symbols())

    batch_signal, symbol_list = padding(batch_signal, symbol_list)
    label = labelize(symbol_list)

    return batch_signal, label


if __name__ == "__main__":
    # signal setting
    carrier_frequency = 2500000
    filter_span = 16
    modulation_type = "QPSK" 
    oversampling_factor = 4
    symbol_num_this_modulation = 2**sst._bits_per_symbol[modulation_type]

    max_encoder_time_step = max_symbol_len * oversampling_factor * 2
    # decoder padding signal shoule be add one since there are EOS symbol at the end.
    max_decoder_time_step = max_symbol_len + 1

    # network construction
    encoder = create_autocoder(2, 1, 128)
    # adding PADDING and EOS symbol
    decoder = create_autocoder(2, 128, symbol_num_this_modulation + 2)
    x = tf.placeholder(dtype = tf.float64, shape = [None, max_encoder_time_step])
    y = tf.placeholder(dtype = tf.float64, shape = [None, max_decoder_time_step, symbol_num_this_modulation + 2])

    # training setting
    batch_size = 32
    batch_step = 128
    epoch_time = 100
    for i in range(epoch_time):
        for ii in range(batch_step):
            # generating training signal batch
            batch_signal, label = get_one_batch(batch_size)