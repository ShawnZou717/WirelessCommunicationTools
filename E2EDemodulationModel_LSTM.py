# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import LSTM as lstm

import ExceptionDealingModule


_code_dict = {1:"Illegal input. ",\
              100:"undefined error. "}
_log = ExceptionDealingModule.log("E2EDemodulationModel_LSTM", _code_dict)


def create_layers(n = 1, x_size = 128, h_size = 64, c_size = 64, w_size = 64):
    lstm_cell_list = list()
    for i in range(n):
        lstm_cell = lstm.updated_LSTM_cell_1()
        lstm_cell.set_x_size(x_size)
        lstm_cell.set_h_size(h_size)
        lstm_cell.set_c_size(c_size)
        lstm_cell.set_w_size(w_size)
        lstm_cell_list.append(lstm_cell)
    return lstm_cell_list


def run_encoder(lstm_cell_layers, xt_list):
    lstm1 = lstm_cell_layers[0][0]
    lstm2 = lstm_cell_layers[1][0]
    ct1, ht1, yt1 = lstm1.run_as_first_cell(xt_list[0])
    ct2, ht2, yt2 = lstm2.run_as_first_cell(yt1)

    for lstm1, lstm2, xt in zip(lstm_cell_layers[0][1:], lstm_cell_layers[1][1:], xt_list[1:]):
        ct1, ht1, yt1 = lstm1.run(ct_1 =ct1 , ht_1 = ht1, xt)
        ct2, ht2, yt2 = lstm1.run(ct_1 =ct2 , ht_1 = ht2, yt1)

    return ht1, ht2




if __name__ == "__main__":
    e_layer1 = create_layers(n = 128, x_size = 128, h_size = 64, c_size = 64, w_size = 64)
    e_layer2 = create_layers(n = 128, x_size = 64, h_size = 64, c_size = 64, w_size = 64)

    d_layer1 = create_layers(n = 128, x_size = 128, h_size = 64, c_size = 64, w_size = 64)
    d_layer2 = create_layers(n = 128, x_size = 64, h_size = 64, c_size = 64, w_size = 64)
    
    lstm_cell_layers = [layer1, layer2]
    ht1, ht2 = run_encoder(lstm_cell_layers, xt_list)