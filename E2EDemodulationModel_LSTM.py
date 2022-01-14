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


class lstm_model:
    def __init__(self):
        self.lstm_cell = None
        self.cell_size = None

    def set_lstm_cell(self, x = lstm.classic_LSTM_cell()):
        self.lstm_cell = x

    def set_cell_size(self, size = 1):
        self.cell_size = size