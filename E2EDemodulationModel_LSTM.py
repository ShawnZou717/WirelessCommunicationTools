# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
from logging import raiseExceptions
import tensorflow as tf
import random as biubiubiu
import matplotlib.pyplot as plt
import struct
import time
import os

import ExceptionDealingModule


_code_dict = {1:"Illegal input. ",\
              100:"undefined error. "}
_log = ExceptionDealingModule.log("E2EDemodulationModel_LSTM", _code_dict)

def weight_variable(shape):
    initial_state = tf.random.uniform(shape, minval= -1, maxval = 1)
    return tf.Variable(initial_state)


def bias_variable(shape):
    initial_state = tf.random.uniform(shape, minval = -1, maxval = 1)
    return tf.Variable(initial_state)



class classic_LSTM_cell:
    def __init__(self):
        self.forget_gate = None
        self.forgate_gate_w = None
        self.forgate_gate_b = None

        self.update_gate = None
        self.update_gate_w_sigmoid = None
        self.update_gate_b_sigmoid = None
        self.update_gate_w_tanh = None
        self.update_gate_b_tanh = None

        self.output_gate = None
        self.output_gate_w_sigmoid = None
        self.output_gate_b_sigmoid = None
        self.output_gate_w_tanh = None
        self.output_gate_b_tanh = None

    def set_forget_gate(self, shape):
        if not isinstane(shape, list):
            _log.error(1, "You should input a list with 2 elements")
            raise Exceptions("You should input a list with 2 elements")

        self.forgate_gate_w = weight_variable(shape)
        self.forgate_gate_b = bias_variable(shape[1])

    def set_update_gate(self, shape):
        if not isinstane(shape, list):
            _log.error(1, "You should input a list with 2 elements")
            raise Exceptions("You should input a list with 2 elements")

        self.update_gate_w_sigmoid = weight_variable(shape)
        self.update_gate_w_tanh = weight_variable(shape)

    def set_output_gate(self, shape):
        if not isinstane(shape, list):
            _log.error(1, "You should input a list with 2 elements")
            raise Exceptions("You should input a list with 2 elements")

        self.output_gate_w_sigmoid = weight_variable(shape)
        self.output_gate_w_tanh = weight_variable(shape)

    def run(self, ct_1, ht_1, xt):
        inp = tf.concat([ht_1, xt], axis = 1)
        
        self.forget_gate = tf.keras.activations.sigmoid(tf.linalg.matmul(inp, self.forgate_gate_w) + self.forgate_gate_b)

        ut = tf.keras.activations.sigmoid(tf.linalg.matmul(inp, self.update_gate_w_sigmoid) + self.update_gate_b_sigmoid)
        uut = tf.keras.activations.tanh(tf.linalg.matmul(inp, self.update_gate_w_tanh) + self.update_gate_b_tanh)
        self.update_gate = self.forget_gate + (ut * uut)

        ot = tf.keras.activations.sigmoid(tf.linalg.matmul(inp, self.output_gate_w_sigmoid) + self.output_gate_b_sigmoid)
        ott = tf.keras.activations.tanh(tf.linalg.matmul(self.update_gate, self.output_gate_w_tanh) + self.output_gate_b_tanh)
        self.output_gate = ot * ott

        return self.update_gate, self.output_gate



"""
reference: Graves, Alex. "Generating sequences with recurrent neural networks." arXiv preprint arXiv:1308.0850 (2013).
state functions are:
    it = sigma_i(W_xi * X_t + W_hi * H_t_1 + W_ci * C_t_1 + b_i)
    ft = sigma_f(W_xf * X_t + W_hf * H_t_1 + W_cf * C_t_1 + b_f)
    yt = sigma_y(W_xy * X_t + W_hy * H_t_1 + W_cy * C_t + b_y)
    ct = ft * c_t_1 + it * tanh(W_xc * X_t + W_hc * H_t_1 + b_c)
    ht = yt * tanh(ct)

"""
class updated_LSTM_cell_1:
    def __init__(self):
        self.it = None
        self.ft = None
        self.yt = None
        self.ct = None
        self.ht = None

        self.W_xi = None
        self.W_hi = None
        self.W_ci = None
        self.b_i = None

        self.W_xf = None
        self.W_hf = None
        self.W_cf = None
        self.b_f = None

        self.W_xy = None
        self.W_hy = None
        self.W_cy = None
        self.b_y = None

        self.W_xc = None
        self.W_hc = None
        self.b_c = None

    def set_forget_gate(self, shape):
        pass

    def set_update_gate(self, shape):
        pass

    def set_output_gate(self, shape):
        pass

    def run(self, ct_1, ht_1, xt):
        pass

