# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import tensorflow as tf

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
        if not isinstance(shape, list):
            _log.error(1, "You should input a list with 2 elements")
            raise Exception("You should input a list with 2 elements")
        
        self.forgate_gate_w = weight_variable(shape)
        self.forgate_gate_b = bias_variable([shape[1]])

    def set_update_gate(self, shape):
        if not isinstance(shape, list):
            _log.error(1, "You should input a list with 2 elements")
            raise Exception("You should input a list with 2 elements")

        self.update_gate_w_sigmoid = weight_variable(shape)
        self.update_gate_w_tanh = weight_variable(shape)

    def set_output_gate(self, shape):
        if not isinstance(shape, list):
            _log.error(1, "You should input a list with 2 elements")
            raise Exception("You should input a list with 2 elements")

        self.output_gate_w_sigmoid = weight_variable(shape)
        self.output_gate_w_tanh = weight_variable(shape)

    def run(self, ct_1, ht_1, xt):
        inp = tf.concat([ht_1, xt], axis = 1)
        
        self.forget_gate = tf.keras.activations.sigmoid(tf.linalg.matmul(inp, self.forgate_gate_w) + self.forgate_gate_b) * ct_1

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
        self.x_size = None
        self.h_size = None
        self.c_size = None
        self.w_size = None

        self.it = None
        self.ft = None
        self.yt = None
        self.ct = None
        self.ht = None

        self.W_xhci = None
        self.b_i = None

        self.W_xhcf = None
        self.b_f = None

        self.W_xhcy = None
        self.b_y = None

        self.W_xhc = None
        self.b_c = None

    def set_x_size(self, x):
        if not isinstance(x, int):
            _log.error(1, "You should input a int value bigger than 0")
            raise Exception("You should input a int value bigger than 0")
        self.x_size = x

    def set_h_size(self, x):
        if not isinstance(x, int):
            _log.error(1, "You should input a int value bigger than 0")
            raise Exception("You should input a int value bigger than 0")
        self.h_size = x

    def set_c_size(self, x):
        if not isinstance(x, int):
            _log.error(1, "You should input a int value bigger than 0")
            raise Exception("You should input a int value bigger than 0")
        self.c_size = x

    def set_w_size(self, x):
        if not isinstance(x, int):
            _log.error(1, "You should input a int value bigger than 0")
            raise Exception("You should input a int value bigger than 0")
        self.w_size = x

    def _set_para_i(self, shape):
        self.W_xhci = weight_variable(shape)
        self.b_i = bias_variable([shape[1]])
        
    def _set_para_f(self, shape):
        self.W_xhcf = weight_variable(shape)
        self.b_f = bias_variable([shape[1]])

    def _set_para_y(self, shape):
        self.W_xhcy = weight_variable(shape)
        self.b_y = bias_variable([shape[1]])

    def _set_para_c(self, shape):
        self.W_xhc = weight_variable(shape)
        self.b_c = bias_variable([shape[1]])

    def _check_para(self):
        return (self.x_size is not None) and\
                (self.h_size is not None) and\
                (self.c_size is not None) and\
                (self.w_size is not None)

    def _initialize_cell(self):
        if self._check_para():
            self._set_para_i([self.x_size+self.h_size+self.c_size, self.w_size])
            self._set_para_f([self.x_size+self.h_size+self.c_size, self.w_size])
            self._set_para_y([self.x_size+self.h_size+self.c_size, self.w_size])
            self._set_para_c([self.x_size+self.h_size, self.w_size])
        else:
            raise Exception("Some size is not set. Not able to run cell.")

    def run(self, ct_1, ht_1, xt):
        self._initialize_cell()

        inp1 = tf.concat([ht_1, ct_1, xt], axis = 1)
        inp2 = tf.concat([ht_1, xt], axis = 1)

        # it = sigma_i(W_xi * X_t + W_hi * H_t_1 + W_ci * C_t_1 + b_i)
        self.it = tf.keras.activations.sigmoid(tf.linalg.matmul(inp1, self.W_xhci) + self.b_i)
        # ft = sigma_f(W_xf * X_t + W_hf * H_t_1 + W_cf * C_t_1 + b_f)
        self.ft = tf.keras.activations.sigmoid(tf.linalg.matmul(inp1, self.W_xhcf) + self.b_f)

        # ct = ft * c_t_1 + it * tanh(W_xc * X_t + W_hc * H_t_1 + b_c)
        self.ct = self.ft * ct_1 + self.it * tf.keras.activations.tanh(tf.linalg.matmul(inp2, self.W_xhc) + self.b_c)

        inp3 = tf.concat([ht_1, self.ct, xt], axis = 1)
        # yt = sigma_y(W_xy * X_t + W_hy * H_t_1 + W_cy * C_t + b_y)
        self.yt = tf.keras.activations.sigmoid(tf.linalg.matmul(inp3, self.W_xhcy) + self.b_y)

        # ht = yt * tanh(ct)
        self.ht = self.yt * tf.keras.activations.tanh(self.ct)

        return self.ct, self.ht, self.yt

    def run_as_first_cell(self, xt):
        batch_size = xt.get_shape().as_list()[0]
        ct_1 = tf.constant(0., shape = [batch_size, self.c_size])
        ht_1 = tf.constant(0., shape = [batch_size, self.h_size])
        
        ct, ht, yt = self.run(ct_1, ht_1, xt)
        return ct, ht, yt


def test_func():
    lstm_cell = updated_LSTM_cell_1()
    lstm_cell.set_x_size(512)
    lstm_cell.set_h_size(128)
    lstm_cell.set_c_size(32)
    lstm_cell.set_w_size(64)

    xt = tf.constant(1., shape=[10, 512])
    ct, ht, yt = lstm_cell.run_as_first_cell(xt)
    pass

if __name__ == "__main__":
    test_func()