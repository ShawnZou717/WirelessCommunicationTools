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




"""
reference: Graves, Alex. "Generating sequences with recurrent neural networks." arXiv preprint arXiv:1308.0850 (2013).
state functions are:
    it = sigma_i(W_xi * X_t + W_hi * H_t_1 + W_ci * C_t_1 + b_i)
    ft = sigma_f(W_xf * X_t + W_hf * H_t_1 + W_cf * C_t_1 + b_f)
    yt = sigma_y(W_xy * X_t + W_hy * H_t_1 + W_cy * C_t + b_y)
    ct = ft * c_t_1 + it * tanh(W_xc * X_t + W_hc * H_t_1 + b_c)
    ht = yt * tanh(ct)

"""
class updated_LSTM_cell_1():
    def __init__(self):
        super(updated_LSTM_cell_1, self).__init__()
        self.x_size = None
        self.h_size = None
        self.c_size = None
        self.w_size = None

        self.it = None
        self.ft = None
        self.yt = None
        self.ct0 = None
        self.ht0 = None
        self._hidden_state_set_flag = False
        #self.ct = None
        #self.ht = None

        self._para_set_flag = False

        self.W_xhci = None
        self.b_i = None

        self.W_xhcf = None
        self.b_f = None

        self.W_xhcy = None
        self.b_y = None

        self.W_xhc = None
        self.b_c = None

    def set_initial_hidden_cell_state(self, ct, ht):
        self.ht0 = ht
        self.ct0 = ct
        self._hidden_state_set_flag = True

    def set_input_size(self, x):
        if not isinstance(x, int):
            _log.error(1, "You should input a int value bigger than 0")
            raise Exception("You should input a int value bigger than 0")
        self.x_size = x

    # there are num_units, cell_size or hidden_size, they all mean the hidden size
    # which equal to the size of output.
    def set_output_size(self, x):
        if not isinstance(x, int):
            _log.error(1, "You should input a int value bigger than 0")
            raise Exception("You should input a int value bigger than 0")
        self.h_size = x
        self.c_size = x
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

    def _initialize_state(self, xt):
        if not self._hidden_state_set_flag:
            batch_size = tf.shape(xt)[0]
            self.ct0 = tf.Variable(tf.zeros([batch_size, self.c_size]), trainable=False)
            self.ht0 = tf.Variable(tf.zeros([batch_size, self.c_size]), trainable=False)

    def _initialize_para(self):
        if self._check_para():
            if not self._para_set_flag:
                self._set_para_i([self.x_size+self.h_size+self.c_size, self.w_size])
                self._set_para_f([self.x_size+self.h_size+self.c_size, self.w_size])
                self._set_para_y([self.x_size+self.h_size+self.c_size, self.w_size])
                self._set_para_c([self.x_size+self.h_size, self.w_size])
                self._para_set_flag = True
        else:
            raise Exception("Some size is not set. Not able to run cell.")

    def initialize_hidden_state(self):
        self.ct = None
        self.ht = None

    def run_at_t(self, xt, ct_1, ht_1):
        self._initialize_para()
        inp1 = tf.concat([ht_1, ct_1, xt], axis = 1)
        inp2 = tf.concat([ht_1, xt], axis = 1)

        # it = sigma_i(W_xi * X_t + W_hi * H_t_1 + W_ci * C_t_1 + b_i)
        it = tf.keras.activations.sigmoid(tf.linalg.matmul(inp1, self.W_xhci) + self.b_i)
        # ft = sigma_f(W_xf * X_t + W_hf * H_t_1 + W_cf * C_t_1 + b_f)
        ft = tf.keras.activations.sigmoid(tf.linalg.matmul(inp1, self.W_xhcf) + self.b_f)

        # ct = ft * c_t_1 + it * tanh(W_xc * X_t + W_hc * H_t_1 + b_c)
        ct = ft * ct_1 + it * tf.keras.activations.tanh(tf.linalg.matmul(inp2, self.W_xhc) + self.b_c)

        inp3 = tf.concat([ht_1, ct, xt], axis = 1)
        # yt = sigma_y(W_xy * X_t + W_hy * H_t_1 + W_cy * C_t + b_y)
        yt = tf.keras.activations.sigmoid(tf.linalg.matmul(inp3, self.W_xhcy) + self.b_y)

        # ht = yt * tanh(C_t)
        ht = yt * tf.keras.activations.tanh(ct)

        return ct, ht, yt

    def run(self, xt_list, time_step, keep_hidden_state = False):
        self._initialize_state(xt_list)
        ct, ht, yt = self.run_at_t(xt_list[:, 0, :], self.ct0, self.ht0)
        batch_size = tf.shape(ht)[0]
        hidden_size = self.h_size

        if keep_hidden_state:
            ht_all = tf.reshape(ht, [batch_size, 1, hidden_size])
            ct_all = tf.reshape(ct, [batch_size, 1, hidden_size])
            yt_all = tf.reshape(yt, [batch_size, 1, hidden_size])

        for t in range(1, time_step, 1):
            ct, ht, yt = self.run_at_t(xt_list[:, t, :], ct, ht)

            if keep_hidden_state:
                ht_ = tf.reshape(ht, [batch_size, 1, hidden_size])
                ct_ = tf.reshape(ct, [batch_size, 1, hidden_size])
                yt_ = tf.reshape(yt, [batch_size, 1, hidden_size])

                ht_all = tf.concat([ht_all, ht_], axis = 1)
                ct_all = tf.concat([ct_all, ct_], axis = 1)
                yt_all = tf.concat([yt_all, yt_], axis = 1)

        if keep_hidden_state:
            return ht_all, ct_all, yt_all
        else:
            return ct, ht, yt



class LSTM_layer:
    def __init__(self, num_cells = 1, cell_type = updated_LSTM_cell_1):
        self.lstm_cells = [cell_type() for i in range(num_cells)]

    def set_size(self, input_size, output_size):
        self.lstm_cells[0].set_input_size(input_size)
        self.lstm_cells[0].set_output_size(output_size)

        for lstm_cell in self.lstm_cells[1:]:
            lstm_cell.set_input_size(output_size)
            lstm_cell.set_output_size(output_size)
    
    def get_cell_num(self):
        return len(self.lstm_cells)

    ## it should be noticed here that the default run method is specifically constructed for updated_LSTM_cell_1
    #def run(self, xt):
    #    _, ht0, yt = self._lstm_cells[0].run(xt)
        
    #    for lstm_cell in self._lstm_cells[1:]:
    #        _, ht, yt = lstm_cell.run(yt)

    #    return ht0, ht



def test_func():
    pass

def test_func1():
    lstm_cell = LSTM_layer(num_cells = 2)
    lstm_cell.set_size(input_size = 3, output_size = 2)

    xt = tf.constant(1., shape=[5, 2, 3])
    ct1, ht1, yt1 = lstm_cell.lstm_cells[0].run(xt, 2, True)
    pass

if __name__ == "__main__":
    test_func1()