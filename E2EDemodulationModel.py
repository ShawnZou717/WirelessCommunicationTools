# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:14 2021

@author: ShawnZou
"""
import tensorflow as tf

def weight_variable(shape):
    initial_state = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial_state)

def bias_variable(shape):
    initial_state = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_state)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2v2(x):
    return tf.nn.max_pool(x, ksize=[1,2,1,1], padding='SAME')


class netblock:
    def __init__(self):
        self.next_block = None
        
        self.sampling_method = None
        self.nonlinear_function = None
        self.weight = None
        self.bias = None
    
    def set_conv_variable(self, shape):
        self.weight = weight_variable(shape)
        self.bias = bias_variable(shape[3:4])
    
    def set_nonlinaer_function(self, func):
        self.nonlinear_function = func
    
    def set_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method
    
    def run(self, x):
        h = self.nonlinear_function(conv2d(x, self.weight) + self.bias)
        
        if self.sampling_method is not None:
            sampled_h = self.sampleing_method(h)
        else:
            sampled_h = h
        
        if self.next_block is not None:
            self.next_block.run(sampled_h)


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
        self.head_block.run(x)
    

def main():
    block1 = netblock()
    block1.set_conv_variable([3,1,1,32])
    block1.set_nonlinaer_function(tf.nn.leaky_relu)
    block1.set_sampling_method(max_pool_2v2)
    
    

if __name__ == "__main__":
    main()







































        