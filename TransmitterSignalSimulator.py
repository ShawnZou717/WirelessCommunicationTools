# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:50:47 2021

@author: ShawnZou
"""
import random
import math
import matplotlib
   
# 定义系统全局变量
symbol_num_per_sequence = 1024
sequence_num = 100
symbol_rate = 2500000
carrier_wave_frequency = symbol_rate
sample_rate = 10000000
samples_per_cycle = int(sample_rate / carrier_wave_frequency)
# 至少需要10个点才能把一个符号周期内的rrcf滤波器的曲线表达完整
sample_per_cycle_expanded_for_conv = samples_per_cycle * (int(10 / samples_per_cycle) + 1)
sample_num_per_sequence = samples_per_cycle * symbol_num_per_sequence
bpsk_modulation = "BPSK"
qpsk_modulation = "QPSK"



# 定义调制方式
modulation_mode = qpsk_modulation


################ start 定义辅助函数 ################
def strcmp(str1, str2):
    return str1 == str2

################ end 定义辅助函数 ################


# 计算比特率，载波频率等于比特率
if strcmp(bpsk_modulation, modulation_mode):
    bit_num_per_symbol = 1
elif strcmp(qpsk_modulation, modulation_mode):
    bit_num_per_symbol = 2
else:
    print("Warning! No modulation type specified. BPSK will be assigned.")
    bit_num_per_symbol = 1


# 产生随机基带信号
def baseband_signal_generator():
    bit_num = symbol_num_per_sequence * bit_num_per_symbol
    baseband_signal = list()
    for i in range(sequence_num):
        item = [random.randint(0, 1) for i in range(0, bit_num, 1)]
        baseband_signal.append(item)
    
    return baseband_signal


# 调制器， 将基带信号调制到载波
def digital_modulator(baseband_signal):
    if strcmp(bpsk_modulation, modulation_mode):
        modulated_signal = BPSK_modulator(baseband_signal)
    elif strcmp(qpsk_modulation, modulation_mode):
        modulated_signal = QPSK_modulator(baseband_signal)
    else:
        print("Warning! No modulation type specified. BPSK will be assigned.")
        modulated_signal = BPSK_modulator(baseband_signal)
    
    return modulated_signal


# IQ路采样预定义
I_phase0 = [-math.cos(i/sample_per_cycle_expanded_for_conv*2*math.pi) for i in range(0, sample_per_cycle_expanded_for_conv, 1)]
I_phase1 = [math.cos(i/sample_per_cycle_expanded_for_conv*2*math.pi) for i in range(0, sample_per_cycle_expanded_for_conv, 1)]

Q_phase0 = [-math.sin(i/sample_per_cycle_expanded_for_conv*2*math.pi) for i in range(0, sample_per_cycle_expanded_for_conv, 1)]
Q_phase1 = [math.sin(i/sample_per_cycle_expanded_for_conv*2*math.pi) for i in range(0, sample_per_cycle_expanded_for_conv, 1)]


# BPSK调制器
def BPSK_modulator(baseband_signal):
    modulated_signal = list()
    for baseband_signal_per_sequence in baseband_signal:
        modulated_signal_per_sequence = list()
        
        for bit in baseband_signal_per_sequence:
            if bit == 0:
                [modulated_signal_per_sequence.append(phase) for phase in I_phase0]
            elif bit == 1:
                [modulated_signal_per_sequence.append(phase) for phase in I_phase1]
                
        modulated_signal.append(modulated_signal_per_sequence)
    
    return modulated_signal


# QPSK调制器
def QPSK_modulator(baseband_signal):
    modulated_signal = list()
    for baseband_signal_per_sequence in baseband_signal:
        modulated_signal_per_sequence = list()
        
        for index in range(0, len(baseband_signal_per_sequence), 2):
            bit = str(baseband_signal_per_sequence[index]) + str(baseband_signal_per_sequence[index+1])
            if bit == "00":
                [modulated_signal_per_sequence.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase0, Q_phase0)]
            elif bit == "01":
                [modulated_signal_per_sequence.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase0, Q_phase1)]
            elif bit == "11":
                [modulated_signal_per_sequence.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase1, Q_phase1)]
            elif bit == "10":
                [modulated_signal_per_sequence.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase1, Q_phase0)]
                
        modulated_signal.append(modulated_signal_per_sequence)
        
    return modulated_signal

# 根升余弦滤波器。
# 输入为符号个数symbol_num、每个符号周期内采样个数sample_per_cycle_expanded_for_conv.
# 输出系数C = 1/sqrt(Tc)与滤波器波形
def root_raied_cosine_filter(symbol_num, sample_per_cycle_expanded_for_conv):
    ft = list()
    for i in range(symbol_num*sample_per_cycle_expanded_for_conv):
        t = i/sample_per_cycle_expanded_for_conv
        if t == 0:
            ft = 
    

# 整形滤波器，使用根余弦滤波器. 整形滤波器与信号卷积形成发送端信号
def pulse_shaping(modulated_signal):
    rrcf = root_raied_cosine_filter(sample_per_cycle_expanded_for_conv * symbol_num_per_sequence)


def plot(baseband_signal, modulated_signal):
    # show modulated signal and baseband signal
    fig = matplotlib.pyplot.figure(num = 1, figsize = (4, 6))
    gs = matplotlib.gridspec.GridSpec(3, 2)
    
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    symbol_num_to_show = 5
    sample_num_to_show = symbol_num_to_show * sample_per_cycle_expanded_for_conv
    ax1.plot(modulated_signal[0:sample_num_to_show], "b-.o")
    ax1.set_ylabel("Modulated Signal")
    
    ax2 = fig.add_subplot(gs[2, 0:2])
    ax2.plot(baseband_signal[0:int(sample_num_to_show/sample_per_cycle_expanded_for_conv*bit_num_per_symbol)],\
             "bd")
    ax2.set_ylabel("Baseband Signal")

    matplotlib.pyplot.show()


# 系统接口
def main():
    baseband_signal = baseband_signal_generator()
    modulated_signal = digital_modulator(baseband_signal)
    
    # 绘制调制信号与基带信号中的部分码元
    plot(baseband_signal[0], modulated_signal[0])
    
    # 整形滤波，采用根升余弦滤波器
    # shaped_pulse = pulse_shaping(modulated_signal)


if __name__ == "__main__":
    main()

















