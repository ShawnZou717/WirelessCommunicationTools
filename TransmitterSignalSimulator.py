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
symbol_rate = 2500000
carrier_wave_frequency = symbol_rate
sample_rate = 10000000
bpsk_modulation = "BPSK"
qpsk_modulation = "QPSK"
samples_per_cycle = int(sample_rate / carrier_wave_frequency)


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
    baseband_signal = [random.randint(0, 1) for i in range(0, bit_num, 1)]
    
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


# BPSK调制器
def BPSK_modulator(baseband_signal):
    I_phase0 = [-math.cos(i/samples_per_cycle*2*math.pi) for i in range(0, samples_per_cycle, 1)]
    I_phase1 = [math.cos(i/samples_per_cycle*2*math.pi) for i in range(0, samples_per_cycle, 1)]
    
    modulated_signal = list()
    for bit in baseband_signal:
        if bit == 0:
            [modulated_signal.append(phase) for phase in I_phase0]
        elif bit == 1:
            [modulated_signal.append(phase) for phase in I_phase1]
    
    return modulated_signal


# QPSK调制器
def QPSK_modulator(baseband_signal):
    
    I_phase0 = [-math.cos(i/samples_per_cycle*2*math.pi) for i in range(0, samples_per_cycle, 1)]
    I_phase1 = [math.cos(i/samples_per_cycle*2*math.pi) for i in range(0, samples_per_cycle, 1)]
    
    Q_phase0 = [-math.sin(i/samples_per_cycle*2*math.pi) for i in range(0, samples_per_cycle, 1)]
    Q_phase1 = [math.sin(i/samples_per_cycle*2*math.pi) for i in range(0, samples_per_cycle, 1)]
    
    modulated_signal = list()
    for index in range(0, len(baseband_signal), 2):
        bit = str(baseband_signal[index]) + str(baseband_signal[index+1])
        if bit == "00":
            [modulated_signal.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase0, Q_phase0)]
        elif bit == "01":
            [modulated_signal.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase0, Q_phase1)]
        elif bit == "11":
            [modulated_signal.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase1, Q_phase1)]
        elif bit == "10":
            [modulated_signal.append(I_phase+Q_phase) for I_phase, Q_phase in zip(I_phase1, Q_phase0)]
        
    return modulated_signal


# 整形滤波器，使用根余弦滤波器. 整形滤波器与信号卷积形成发送端信号
def pulse_shaping(modulated_signal):
    


def plot(baseband_signal, modulated_signal):
    # show modulated signal and baseband signal
    fig = matplotlib.pyplot.figure(num = 1, figsize = (4, 6))
    gs = matplotlib.gridspec.GridSpec(3, 2)
    
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    symbol_num_to_show = 5
    sample_num_to_show = symbol_num_to_show * samples_per_cycle
    ax1.plot(modulated_signal[0:sample_num_to_show], "b-.o")
    ax1.set_ylabel("Modulated Signal")
    
    ax2 = fig.add_subplot(gs[2, 0:2])
    ax2.plot(baseband_signal[0:int(sample_num_to_show/samples_per_cycle*bit_num_per_symbol)],\
             "bd")
    ax2.set_ylabel("Baseband Signal")

    matplotlib.pyplot.show()


# 系统接口
def main():
    baseband_signal = baseband_signal_generator()
    modulated_signal = digital_modulator(baseband_signal)
    shaped_pulse = pulse_shaping(modulated_signal)


if __name__ == "__main__":
    main()

















