# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:50:47 2021

@author: ShawnZou
"""
import random
import math
import matplotlib.pyplot as plt
import numpy
import struct
import time
import sys
   

# 定义系统全局变量
symbol_num_per_sequence = 1024
symbol_rate = 2500000
carrier_wave_frequency = symbol_rate
sample_rate = 10000000
samples_per_cycle = int(sample_rate / carrier_wave_frequency)

# 卷积太长，太耗费时间了，把卷积间隔变短一点
# 卷积参数可以保留在这里，采样率比较低的时候还可以用这个参数来拓展卷积
conv_dot_per_sample = 1
sample_per_cycle_expanded_for_conv = samples_per_cycle * conv_dot_per_sample
sample_num_per_sequence = samples_per_cycle * symbol_num_per_sequence
bpsk_modulation = "BPSK"
qpsk_modulation = "QPSK"


# 将输入参数设置为全局变量（需要改变指的全局变量初始值必须设置成None，不然容易出错）
modulation_mode = None
bit_num_per_symbol = None
signal_noise_ratio = None
rolloff_factor = None
sequence_num = 100


################ start 定义辅助函数 ################
def strcmp(str1, str2):
    return str1 == str2

################ end 定义辅助函数 ################


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
# 输出系数C = 1/sqrt(Tc)、滤波器波形rrcf、波形横坐标time_seq(time_seq表示Tc的倍数)
def root_raied_cosine_filter(symbol_num, sample_per_cycle_expanded_for_conv, roll_ratio):
    rrcf = list()
    time_seq = list()
    
    Tc = 1/symbol_rate
    C = 1/math.sqrt(Tc)
    
    filter_len = 2 * symbol_num * sample_per_cycle_expanded_for_conv
    for i in range(filter_len):
        t = i/sample_per_cycle_expanded_for_conv - symbol_num
        time_seq.append(t)
        if t == 0:
            rrcf_elem = 1 - roll_ratio + 4 * roll_ratio / math.pi
        elif abs(abs(t) - 1/4/roll_ratio) <= 10**(-6):
            rrcf_elem = roll_ratio/math.sqrt(2) * \
                ((1+2/math.pi)*math.sin(math.pi/4/roll_ratio) + (1-2/math.pi)*math.cos(math.pi/4/roll_ratio))
        else:
            rrcf_elem = (math.sin(math.pi*t*(1-roll_ratio)) + \
                4*t*roll_ratio*math.cos(math.pi*t*(1+roll_ratio))) /\
                    (math.pi*t*(1-(4*t*roll_ratio)**2))
        
        rrcf.append(rrcf_elem)
    
    dt = Tc / sample_per_cycle_expanded_for_conv
    C = C*dt
    return C, rrcf, time_seq


# 整形滤波器，使用根余弦滤波器. 整形滤波器与信号卷积形成发送端信号
def pulse_shaping(modulated_signal_list):
    C, rrcf, time_seq = root_raied_cosine_filter(symbol_num_per_sequence, sample_per_cycle_expanded_for_conv, rolloff_factor)
    
    ## 绘制整形滤波器时域波形
    #fig = plt.figure(num = 1, figsize = (4, 4))
    #ax1 = fig.add_subplot(111)
    #x = time_seq[int(len(time_seq)/2)-100:int(len(time_seq)/2)+100]
    #y = rrcf[int(len(time_seq)/2)-100:int(len(time_seq)/2)+100]
    #ax1.plot(x, y, "b-o")
    #ax1.set_title("Impulse response of RRCF")
    #ax1.set_ylabel("Impulse Response")
    #ax1.set_xlabel("t(Tc)")
    #plt.show()
    shaped_pulse_list = list()

    for modulated_signal in modulated_signal_list:
    # 信号的时间轴与滤波器的时间轴要相互对应。这里信号只有正时间轴，而rrcf有全时间轴。
        modulated_signal_expanded = [0 for i in range(len(rrcf)-len(modulated_signal))]
        modulated_signal_expanded += modulated_signal
        shaped_pulse_full = numpy.convolve(modulated_signal_expanded, rrcf, 'full')
        shaped_pulse_conv = shaped_pulse_full[int(len(shaped_pulse_full)/2):]

    # 卷积过后的时间轴取与信号对应的时间轴信息
        shaped_pulse = [shaped_pulse_conv[i] for i in range(0, int(len(shaped_pulse_conv)/2), conv_dot_per_sample)]
        shaped_pulse_list.append(shaped_pulse)
    # fig = plt.figure(num = 1, figsize = (4, 4))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(shaped_pulse[0+100:100+4*20], "b-.o")
    # ax1.set_title("Shaped pulse")
    # plt.show()
    
    return C, shaped_pulse_list

def plot(baseband_signal, modulated_signal):
    # show modulated signal and baseband signal
    fig = plt.figure(num = 1, figsize = (4, 6))
    gs = plt.gridspec.GridSpec(3, 2)
    
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    symbol_num_to_show = 5
    sample_num_to_show = symbol_num_to_show * sample_per_cycle_expanded_for_conv
    ax1.plot(modulated_signal[0:sample_num_to_show], "b-.o")
    ax1.set_ylabel("Modulated Signal")
    
    ax2 = fig.add_subplot(gs[2, 0:2])
    ax2.plot(baseband_signal[0:int(sample_num_to_show/sample_per_cycle_expanded_for_conv*bit_num_per_symbol)],\
             "bd")
    ax2.set_ylabel("Baseband Signal")

    plt.show()


# 添加高斯白噪声
def add_noise(x, snr):
    xpower = sum([item**2 for item in x])
    npower = xpower / (10**(snr/10)) / len(x)
    
    noise = numpy.random.randn(len(x)) * math.sqrt(npower)
    
    return list(x + noise)


def labelize(baseband_signal):
    true_label = list()
    if strcmp(bpsk_modulation, modulation_mode):
        for baseband_item in baseband_signal:
            true_label_one_bar = list()
            for baseband_dot in baseband_item:
                if baseband_dot == 0:
                    true_label_one_bar.append([b'1', b'0', b'0', b'0'])
                elif baseband_dot == 1:
                    true_label_one_bar.append([b'0', b'1', b'0', b'0'])
            true_label.append(true_label_one_bar)
    elif strcmp(qpsk_modulation, modulation_mode):
        for baseband_item in baseband_signal:
            true_label_one_bar = list()
            for i in range(0, len(baseband_item), 2):
                bit1 = baseband_item[i]
                bit2 = baseband_item[i+1]
                bit = str(bit1) + str(bit2)

                if bit == '00':
                    true_label_one_bar.append([b'1', b'0', b'0', b'0'])
                elif bit == '01':
                    true_label_one_bar.append([b'0', b'1', b'0', b'0'])
                elif bit == '10':
                    true_label_one_bar.append([b'0', b'0', b'1', b'0'])
                elif bit == '11':
                    true_label_one_bar.append([b'0', b'0', b'0', b'1'])
            true_label.append(true_label_one_bar)

    return true_label
        
            

# 系统接口
def main():
    baseband_signal = baseband_signal_generator()
    print("%s baseband_signal_generator finished" % (time.time()))
    modulated_signal = digital_modulator(baseband_signal)
    print("%s digital_modulator finished" % (time.time()))
    
    # # 绘制调制信号与基带信号中的部分码元
    # plot(baseband_signal[0], modulated_signal[0])
    
    # 整形滤波，采用根升余弦滤波器, C为输出信号的系数
    # 实际上信号应该等于信号波形乘以系数C，但是因为计算机处理浮点型数据精度有限，所以我们可以忽略常数C。
    C, shaped_pulse = pulse_shaping(modulated_signal)
    print("%s pulse_shaping finished" % (time.time()))
    
    # 加性高斯白噪声
    final_signal_list = list()
    for pulse in shaped_pulse:
        final_signal = add_noise(pulse, signal_noise_ratio)
        final_signal_list.append(final_signal)
    print("%s add_noise finished" % (time.time()))
    
    # 编码为二进制数据并存储
    final_signal_list_br = [[struct.pack('d', signal_dot) for signal_dot in final_signal] for final_signal in final_signal_list]
    true_label = labelize(baseband_signal)

    save_path = "D:\\[0]MyFiles\\FilesCache\\DataSet\\%s" % \
        (modulation_mode + "_" + str(sequence_num) + "bars_" + str(signal_noise_ratio) + "dB_r" + str(rolloff_factor) + ".dat")
    with open(save_path, 'wb') as f:
        for final_signal in final_signal_list_br:
            for elem in final_signal:
                f.write(elem)

        for label_bar in true_label:
            for bar_elem in label_bar:
                for byte in bar_elem:
                    f.write(byte)


if __name__ == "__main__":
    a1 = ["BPSK"]
    a2 = [0.1*(i+1) for i in range(5)]
    a3 = list(range(-2, 8, 1))
    for h in a1:
        modulation_mode = h

        # 计算比特率，载波频率等于比特率
        if strcmp(bpsk_modulation, modulation_mode):
            bit_num_per_symbol = 1
        elif strcmp(qpsk_modulation, modulation_mode):
            bit_num_per_symbol = 2
        else:
            print("Warning! No modulation type specified. BPSK will be assigned.")
            bit_num_per_symbol = 1

        for hh in a2:
            rolloff_factor = hh
            for hhh in a3:
                signal_noise_ratio = hhh
                start_t = time.time()
                print("Started %s modulated with rolloff factor = %s and snr = %d, at %d seconds." % (modulation_mode, rolloff_factor, signal_noise_ratio, start_t))
                main()
                end_t = time.time()
                print("Started %s modulated with rolloff factor = %s and snr = %d, totally cost %d seconds." % (modulation_mode, rolloff_factor, signal_noise_ratio, end_t - start_t))

                

















