# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:50:47 2021

@author: ShawnZou

@purpose:
    This module is created to make signal processing simulation more easily.

@accessable functions:
    1. baseband_generator(n = 1)
        input type:int      notion: number of bits you want to generate
        output type:list        notion: list of bits

"""

import ExceptionDealingModule
import random as biubiubiu
import math
import numpy
import matplotlib.pyplot as plt
import time


# global varaible to be used only inside.
_code_dict = {1:"Given parameter illegal. ",\
             100:"Undefined error. "}
_log = ExceptionDealingModule.log("test_module", _code_dict)


# modulation amp list. 
# structure is [symbol1, symbol2,... symboln], symboln = [I_amp_n, Q_amp_n]
_bpsk_amp = [[-1, 0],[1, 0]]
_qpsk_amp = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

_global_modulation_type = ["BPSK", "QPSK"]
_bits_per_symbol = {"BPSK":1,\
                     "QPSK":2}
_modulation_amplitude = {"BPSK":_bpsk_amp,\
                          "QPSK":_qpsk_amp}

_sine = list()
_cosine = list()


def _bit_check(bits):
    if not isinstance(bits, list):
        raise Exception("given bits should be a list")
    for bit in bits:
        if bit != 0 and bit != 1:
            return False
    return True


class bit_message:
    def __init__(self, bits, modulation_type = None):
        if _bit_check(bits):
            self.bits = list(bits)
            self.modulation_type = modulation_type
        else:
            raise Exception("given bits illegal.")

    def decode(self):
        if self.modulation_type is not None and self.modulation_type in _global_modulation_type:
            return _modulation_decode_func[self.modulation_type](self.bits)
        else:
            return None


def _bpsk_decode(bits):
    symbols = list(bits)
    return symbols

def _qpsk_decode(bits):
    symbol = list()
    for i in range(0, len(bits), 2):
        bit_bar = bits[i:i+2]
        if bit_bar[0] == 0 and bit_bar[1] == 0:
            symbol.append(0)
        elif bit_bar[0] == 0 and bit_bar[1] == 1:
            symbol.append(1)
        elif bit_bar[0] == 1 and bit_bar[1] == 0:
            symbol.append(2)
        elif bit_bar[0] == 1 and bit_bar[1] == 1:
            symbol.append(3)
    return symbol

_modulation_decode_func = {"BPSK":_bpsk_decode,\
                          "QPSK":_qpsk_decode}

def bpsk_modulate(bits):
    if not _bit_check(bits):
        return None

    modulated_signal = list()
    amp_list = _modulation_amplitude["BPSK"]

    symbols = _bpsk_decode(bits)
    for symbol in symbols:
        I_amp, Q_amp = amp_list[symbol]
        IQ_signal = I_amp * numpy.array(_sine) + Q_amp * numpy.array(_cosine)
        IQ_signal = list(IQ_signal)
        modulated_signal.extend(IQ_signal)

    return modulated_signal

def qpsk_modulate(bits):
    if not _bit_check(bits):
        return None

    if len(bits) % 2 != 0:
        _log.error(2, "Given bits is of illegal length for QPSK.")
        return None

    modulated_signal = list()
    amp_list = _modulation_amplitude["QPSK"]

    symbols = _qpsk_decode(bits)
    for symbol in symbols:
        I_amp, Q_amp = amp_list[symbol]
        IQ_signal = I_amp * numpy.array(_sine) + Q_amp * numpy.array(_cosine)
        IQ_signal = list(IQ_signal)
        modulated_signal.extend(IQ_signal)

    return modulated_signal

_modulation_func_dict = {"BPSK":bpsk_modulate,\
                          "QPSK":qpsk_modulate}

def pulse_shaping(modulated_signal, shaping_filter):
    n = len(modulated_signal)

    res = numpy.convolve(modulated_signal, shaping_filter, 'full')
    L = len(res)

    mid_index = int(L/2) - 1
    left_index = mid_index - int(n/2)
    sigma = 0
    if n % 2 != 0:
        sigma = 1
    right_index = mid_index + int(n/2) + sigma

    return list(res[left_index:right_index])

def awgn(x, snr):
    xpower = sum([item**2 for item in x])
    npower = xpower / (10**(snr/10)) / len(x)
    
    noise = numpy.random.randn(len(x)) * math.sqrt(npower)
    
    return list(x + noise)

def root_raied_cosine_filter(filter_span_in_symbols, symbol_frequency, oversampling_factor, roll_ratio):
    rrcf = list()
    time_seq = list()
    
    Tc = 1/symbol_frequency
    C = 1/math.sqrt(Tc)
    
    filter_len = 2 * oversampling_factor * filter_span_in_symbols

    for i in range(filter_len):
        t = i/(2 * oversampling_factor) - filter_span_in_symbols / 2
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
    
    dt = Tc / (2 * oversampling_factor)

    return rrcf, time_seq, C, dt



def generate_baseband_signal(n = 1):
    if not isinstance(n, int) or n < 0:
        return None

    baseband_signal = [biubiubiu.randint(0, 1) for i in range(0, n, 1)]
    return baseband_signal

# inner func, do not use outside
def _init_sine_cosine():
    global _sine
    global _cosine
    _sine = list()
    _cosine = list()

class transmitter:
    def __init__(self):
        self.set_para_to_none()

    def set_para_to_none(self):
        self._carrier_frequency = None
        self._oversampling_factor = None
        self._filter_span = None
        self._modulation_type = None
        self._sampling_frequency = None
        self._rrcf = None
        self._roll_ratio = None
        self._snr = None
        self._time_seq = None
        self._propotion = None
        self._bits = None
        self._symbols = None
        self._modulated_signal = None
        self._shaped_signal = None
        pass

    def set_carrier_frequency(self, x):
        if (not isinstance(x, int)) and (not isinstance(x, float)):
            _log.error(1)
            return
        self._carrier_frequency = x

    # sampling frequency = oversampling factor * 2 * carrier_frequency
    def set_oversamping_factor(self, x):
        if (not isinstance(x, int)) and (not isinstance(x, float)):
            _log.error(1)
            return
        self._oversampling_factor = x

    # filter span is set to of length equal x symbols
    def set_filter_span(self, x):
        if (not isinstance(x, int)) and (not isinstance(x, float)):
            _log.error(1)
            return
        self._filter_span = x

    def set_roll_ratio(self, x):
        if (not isinstance(x, int)) and (not isinstance(x, float)):
            _log.error(1)
            return
        self._roll_ratio = x

    def set_modulation_type(self, x):
        if x not in _global_modulation_type:
            _log.error(1, "Modulation shoule be one in %s"%(",".join(_global_modulation_type)))
            return
        self._modulation_type = x
        self._modulate_signal = _modulation_func_dict[self._modulation_type]

    def set_snr(self, x):
        if (not isinstance(x, int)) and (not isinstance(x, float)):
            _log.error(1)
            return
        self._snr = x

    def get_roll_factor(self):
        return self._roll_ratio

    def get_snr(self):
        return self._snr

    def get_time_seq(self):
        return list(self._time_seq)

    def get_bits(self):
        return list(self._bits)

    def get_symbols(self):
        return list(self._symbols)

    def get_propotion(self):
        return dict(self._propotion)

    def get_modulated_signal(self):
        return list(self._modulated_signal)
    
    def get_shaped_signal(self):
        return list(self._shaped_signal)

    def get_noised_signal(self):
        return list(self._noised_signal)

    def get_modulation_type(self):
        return str(self._modulation_type)

    # after set all needed parameter like carrier_frequency, oversampling_factor
    # it is time to compute rest parameters incluing pre para and randomly generate not-given para
    def init_setting(self):
        if self._oversampling_factor is None or self._carrier_frequency is None:
            _log.error(1, "oversampling_factor or carrier_frequency is not set.")
        self._sampling_frequency = self._oversampling_factor * 2 * self._carrier_frequency
        self._IQ_pregenerate()

        # if roll ratio is not given, randomly generate between [0.1, 0.5)
        if self._roll_ratio is None:
            _log.warn("roll ratio not set. generate randomly.")
            self._roll_ratio = 0.4*biubiubiu.random() + 0.1
        self._rrcf, self._time_seq, C, dt = root_raied_cosine_filter(self._filter_span, self._carrier_frequency, self._oversampling_factor, self._roll_ratio)
        self._propotion = {"C":C, "dt":dt}

        # if snr is not given, randomly generate between [-2, 8)dB
        if self._snr is None:
            _log.warn("snr not set. generate randomly.")
            self._snr = 10.0*biubiubiu.random() - 2.0

    def init(self, carrier_frequency, oversampling_factor, filter_span, modulation_type):
        self.set_para_to_none()
        self.set_carrier_frequency(carrier_frequency)
        self.set_oversamping_factor(oversampling_factor)
        self.set_filter_span(filter_span)
        self.set_modulation_type(modulation_type)
        self.init_setting()

    # pre generate sine cosine point since triangle function computation has great time cost.
    # generated during one symbol period
    def _IQ_pregenerate(self):
        _init_sine_cosine()
        d_phase = 2*math.pi/(self._oversampling_factor*2)
        for i in range(self._oversampling_factor*2):
            phase = i * d_phase
            _sine.append(math.sin(phase))
            _cosine.append(math.cos(phase))

    # generate noised signal by given number of bits
    # bits will be generate randomly
    def generate_signal_by_bit_num(self, bits_num):
        self._bits = generate_baseband_signal(bits_num)
        bit_m = bit_message(self._bits, self._modulation_type)
        self._symbols = bit_m.decode()
        
        module_signal_func = _modulation_func_dict[self._modulation_type]
        self._modulated_signal = module_signal_func(self._bits)

        self._shaped_signal = pulse_shaping(self._modulated_signal, self._rrcf)
        noised_signal = awgn(self._shaped_signal, self._snr)
        self._noised_signal = list(noised_signal)
        return noised_signal

    # generate noised signal by given number of symbols
    # symbols will be generate randomly
    def generate_signal_by_symbol_num(self, symbols_num = 1):
        bits_num = _bits_per_symbol[self._modulation_type] * symbols_num
        return self.generate_signal_by_bit_num(bits_num)

    # generate noised signal by given bit list
    def generate_signal_by_bits(self, bit_list):
        _log.error("This func is not finished yet.")
        return None

    # generate noised signal by given symbol list
    def generate_signal_by_symbols(self, symbol_list = [0]):
        _log.error("This func is not finished yet.")
        return None


def save_pic(transer, path = ".\\"):
    if not isinstance(transer, transmitter):
        raise Exception("Wrong instance inputted in save_pic")

    signal = transer.get_noised_signal()
    modulated_signal = transer.get_modulated_signal()
    shaped_signal = transer.get_shaped_signal()
    symbols = transer.get_symbols()
    bits = transer.get_bits()
    snr = transer.get_snr()
    roll_ratio = transer.get_roll_factor()

    rrcf = transer._rrcf
    time_seq = transer._time_seq
    
    fig = plt.figure(num = 1, figsize=[24, 24])
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)

    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    ax1.plot(symbols, c='k',ls='', marker='*', mec='r',mfc='w')
    ax1.set_title("symbols")

    ax2.plot(bits, c='k',ls='', marker='*', mec='r',mfc='w')
    ax2.set_title("bits")

    ax3.plot(modulated_signal, c='b',ls='-.', marker='', mec='r',mfc='w')
    ax3.set_title("modulated_signal")

    ax4.plot(shaped_signal, c='b',ls='-.', marker='', mec='r',mfc='w')
    ax4.set_title("shaped_signal")

    ax5.plot(signal, c='b',ls='-.', marker='', mec='r',mfc='w')
    ax5.set_title("signal, snr = %sdB"%(str(snr)))

    ax6.plot(time_seq, rrcf, c='b',ls='-.', marker='', mec='r',mfc='w')
    ax6.set_title("rrcf, roll ratio = %s"%(str(roll_ratio)))

    str_l = "%s_snr%s_r%s_%s.jpg"%(transer.get_modulation_type(), str(snr), \
        str(roll_ratio), time.strftime("%Y%m%d%H%M%S", time.localtime()))

    plt.savefig(path + str_l)


def test_func():
    transer = transmitter()
    transer.set_carrier_frequency(2500000)
    transer.set_filter_span(16)
    transer.set_modulation_type("QPSK")
    transer.set_oversamping_factor(4)
    transer.set_roll_ratio(0.5)
    transer.init_setting()

    transer.generate_signal_by_symbol_num(symbols_num = 32)
    save_pic(transer)
    pass


    





if __name__ == "__main__":
    test_func()
    print("test for pushing.")