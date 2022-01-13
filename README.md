File SignalSimulatorTools.py are to make signal processing simulation easier. One can accomplish BPSK/QPSK modulation, add noise or shaping filtering after several simple codes. Following are the useful details about this file:

1. To use this file, use the import code:

`import SignalSimulatorTools as sst`

2. To generate baseband signals. The generated signal is a list composed of bits 0 and 1.

`signal = sst.generate_baseband_signal(n = 1)`

3. To transfer bit list to symbols with given modulation type. For instance, 00, 01, 10, 11 bit in QPSK are considered to be symbol of value 0, 1, 2, 3 respectively.

```
bit_m = sst.bit_message(signal, "QPSK")
symbols = bit_m.decode()
```

4. Generate a signal with AWGN and shaping by root raised cosine filter:
```
transer = sst.transmitter()
transer.set_carrier_frequency(2500000)
transer.set_filter_span(16)
transer.set_modulation_type("QPSK")
transer.set_oversamping_factor(4)
transer.set_roll_ratio(0.5) # if not set, will generate randomly in [0.1 ,0.5)
transer.set_snr(2.0) # if not set, will generate randomly in [-2 ,8)dB
transer.init_setting()

transer.generate_signal_by_symbol_num(symbols_num = 32)
sst.save_pic(transer, ".\\")
```
you can check the pig for details. Usually looking like this:
![QPSK_snr7 799812394542469_r0 5_20220113165630](https://user-images.githubusercontent.com/48830288/149298047-239bc388-592a-4fd2-9d42-ff6008650f11.jpg)
