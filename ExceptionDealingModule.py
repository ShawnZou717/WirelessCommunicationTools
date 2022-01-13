# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:50:47 2021

@author: ShawnZou

@purpose:
    This module is created to make transmitter simulation more easily.

@functions:
    1. baseband_generator(n = 1)
        input:n  type:int  notion:number of bits you want to generate
        output:not named  type:list  notion:list of bits

"""

from inspect import stack
import time


class log:
    def __init__(self, module_name, code_dict):
        self._module_name = module_name
        self._log_file = self._open_log()
        self._code_dict = code_dict

    def _open_log(self):
        return open(self._module_name+".log", "w")

    def close(self):
        self._log_file.close()

    def _print_log(self, level, content):
        stack_info = stack()
        filename = stack_info[2][1].split("\\")[-1]
        function_name = stack_info[2][3]
        lineno = stack_info[2][2]

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        str_l = "[%s\t%s\t%s:%d|%s] %s\n" % (level, time_str, filename, lineno, function_name, content)
        
        if level != "INFO":
            print(str_l)
        
        self._log_file.write(str_l)
        self._log_file.flush()

    def info(self, content):
        self._print_log("INFO", content)

    def warn(self, content):
        self._print_log("WARN", content)

    def error(self, code, addition_content = ""):
        self._print_log("ERROR", self._code_dict[code]+addition_content)





