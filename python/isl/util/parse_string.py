from ast import keyword
from calendar import leapdays
from tabnanny import check
import numpy as np
import pyparsing as pp

'''
    The purpose of this module is to parse strings. 
    The strings should consist of keywords that you also could 
    write directly using python.
'''

Integers = pp.Word(pp.nums)
Floats   = Integers + "." + Integers
Numbers  = (Floats | Integers)
Letters  = pp.Word(pp.alphas)
List     = "[" + pp.delimitedList(Numbers, delim=",") + "]"

def parse_rand(input):
    rand_template = "rand" + ":" + Numbers + ":" + Numbers + pp.LineEnd()
    try:
        rand_template.parse_string(input)
        return True
    except:
        return False

def parse_ru(input):
    ru_template = "ru" + ":" + Numbers + ":" +  List + pp.LineEnd()
    try:
        ru_template.parse_string(input)
        return True
    except:
        return False

def parse_array(input):
    array_input_template = List + pp.LineEnd()
    try:
        array_input_template.parse_string(input)
        return True
    except:
        return False

def parse_rotation(input):
    keywords           = ["rx", "ry", "rz"]
    rot_input_template = Letters + ":" + Numbers + pp.LineEnd()
    try: 
        rot_input_array    = rot_input_template.parse_string(input)
        if rot_input_array[0] in keywords:
            return True
    except:
        return False

def parse_keywords(input):
    keywords = ["identity"]
    return input in keywords