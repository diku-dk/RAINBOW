import numpy as np
import pyparsing as pp

'''
    The purpose of this module is to parse strings. 
    The strings should consist of keywords that you also could 
    write directly using python.
'''

Integer          = pp.Regex("([1-9]\d*|0)")
Float            = Integer + "." + Integer
Number_unsigned  = (Float | Integer)
Number_signed    = ("-" + Number_unsigned | Number_unsigned)
Number_pos_scientific  = Number_signed + "e" + "+" + Number_unsigned
Number_neg_scientific  = Number_signed + "e" + "-" + Number_unsigned
Number_scientific      = (Number_pos_scientific | Number_neg_scientific)
Number                 = (Number_scientific | Number_signed)
Letters  = pp.Word(pp.alphas)
List     = "[" + pp.delimitedList(Number, delim=",") + "]"

def parse_string_to_random_range_check(input):
    rand_template = "rand" + ":" + Number + ":" + Number + pp.LineEnd()
    try:
        rand_template.parse_string(input)
        return True
    except:
        return False

def parse_string_to_ru_check(input):
    ru_template = "ru" + ":" + Number + ":" +  List + pp.LineEnd()
    try:
        ru_template.parse_string(input)
        return True
    except:
        return False

def parse_string_to_array_check(input):
    array_input_template = List + pp.LineEnd()
    try:
        array_input_template.parse_string(input)
        return True
    except:
        return False

def parse_string_to_rotation_check(input):
    keywords           = ["rx", "ry", "rz"]
    rot_input_template = Letters + ":" + Number + pp.LineEnd()
    try: 
        rot_input_array    = rot_input_template.parse_string(input)
        if rot_input_array[0] in keywords:
            return True
    except:
        return False

def parse_string_to_keywords_check(input):
    keywords = ["identity"]
    return input in keywords