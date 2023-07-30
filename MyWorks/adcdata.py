import pandas
import numpy as np
from scipy import interpolate
import torch

__all__=['ss_lin', 'cco_lin', 'ideal']

def get_thresholds(x, y): 
    # takes analog x and quantized y (full scaled mapping according to adc output)
    # gives output of the threshold levels where the adc digital code changes
    # zero offset is used to provide any initial count value if required
    # make sure to give fully scaled x and y values, 
    # such as x: analog 0 to 10
    # y: quantized 0 to 10

    op_diff = np.diff(y)
    change_idx = np.nonzero(op_diff)[0]   # [0] is used as return is [n,1] in shape
    th_levels = x[change_idx+1]    # +1 as indices returned are those after which the output y changes
    return th_levels

def ss_lin(adc_bits, adc_index, adc_range):
    data=pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/SS_ADC_10u.csv')
    data['delay_switch']=data['delay_switch']-data['delay_switch'].min()
    fclk=1/(data['delay_switch'].max()-data['delay_switch'].min()) # most ideal count evaluation
    corner_dict={
        0: 'nom',
        1: 'C0',
        2: 'C1',
        3: 'C2',
        4: 'C3'
    }
    data=data[data['Corner']==corner_dict[adc_index]]
    If=interpolate.interp1d(data['Iinput'],data['delay_switch'])
    x=np.arange(data['Iinput'].min(), data['Iinput'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*2**adc_bits*fclk)
    xnorm=x*adc_range/x.max()
    ynorm=ycount*adc_range/2**adc_bits
    th_levels=get_thresholds(xnorm, ynorm)
    return torch.tensor(th_levels), ynorm[0]

def cco_lin(adc_bits, adc_index, adc_range):
    data=pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_char_all_corners.csv')
    fmax=data.loc[99].max()
    corner_dict={
        0: 'Nom',
        1: 'C0',
        2: 'C1',
        3: 'C2',
        4: 'C3'
    }
    If=interpolate.interp1d(data['Iin'], data[corner_dict[adc_index]])
    x=np.arange(data['Iin'].min(), data['Iin'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*2**adc_bits/fmax)
    xnorm=x*adc_range/x.max()
    ynorm=ycount*adc_range/2**adc_bits
    th_levels=get_thresholds(xnorm, ynorm)
    return torch.tensor(th_levels), ynorm[0]

def ideal(adc_bits, adc_index, adc_range):
    return torch.tensor(np.arange(1,2**adc_bits,1)*adc_range/2**adc_bits), 0

def noadc(adc_bits, adc_index, adc_range):
    return torch.tensor((1)), 0

if __name__=="__main__":
    adc_bits=(7,4)
    adc_index=0
    range=2**(adc_bits[0]-adc_bits[1]) - 1/2**adc_bits[1]
    print(ss_lin(adc_bits=adc_bits[0], adc_index=adc_index, adc_range=range))