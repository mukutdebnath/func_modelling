import torch
import torch.nn as nn

# threshold has shape [1, #adc levels]
# x.unsqueeze(-1) has shape [shape, 1]
# the comparison gives output of shape  [shape, #adc_levles]
# sum is taken of last dimension i.e., -1 which gives the number of times the 
# comparison output is True, which gives the count in integer,
# convert it to float.
# scale down to fractional bits
    
class ADCActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, adc_f_bits, adc_char, zero_off, bit_scale):
        ctx.save_for_backward(input)
        thresholds=adc_char.to(input.device)
        quant_op=torch.sum(input.unsqueeze(-1) >= thresholds, dim=-1).float()
        quant_op=quant_op/2**adc_f_bits
        op=bit_scale*(quant_op+zero_off)
        return op
    
    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors        
        # grad_input = grad_output.clone()
        # grad_input[input < 0.0] = 0.0
        # grad_input[input > 1.0] = 0.0
        return grad_output, None, None, None, None

class ADCActivation(nn.Module):
    def __init__(self, adc_f_bits, adc_char, zero_off, bit_scale):
        super(ADCActivation, self).__init__()
        self.adc_f_bits=adc_f_bits
        self.adc_char=adc_char
        self.zer_off=zero_off
        self.bit_scale=bit_scale

    def forward(self, x):
        if (self.adc_char.shape==torch.Size([])): # no adc
            return x
        else:
            return ADCActivationFunction.apply(x, self.adc_f_bits, self.adc_char,
                                           self.zer_off, self.bit_scale)

if __name__=="__main__":
    from adcdata import ss_lin
    adc_bits=(7,4)
    adc_index=0
    bit_scale=1
    adc_range=(2**(adc_bits[0]-adc_bits[1])-1/2**adc_bits[1])*bit_scale
    adc_charac, zero_off=ss_lin(adc_bits=adc_bits[0], adc_index=adc_index, adc_range=adc_range)
    adcnet=ADCActivation(adc_f_bits=adc_bits[1], adc_characteristics=adc_charac, zero_offset=zero_off,bit_scale=bit_scale)
    x=torch.tensor((5.05))
    print(x)
    print(adcnet(x))