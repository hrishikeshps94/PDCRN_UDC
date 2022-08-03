import sys
import torch.nn as nn
import torch
import numpy as np



class DWT(nn.Module):
    def __init__(self):
        super(DWT,self).__init__()
    def forward(self,input):
        x01 = input[:,:,0::2,:] / 4.0
        x02 = input[:,:,1::2,:] / 4.0
        x1 = x01[:, :,:, 0::2]
        x2 = x01[:, :,:, 1::2]
        x3 = x02[:, :,:, 0::2]
        x4 = x02[:, :,:, 1::2]
        y1 = x1+x2+x3+x4
        y2 = x1-x2+x3-x4
        y3 = x1+x2-x3-x4
        y4 = x1-x2-x3+x4
        return torch.cat([y1, y2, y3, y4], axis=1)

class IWT(nn.Module):
    def __init__(self,scale=2,device_name='cuda:0'):
        super(IWT,self).__init__()
        self.upsampler = nn.PixelShuffle(scale)
        self.device = device_name

    def kernel_build(self,input_shape):
        c = input_shape[1]
        out_c = c >> 2
        kernel = np.zeros((c, c,1, 1), dtype=np.float32)
        for i in range(0, c, 4):
            idx = i >> 2
            kernel[idx,idx::out_c,0,0]          = [1, 1, 1, 1]
            kernel[idx+out_c,idx::out_c,0,0]    = [1,-1, 1,-1]
            kernel[idx+out_c*2,idx::out_c,0,0]  = [1, 1,-1,-1]
            kernel[idx+out_c*3,idx::out_c,0,0]  = [1,-1,-1, 1]
        self.kernel = torch.tensor(data=kernel,dtype=torch.float32,requires_grad=True).cuda()
        return None

    def forward(self,input):
        self.kernel_build(input.shape)
        y = nn.functional.conv2d(input,weight = self.kernel, padding='same')
        y = self.upsampler(y)
        return y


class SeparableConv2d(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = torch.nn.Conv2d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels,
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv2d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
    
    def forward(self, x):
        return self.pointConv(self.spatialConv(x))
