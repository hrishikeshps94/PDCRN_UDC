import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn
from model.core import DWT,IWT
from torch.nn import functional as F
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class DilationBlock(nn.Module):
    def __init__(self, in_c, rate_exponent):
        super(DilationBlock,self).__init__()
        self.dilated_conv_1 =  nn.Conv2d(in_c,in_c//2,3,padding='same',dilation= 2**(rate_exponent-1))
        self.dilated_conv_2 =  nn.Conv2d(in_c,in_c//2,3,padding='same',dilation= 2**(rate_exponent))
        self.dilated_conv_3 =  nn.Conv2d(in_c,in_c//2,3,padding='same',dilation= 2**(rate_exponent+1))
        self.dilated_conv_4 =  nn.Conv2d(in_c,in_c//2,3,padding='same',dilation= 2**(rate_exponent+2))
        self.conv = nn.Conv2d(in_c*2,in_c,3,padding='same')

    def forward(self,input):
        x1 = F.relu(self.dilated_conv_1(input))
        x2 = F.relu(self.dilated_conv_2(input))
        x3 = F.relu(self.dilated_conv_3(input))
        x4 = F.relu(self.dilated_conv_4(input))
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = F.relu(self.conv(x))
        x = x + input
        return x

class ResidualDilationBlock(nn.Module):
    def __init__(self, num_blocks, num_block_channels, rate_exponent=1):
        super(ResidualDilationBlock,self).__init__()
        self.blocks = nn.Sequential(*[DilationBlock(num_block_channels,rate_exponent) for i in range(num_blocks)])

    def forward(self,input):
        # block_input = input
        # for block in self.blocks:
        #     block_input = block(block_input)
        return self.blocks(input)
        # return block_input

class PALayer(nn.Module):
    def __init__(self, channel: int):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel: int):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class ResidualFFABlock(nn.Module):
    def __init__(self, in_c):
        super(ResidualFFABlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c,in_c,3,padding='same')
        self.conv2 = nn.Conv2d(in_c,in_c,3,padding='same')
        self.calayer = CALayer(in_c)
        self.palayer = PALayer(in_c)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        y = x + input
        x = self.conv2(x)
        x = self.palayer(self.calayer(x))
        x = x + input
        x = F.relu(x)
        return x

class HQNet(nn.Module):
    def __init__(self, in_c,num_block_groups, num_blocks, num_block_channels, rate_exponents=[1,2,3]):
        super(HQNet,self).__init__()
        assert num_block_groups == len(rate_exponents), "kernel size should be odd"
        self.conv_1 = nn.Conv2d(in_c,num_block_channels,3,padding='same')
        self.block_groups = nn.ModuleList([ResidualDilationBlock(num_blocks, num_block_channels, rate_exponents[i]) for i in range(num_block_groups)])
        self.residual = ResidualFFABlock(num_block_channels)
        self.gate_conv = nn.Conv2d(num_block_channels * (num_block_groups+1), num_block_groups+1, 3, padding='same',bias=True)
        self.conv_2 = nn.Conv2d(num_block_channels, num_block_channels, 3, padding='same')
        self.bottleneck_gain = nn.Conv2d(num_block_channels, in_c, 1, padding='same')
        self.bottleneck_bias = nn.Conv2d(num_block_channels, in_c, 1, padding='same')

    def forward(self,input):
        ga_inputs = [self.conv_1(input)]
        for block_group in self.block_groups:
            ga_inputs.append(block_group(ga_inputs[-1]))
        ga_inputs[-1] = self.residual(ga_inputs[-1])
        hq_out = torch.cat(ga_inputs, dim=1)
        hq_weighting = self.gate_conv(hq_out)
        gated_output = torch.stack([ga_inputs[i]*hq_weighting[:, [i]] for i in range(len(ga_inputs))], dim=1).sum(dim=1)
        gated_output = F.relu(self.conv_2(gated_output))
        bottleneck_gain = self.bottleneck_gain(gated_output)
        bottleneck_bias = self.bottleneck_bias(gated_output)
        return bottleneck_gain, bottleneck_bias

class DilationPyramid(nn.Module):
    def __init__(self, in_c, dilation_rates = [3,2,1,1]):
        super(DilationPyramid,self).__init__()   
        self.dilated_convs = nn.ModuleList([nn.Conv2d(in_c*(i+1), in_c, 3, padding='same',dilation=dilation_rates[i]) for i in range(len(dilation_rates))])
        self.bottleneck = nn.Conv2d(in_c*(len(dilation_rates)+1), in_c, 1, padding='same')

    def forward(self,input):
        x_in = input
        for dilated_conv in self.dilated_convs:
            x_out = F.relu(x_in)
            x_out = dilated_conv(x_out)
            x_in = torch.cat((x_in,x_out), dim=1)
        x_out = self.bottleneck(x_in)
        x_out = x_out+input
        return x_out

class PDCRN(nn.Module):
    def __init__(self, in_c, block_channels, num_blocks=4, dilation_rates = [7,5,3,2,1]):
        super(PDCRN,self).__init__()
        self.conv_1 = nn.Conv2d(in_c, block_channels, 3, padding='same')
        self.dilation_blocks = nn.ModuleList([DilationPyramid(block_channels, dilation_rates) for i in range(num_blocks)])
        self.conv_2 = nn.Conv2d(block_channels*4, block_channels, 3, padding='same')
        self.conv_3 = nn.Conv2d(block_channels, block_channels, 3, padding='same')
        self.conv_4 = nn.Conv2d(block_channels, 9, 1, padding='same')
        self.conv_5 = nn.Conv2d(block_channels, 3, 1, padding='same')
        self.upsample_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,input):
        x0 = self.conv_1(input)
        x0_act = F.relu(x0)
        x1 = self.dilation_blocks[0](x0)
        x2 = self.dilation_blocks[0](x1)
        x3 = self.dilation_blocks[0](x2)
        x4 = self.dilation_blocks[0](x3)
        x5 = torch.cat((x1,x2,x3,x4), dim=1)
        x5 = F.relu(x5)
        x5 = F.relu(self.conv_2(x5))
        x_out = x5 + x0_act
        x_out = F.relu(self.conv_3(x_out))
        x_out_gain = self.upsample_1(self.conv_4(x_out))
        x_out_bias = self.upsample_2(self.conv_5(x_out))
        # x_out_gain = F.interpolate(self.conv_4(x_out),scale_factor=2, mode='bilenear', align_corners=True)
        # x_out_bias = F.interpolate(self.conv_5(x_out),scale_factor=2, mode='bilenear', align_corners=True)
        return x_out_gain, x_out_bias

class DBWN(nn.Module):
    def __init__(self):
        super(DBWN,self).__init__()
        self.dwt = DWT()
        self.idwt = IWT()
        self.hq_net = HQNet(9, 3, 4, 32, [1,2,3])
        self.lr_net = PDCRN(3, 32)

    def forward(self,input):
        x = self.dwt(input)
        x_lq, x_hq = x[:,:3], x[:,3:]
        xhq_gain, xhq_bias = self.hq_net(x_hq)
        xhq_gain = torch.cat((x_lq,xhq_gain), dim=1)
        xhq_gain = self.idwt(xhq_gain)
        xhq_gain = xhq_gain*input
        xhq_bias = torch.cat((x_lq,xhq_bias), dim=1)
        xhq_bias = self.idwt(xhq_bias)
        xhq = xhq_gain+xhq_bias
        xlq_gain, xlq_bias = self.lr_net(x_lq)
        x_affine = torch.cat(((xlq_gain[:,:3]*xhq).sum(dim=1).unsqueeze(1),\
                              (xlq_gain[:,3:6]*xhq).sum(dim=1).unsqueeze(1),\
                              (xlq_gain[:,6:]*xhq).sum(dim=1).unsqueeze(1)), dim=1)
        x_out = x_affine + xlq_bias
        return x_out

# model = DBWN()
# summary(model.to(device),(3,256,256))

