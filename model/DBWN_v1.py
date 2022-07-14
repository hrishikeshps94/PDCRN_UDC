from turtle import forward
import torch
import torch.nn as nn
from core import SeparableConv2d


class SmoothedDilatedResidualBlock(nn.Module):
    def __init__(self,kernels=(1,3,7,15),dilations=(1,2,4,8),num_filters=64) -> None:
        super(SmoothedDilatedResidualBlock,self).__init__()
        model_layers  = {}
        self.length = len(kernels) 
        for layer_count in range(len(kernels)):
            model_layers[str(layer_count)] = nn.Sequential(SeparableConv2d(num_filters,num_filters,kernel_size=kernels[layer_count],padding='same'),nn.ReLU(),\
                nn.Conv2d(num_filters,num_filters,kernel_size=3,dilation=dilations[layer_count],padding='same'),nn.ReLU())
        self.model_layers = nn.ModuleDict(model_layers)
        self.out_conv = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding='same')
    def forward(self,input):
        out = 0
        for layer_count in range(self.length):
            out+=self.out_conv(self.model_layers[str(layer_count)](input))
        out = out+input
        return out

class LowFrequencyReconstructNet(nn.Module):
    def __init__(self,in_ch,out_ch,num_filters) -> None:
        super(LowFrequencyReconstructNet,self).__init__()
        self.in_conv = nn.Conv2d(in_ch,num_filters,kernel_size=3,padding='same')
        self.lvl_1 = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock() for _ in range(3)]))
        self.lvl_2 = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock() for _ in range(2)]))
        self.lvl_3 = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock() for _ in range(4)]))
        self.lvl_4 =  nn.Sequential(*nn.ModuleList([nn.Conv2d(num_filters,num_filters,kernel_size=3,padding='same'),SmoothedDilatedResidualBlock(),SmoothedDilatedResidualBlock()]))
        self.lvl_5 = nn.Sequential(*nn.ModuleList([nn.Conv2d(num_filters,num_filters,kernel_size=3,padding='same'),SmoothedDilatedResidualBlock(),SmoothedDilatedResidualBlock(),\
            nn.Conv2d(num_filters,out_ch,kernel_size=3,padding='same')]))
        self.inter_lvl_1 = SmoothedDilatedResidualBlock()
        self.inter_lvl_2 = SmoothedDilatedResidualBlock()
        self.down = nn.Upsample(scale_factor=0.5)
        self.up = nn.Upsample(scale_factor=2)
    def forward(self,input):
        x_in = self.in_conv(input)
        x_lvl_1 = self.lvl_1(x_in)
        x_lvl_2 = self.down(x_lvl_1)
        x_lvl_2 = self.lvl_2(x_lvl_2)
        x_lvl_3 = self.down(x_lvl_2)
        x_lvl_3 = self.lvl_3(x_lvl_3)
        x_lvl_4 = self.up(x_lvl_3)+self.inter_lvl_2(x_lvl_2)
        x_lvl_4 = self.lvl_4(x_lvl_4)
        x_lvl_5 = self.up(x_lvl_4)+self.inter_lvl_1(x_lvl_1)
        x_out = self.lvl_5(x_lvl_5)
        return x_out


class OrginalReslutionBlock(nn.Module):
    def __init__(self,out_ch=32,num_filters=32) -> None:
        super(self,OrginalReslutionBlock).__init__()
        self.layer_1  = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock(num_filters=num_filters) for _ in range(6)]))
        self.out_conv = nn.Conv2d(out_ch,num_filters,kernel_size=3,padding='same')
    def forward(self,input):
        x = self.layer_1(input)
        return self.out_conv(x)




# model = OrginalReslutionBlock().to('cuda:0')
# out = model(torch.randn(1,3,256,256).to('cuda:0'))      



