from turtle import forward
import torch
import torch.nn as nn
from core import IWT, SeparableConv2d,DWT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.lvl_1 = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock(num_filters=num_filters) for _ in range(3)]))
        self.lvl_2 = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock(num_filters=num_filters) for _ in range(2)]))
        self.lvl_3 = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock(num_filters=num_filters) for _ in range(4)]))
        self.lvl_4 =  nn.Sequential(*nn.ModuleList([nn.Conv2d(num_filters,num_filters,kernel_size=3,padding='same'),SmoothedDilatedResidualBlock(num_filters=num_filters),SmoothedDilatedResidualBlock(num_filters=num_filters)]))
        self.lvl_5 = nn.Sequential(*nn.ModuleList([nn.Conv2d(num_filters,num_filters,kernel_size=3,padding='same'),SmoothedDilatedResidualBlock(num_filters=num_filters),SmoothedDilatedResidualBlock(num_filters=num_filters),\
            nn.Conv2d(num_filters,out_ch,kernel_size=3,padding='same')]))
        self.inter_lvl_1 = SmoothedDilatedResidualBlock(num_filters=num_filters)
        self.inter_lvl_2 = SmoothedDilatedResidualBlock(num_filters=num_filters)
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
        super(OrginalReslutionBlock,self).__init__()
        self.layer_1  = nn.Sequential(*nn.ModuleList([SmoothedDilatedResidualBlock(num_filters=num_filters) for _ in range(6)]))
        self.out_conv = nn.Conv2d(num_filters,out_ch,kernel_size=3,padding='same')
    def forward(self,input):
        x = self.layer_1(input)
        return self.out_conv(x)


class HighFrequencyReconstructionNet(nn.Module):
    def __init__(self,in_ch,out_ch,num_filters=32) -> None:
        super(HighFrequencyReconstructionNet,self).__init__()
        self.in_conv = nn.Conv2d(in_ch,num_filters,kernel_size=3,padding='same')
        self.layer_1 = SmoothedDilatedResidualBlock(num_filters=num_filters)
        self.layer_2 = nn.Sequential(*nn.ModuleList([OrginalReslutionBlock(out_ch=num_filters,num_filters=num_filters) for _ in range(3)]))
        self.out_layer = nn.Conv2d(num_filters,out_ch,kernel_size=3,padding='same')
    def forward(self,input):
        x = self.in_conv(input)
        x = self.layer_1(x)
        x= self.layer_2(x)
        out = self.out_layer(x)
        return out



class DBWN(nn.Module):
    def __init__(self) -> None:
        super(DBWN,self).__init__()
        self.downsampler = DWT()
        self.HF_layer = HighFrequencyReconstructionNet(in_ch=12,out_ch=24,num_filters=32)
        self.LF_layer = LowFrequencyReconstructNet(in_ch=3,out_ch=12,num_filters=64)
        self.upsampler_HF = IWT()
        self.upsampler_LF = nn.Upsample(scale_factor=4)
    def forward(self,input):
        x_HF = self.downsampler(input)
        x_LF = self.downsampler(x_HF[:,:3,:,:])[:,:3,:,:]
        x_HF_out = self.HF_layer(x_HF)
        alpha = self.upsampler_HF(x_HF_out[:,:12,:,:])
        beta = self.upsampler_HF(x_HF_out[:,12:,:,:])
        x_HF_inter = (input*alpha)+beta
        x_LF_out = self.LF_layer(x_LF)
        gamma  = self.upsampler_LF(x_LF_out[:,:9,:,:])
        etta = self.upsampler_LF(x_LF_out[:,9:,:,:])
        x_out = torch.einsum('blhw,bchw->bchw',gamma,x_HF_inter)+etta
        return x_out

# model = DBWN().to(device)
# out = model(torch.randn(1,3,256,256).to(device))
# print(out.shape)




