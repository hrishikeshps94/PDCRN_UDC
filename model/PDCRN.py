from turtle import forward
import torch
import torch.nn as nn
from core import DWT,IWT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DilationPyramid(nn.Module):
    def __init__(self,num_filters,dilation_rates):
        super(DilationPyramid,self).__init__()
        self.layer_1 = nn.Conv2d(num_filters,num_filters*2,3,padding='same')
        self.layer_2 = nn.Conv2d(num_filters*2,num_filters,3,padding='same',dilation=dilation_rates[0])
        self.layer_3 = []
        for dil_rate in dilation_rates[1:]:
          self.layer_3.append(nn.Conv2d(num_filters,num_filters,3,padding='same',dilation=dil_rate))
          self.layer_3.append(nn.ReLU())
        self.layer_3 = nn.Sequential(*self.layer_3)
        # self.layer_3 = nn.Sequential(*[[nn.Conv2d(num_filters,num_filters,3,padding='same',dilation=dil_rate),nn.ReLU()] for dil_rate in dilation_rates[1:]])
        self.layer_4 = nn.Conv2d(num_filters*2,num_filters,1,padding='same')
    def forward(self,input):
        x = self.layer_1(input)
        x = nn.functional.relu(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = torch.cat([input,x],dim=1)
        x = self.layer_4(x)
        out = nn.functional.relu(x)
        return out


class PyramidBlock(nn.Module):
    def __init__(self,num_filters,dilation_rates,nPyramidFilters):
        super(PyramidBlock,self).__init__()
        self.feat_extract = nn.Sequential(*[DilationPyramid(nPyramidFilters,dilation_rates),\
        nn.Conv2d(num_filters,num_filters,3,padding='same')])
    def forward(self,input):
        x = self.feat_extract(input)*0.1
        return input+x



class UDC_Arc(nn.Module):
    def __init__(self,device,in_ch,num_filters,dilation_rates,nPyramidFilters):
        super(UDC_Arc,self).__init__()
        self.encoder = nn.Sequential(*nn.ModuleList([DWT(),nn.PixelUnshuffle(downscale_factor=2),\
        nn.Conv2d(in_channels=in_ch*4*4,out_channels=num_filters,kernel_size=5,padding='same'),\
        nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,padding='same'),\
        PyramidBlock(num_filters,dilation_rates,nPyramidFilters),\
        nn.Conv2d(num_filters,num_filters*2,kernel_size=5,stride=2,padding=2),\
        PyramidBlock(num_filters*2,dilation_rates,nPyramidFilters*2),\
        nn.Conv2d(num_filters*2,num_filters*4,kernel_size=5,stride=2,padding=2),\
        PyramidBlock(num_filters*4,dilation_rates,nPyramidFilters*4)
        ]))
        self.decoder = nn.Sequential(*nn.ModuleList([PyramidBlock(num_filters*4,dilation_rates,nPyramidFilters*4),\
        nn.ConvTranspose2d(num_filters*4,num_filters*2,kernel_size = 4,stride=2,padding=1),\
        PyramidBlock(num_filters*2,dilation_rates,nPyramidFilters*2),\
        nn.ConvTranspose2d(num_filters*2,num_filters,kernel_size = 4,stride=2,padding=1),\
        PyramidBlock(num_filters,dilation_rates,nPyramidFilters),\
        nn.PixelShuffle(upscale_factor=2),nn.Conv2d(num_filters//4,in_ch*4,3,padding='same'),IWT(device_name=device),\
        # nn.Tanh()
        ]))
        self.relu = nn.ReLU()
    def forward(self,input):
        x_enc = self.encoder(input)
        x_dec = self.decoder(x_enc)
        x_dec = self.relu(x_dec)
        return x_dec
