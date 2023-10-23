
import torch
from torch import nn
from utils import *


def encode_x(sea, det):
    'Encodes the visible information to an input to the neural network'
    return np.stack([(1-sea)*det, (1-det), sea*det], -1)
   
    
    
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()



class bship_unet(nn.Module):
    def __init__(self, n=32):
        'BattleShip CNN. More than 1 blocks didnt make a difference.'
        super().__init__()
        self.conv1 = nn.Conv2d(3, n, 5, padding=2) # 10x10
        self.down1 = ResBlockDown(n, 2*n)  # 5x5
        self.down2 = ResBlockDown(2*n, 4*n) # 3x3
        self.down3 = ResBlockDown(4*n, 8*n) # 2x2
        self.resblock4 = ResBlock(8*n, 8*n)
        self.resblock3 = ResBlock(4*n, 4*n)
        self.resblock2 = ResBlock(2*n, 2*n)
        self.resblock1 = ResBlock(n, n)
        self.up3 = ResBlockUp(8*n, 4*n)
        self.up2 = ResBlockUp(4*n, 2*n)
        self.up1 = ResBlockUp(2*n, n)
        self.conv_m1 = nn.Conv2d(n, 1, 3, padding=1)
        self.cuda()
        
    def forward(self, x):
        # NHWC zu NCHW
        x = x.permute([0, 3, 1, 2])
        
        x = relu(self.conv1(x))
        #print(x.shape)
        xskip1 = x
        x = expandtoeven(x)
        x = self.down1(x)
        #print(x.shape)
        xskip2 = x
        x = expandtoeven(x)
        x = self.down2(x)
        #print(x.shape)
        xskip3 = x
        x = expandtoeven(x)
        x = self.down3(x)
        #print(x.shape)

        x = self.resblock4(x)
        #print(x.shape)
        self.x = x
        
        x = self.up3(x)
        x = addskip(x, xskip3)
        x = self.resblock3(x)
        #print(x.shape)
        x = self.up2(x)
        x = addskip(x, xskip2)
        x = self.resblock2(x)
        #print(x.shape)
        x = self.up1(x)
        #print(x.shape, xskip1.shape)
        x = addskip(x, xskip1)
        x = self.resblock1(x)
        #print(x.shape)
        
        x = self.conv_m1(x)
        self.p = x
        x = sigmoid(x)
        
        # NCHW zu NHWC
        x = x.permute([0, 2, 3, 1])
        return x
    
    def predict(self, x):
        'Takes a numpy array and give out one, i. e. 10x10 -> 10x10'
        x = np2t(x[None,:])
        y = self(x)
        return t2np(y[0,:,:,0])
    
    
    
def augment(x, y):
    r = np.random.rand
    if r()<0.5:
        x, y = x.flip(1), y.flip(1)
    if r()<0.5:
        x, y = x.flip(2), y.flip(2)
    if r()<0.5:
        x, y = x.transpose(1,2), y.transpose(1,2)
    return x, y




