


from bs_unet import *



softmax = nn.Softmax()

class Net(nn.Module):
    def __init__(self, n=32):
        super().__init__()
        self.unet = battleship_unet(n)
        self.vlin1 = torch.Linear(2*2*8*n, 256)
        self.vlin2 = torch.Linear(256, 1)
        
    def forward(self, x):
        p = self.unet(x)
        b, h, w, c = p.shape
        p = p.reshape(b, h*w*c)
        p = softmax(p, 1)
        p = p.reshape(b, h, w, c)
        
        x = self.unet.x
        x = relu(self.vlin1(x))
        v = self.vlin2(x)
        
        return p, v