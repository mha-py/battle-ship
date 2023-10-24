


from bs_unet import *



softmax = nn.Softmax(dim=1)

class Net(nn.Module):
    def __init__(self, n=32):
        super().__init__()
        self.unet = bs_unet(n)
        self.bn1 = torch.nn.BatchNorm1d(512) ##########
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.vlin1 = torch.nn.Linear(2*2*8*n, 512)
        self.vlin2 = torch.nn.Linear(512, 256)
        self.vlin3 = torch.nn.Linear(256, 1)
        self.cuda()
        
    def forward(self, x):
        self.unet(x)
        p = self.unet.p
        b, h, w, c = p.shape
        p = p.reshape(b, h*w*c)
        p = softmax(p)
        p = p.reshape(b, h, w, c)
        
        x = self.unet.x.reshape(b, -1)
        #x = relu(self.bn1(self.vlin1(x))) ############
        #x = relu(self.bn2(self.vlin2(x))) 
        x = relu(self.vlin1(x)) ############
        x = relu(self.vlin2(x)) 
        v = self.vlin3(x)
        
        return p, v