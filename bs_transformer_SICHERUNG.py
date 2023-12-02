# https://github.com/hyunwoongko/transformer
# https://github.com/devjwsong/transformer-translator-pytorch/tree/master
# https://towardsdatascience.com/building-a-chess-engine-part2-db4784e843d5

from utils import *
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, n):
        super().__init__()
        maxlength = 500
        self.emb = torch.zeros((1, maxlength, n))
        
        pos = torch.arange(0, maxlength).float()
        pos = pos[None, :, None]
        _2i = torch.arange(0, n, 2).float()
        _2i = _2i[None, None, :]
        self.emb[0, :, 0::2] = torch.sin(pos/(10000**(_2i/n)))
        self.emb[0, :, 1::2] = torch.cos(pos/(10000**(_2i/n)))
        
        if GPU:
            self.emb = self.emb.cuda()
        self.emb.requires_grad = False
        
    def forward(self, x):
        b, p, n = x.shape
        return x + self.emb[:,:p,:].repeat(b, 1, 1)
    
    
class PositionalEncoding2d(nn.Module):
    def __init__(self, n):
        super().__init__()
        maxlength = 500
        self.rows = PositionalEncoding(n//2)
        self.cols = PositionalEncoding(n//2)
        
    def forward(self, x):
        b, h, w, c = x.shape
        emb = torch.cat((self.rows.emb[:,:h,None,:].repeat(b, 1, w, 1), 
                         self.cols.emb[:,None,:w,:].repeat(b, h, 1, 1)), dim=-1)
        return x + emb
    

class LayerNorm(nn.Module):
    def __init__(self, n, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n))
        self.beta = nn.Parameter(torch.zeros(n))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out



def _Att(q, k, v, mask=None):
    b, i, m = q.shape
    b, j, m = k.shape
    b, j, n = v.shape
    ###beta = torch.bmm(k.permute(0,2,1), q) # has dimensions b, p, p
    ###beta = beta / np.sqrt(m)
    beta = torch.einsum('bim, bjm -> bij', q, k) / np.sqrt(m)

    if mask is not None:
        beta = beta -9999*mask

    beta = F.softmax(beta, dim=-1)

    o = torch.einsum('bij,bjn->bin', beta, v)
    ###o = torch.bmm(v, beta)
    ###o = o.reshape(b, n, pq)
    return o



class SelfAttentionLayer(nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.q = nn.Linear(n, m)
        self.k = nn.Linear(n, m)
        self.v = nn.Linear(n, n)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        return _Att(q, k, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.q = nn.Linear(n, m)
        self.k = nn.Linear(n, m)
        self.v = nn.Linear(n, n)
        self.p = nn.Linear(n, n)
        self.nh = nh
        assert n%nh==m%nh==0, 'n and m must be disible by number of heads'

    def forward(self, x, y, z, mask=None):
        b, n, px = x.shape
        b, n, py = y.shape
        
        q = rearrange(self.q(x), 'b p (h n) -> (b h) p n', h=self.nh)
        k = rearrange(self.k(y), 'b p (h n) -> (b h) p n', h=self.nh)
        v = rearrange(self.v(z), 'b p (h n) -> (b h) p n', h=self.nh)

        ###q = self.q(x).reshape(b*nh, -1, px)
        ###k = self.k(y).reshape(b*nh, -1, py)
        ###v = self.v(z).reshape(b*nh, -1, py)

        x = _Att(q, k, v, mask)
        ###x = x.reshape(b, n, px)
        x = rearrange(x, '(b h) p n -> b p (h n)', h=self.nh)
        
        x = self.p(x)

        return x
    
    
class FeedForwardLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.dense1 = nn.Linear(n, 4*n)
        self.dense2 = nn.Linear(4*n, n)
    def forward(self, x):
        x = x + self.dense2(relu(self.dense1(x)))
        return x
        


class EncoderBlock(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.mha = MultiHeadAttention(n, m, nh)
        self.ln1 = LayerNorm(n)
        self.ff = FeedForwardLayer(n)
        self.ln2 = LayerNorm(n)
    def forward(self, x):
        x = x + self.mha(x, x, x)
        x = self.ln1(x)
        x = self.ff(x)
        x = self.ln2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.mha1 = MultiHeadAttention(n, m, nh)
        self.ln1 = LayerNorm(n)
        self.mha2 = MultiHeadAttention(n, m, nh)
        self.ln2 = LayerNorm(n)
        self.ff = FeedForwardLayer(n)
        self.ln3 = LayerNorm(n)
    def forward(self, x, y):
        x = x + self.mha1(x, x, x)
        x = self.ln1(x)
        x = x + self.mha2(x, y, y)
        x = self.ln2(x)
        x = self.ff(x)
        x = self.ln3(x)
        return x

    
class MultiAttentionBlock(nn.Module):
    'Quite similar to the EncoderBlock but with two inputs x and y'
    def __init__(self, n, m, nh):
        super().__init__()
        self.mha = MultiHeadAttention(n, m, nh)
        self.ln1 = LayerNorm(n)
        self.dense1 = nn.Linear(n, 4*n)
        self.dense2 = nn.Linear(4*n, n)
        self.ln2 = LayerNorm(n)
    def forward(self, x, y):
        x = x + self.mha(x, y, y)
        x = self.ln1(x)
        x = x + self.dense2(relu(self.dense1(x)))
        x = self.ln2(x)
        return x
    

class TOL(nn.Module):
    def __init__(self, n, m, nh, M):
        super().__init__()
        self.emb = nn.Linear(M, n, bias=False)
        self.mab = MultiAttentionBlock(n, m, nh)
        self.y = torch.eye(M).cuda() # shape b, M, M
    def forward(self, x):
        b, p, n = x.shape
        y = self.emb(self.y.repeat(b, 1, 1))
        y = self.mab(y, x)
        return y


class ISAB(nn.Module):
    def __init__(self, n, m, nh, M):
        super().__init__()
        self.tol = TOL(n, m, nh, M)
        self.mab2 = MultiAttentionBlock(n, m, nh)

    def forward(self, x):
        h = self.tol(x)
        return x + self.mab2(x, h)
    


'''
class Net(nn.Module):
    def __init__(self, n=256):
        super().__init__()
        self.dense1 = nn.Linear(22+8, n)
        self.dense12 = nn.Linear(n, n)
        self.dense13 = nn.Linear(n, n)
        self.posemb = PositionalEncoding(n)
        self.posemb2d = PositionalEncoding2d(n)
        self.enc1 = EncoderBlock(n, n//4, 2)
        self.enc2 = EncoderBlock(n, n//4, 2)
        self.enc3 = EncoderBlock(n, n//4, 2)
        self.dense2 = nn.Linear(n, 4*n)
        self.tol = TOL(4*n, n, 8, 1)
        self.mha1 = MultiHeadAttention(4*n, n, 2)
        self.mha2 = MultiHeadAttention(4*n, n, 2)
        self.mha3 = MultiHeadAttention(4*n, n, 2)
        self.ff1 = FeedForwardLayer(4*n)
        self.ff2 = FeedForwardLayer(4*n)
        self.ff3 = FeedForwardLayer(4*n)
        self.ln1 = LayerNorm(4*n)
        self.ln2 = LayerNorm(4*n)
        self.ln3 = LayerNorm(4*n)
        self.ln4 = LayerNorm(4*n)
        self.dense3 = nn.Linear(4*n, 100)
        self.cuda()
        self.n = n

    def forward(self, x):
        x = self.dense1(x)
        x = self.posemb(x)
        x = relu(self.dense12(x))
        x = relu(self.dense13(x))
        x = self.enc1(x)
        x = self.enc2(x)
        #x = self.enc3(x)
        x0 = self.dense2(x)
        
        x = self.tol(x0)
        x = self.ln1(x + self.mha1(x, x0, x0))
        x = self.ln2(self.ff1(x))
        x = self.ln3(x + self.mha2(x, x0, x0))
        x = self.ln4(self.ff2(x))
        #x = self.ff3(x)
        #x = x + self.mha3(x, x0, x0)
        x = self.dense3(x)
        x = x.reshape(-1, 10*10)
        x = torch.sigmoid(x)
        return x
    '''
    

class Net(nn.Module):
    def __init__(self, n=256):
        super().__init__()
        self.dense1 = nn.Linear(22+8, n)
        self.posemb = PositionalEncoding(n)
        self.posemb2d = PositionalEncoding2d(n)
        self.enc1 = EncoderBlock(n, n//4, 4)
        self.enc2 = EncoderBlock(n, n//4, 4)
        self.enc3 = EncoderBlock(n, n//4, 4)
        self.dec1 = DecoderBlock(n, n//4, 4)
        self.dec2 = DecoderBlock(n, n//4, 4)
        self.dec3 = DecoderBlock(n, n//4, 4)
        self.tol = TOL(n, n//4, 4, SX**2)
        self.dense2 = nn.Linear(n, 1)
        self.cuda()
        self.n = n

    def forward(self, x):
        x = self.dense1(x)
        x = self.posemb(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x0 = self.enc3(x)
        
        b, p, c = x.shape
        SX = 10
        x = torch.zeros(b, SX, SX, c).cuda()
        
        ##x = self.tol(x)
        ##x = x.reshape(b, SX, SX, c)
        x = self.posemb2d(x)
        x = x.reshape(b, SX**2, c)
        x = self.dec1(x, x0)
        x = self.dec2(x, x0)
        x = self.dec3(x, x0)
        x = self.dense2(x)
        x = x.reshape(-1, 10*10)
        x = torch.sigmoid(x)
        return x
