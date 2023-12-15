# https://github.com/hyunwoongko/transformer
# https://github.com/devjwsong/transformer-translator-pytorch/tree/master
# https://towardsdatascience.com/building-a-chess-engine-part2-db4784e843d5

from utils import *
from einops import rearrange

SX = 10


gelu = F.gelu

'''
class MultiEmbedding(nn.Module):
    def __init__(self, n, N, M):
        super().__init__()
        self.embs = nn.ModueList(nn.Embedding(N, n) for _ in range(M))

    def forward(self, x):
        return torch.stack([ self.embs[i](x[:,i]) ]).sum(dim=0)
'''



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

        ###self.emb -= torch.mean(self.emb, dim=1)
        
        self.emb.requires_grad = False
        
    def forward(self, x):
        b, p, n = x.shape
        self.emb = self.emb.to(x.device)
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
        return x + emb.to(x.device)
    

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



def get_relative_positions(n, m) -> torch.tensor:
    x = torch.arange(m)[None, :]
    y = torch.arange(n)[:, None]
    return torch.abs(x - y)


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


def _Att(q, k, v, mask=None, bias=True):
    b, h, i, m = q.shape
    b, h, j, m = k.shape
    b, h, j, n = v.shape
    
    beta = torch.einsum('bhim, bhjm -> bhij', q, k) / np.sqrt(m)
    if bias:
        m = get_alibi_slope(h)
        bias = m*get_relative_positions(i, j)
        bias = bias[None,:].to(q.device)
        beta = beta - bias

    if mask is not None:
        beta = beta.masked_fill(mask == 0, -1e12)

    beta = F.softmax(beta, dim=-1)
    
    if mask is not None:
        beta = beta.masked_fill(mask == 0, 0)  # make sure its really closed

    o = torch.einsum('bhij,bhjn->bhin', beta, v)
    
    return o, beta


'''
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

        return _Att(q, k, v)[0]'''


class MultiHeadAttention_SICHERUNG(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.q = nn.Linear(n, m)
        self.k = nn.Linear(n, m)
        self.v = nn.Linear(n, n)
        self.p = nn.Linear(n, n)
        self.nh = nh

    def forward(self, x, y, z, mask=None):

        q = rearrange(self.q(x), 'b p (h n) -> (b h) p n', h=self.nh)
        k = rearrange(self.k(y), 'b p (h n) -> (b h) p n', h=self.nh)
        v = rearrange(self.v(z), 'b p (h n) -> (b h) p n', h=self.nh)

        x, self.beta = _Att(q, k, v, mask)
        x = rearrange(x, '(b h) p n -> b p (h n)', h=self.nh)
        
        x = self.p(x)
        return x



'''
class MultiHeadAttention(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.q = nn.Linear(n, m)
        self.k = nn.Linear(n, m)
        self.v = nn.Linear(n, n)
        self.p = nn.Linear(n, n)
        self.nh = nh
        self.cat_mode = False

    def forward(self, x, y, z, mask=None):
        b, n, px = x.shape
        b, n, py = y.shape

        q = rearrange(self.q(x), 'b p (h n) -> (b h) p n', h=self.nh)
        k = rearrange(self.k(y), 'b p (h n) -> (b h) p n', h=self.nh)
        v = rearrange(self.v(z), 'b p (h n) -> (b h) p n', h=self.nh)

       # ic(q[0, 0, 0])
      #  ic(k[0, 0, 0])
       # ic(v[0, 0, 0])

        if self.cat_mode=='cat':
            if self.k_saved is not None:
                k = torch.cat((self.k_saved, k), dim=1)
                v = torch.cat((self.v_saved, v), dim=1)
    
            self.k_saved = k
            self.v_saved = v
            x, self.beta = _Att(q, k, v)
        else:
            x, self.beta = _Att(q, k, v, mask)
      #  ic(x[0, 0, 0])
            
        x = rearrange(x, '(b h) p n -> b p (h n)', h=self.nh)

        
        x = self.p(x)
        return x
    def reset(self):
        self.k_saved = None
        self.v_saved = None

'''

class MultiHeadAttention(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.q = nn.Linear(n, m)
        self.k = nn.Linear(n, m)
        self.v = nn.Linear(n, n)
        self.p = nn.Linear(n, n)
        self.nh = nh
        self.cat_mode = False

    def forward(self, x, y, z, mask=None):
        if self.cat_mode == 'save':
            # we assume that y and z are the same in each call
            if self.k_saved is not None:
                q = rearrange(self.q(x), 'b p (h n) -> b h p n', h=self.nh)
                k = self.k_saved
                v = self.v_saved
            else:
                q = rearrange(self.q(x), 'b p (h n) -> b h p n', h=self.nh)
                k = rearrange(self.k(y), 'b p (h n) -> b h p n', h=self.nh)
                v = rearrange(self.v(z), 'b p (h n) -> b h p n', h=self.nh)
                self.k_saved = k
                self.v_saved = v
            #q = rearrange(self.q(x), 'b p (h n) -> b h p n', h=self.nh) ### ##################################
            #k = rearrange(self.k(y), 'b p (h n) -> b h p n', h=self.nh) ###
            #v = rearrange(self.v(z), 'b p (h n) -> b h p n', h=self.nh) ###
                
        elif self.cat_mode == 'cat':
            # with each call, x is expanded and we concatenate the expansion to the part we already have
            q = rearrange(self.q(x), 'b p (h n) -> b h p n', h=self.nh)
            k = rearrange(self.k(y), 'b p (h n) -> b h p n', h=self.nh)
            v = rearrange(self.v(z), 'b p (h n) -> b h p n', h=self.nh)
            if self.k_saved is not None:
                k = torch.cat((self.k_saved, k), dim=2)
                v = torch.cat((self.v_saved, v), dim=2)
            self.k_saved = k
            self.v_saved = v
            mask = None
                
        else:
            # normal mode: full calculation
            q = rearrange(self.q(x), 'b p (h n) -> b h p n', h=self.nh)
            k = rearrange(self.k(y), 'b p (h n) -> b h p n', h=self.nh)
            v = rearrange(self.v(z), 'b p (h n) -> b h p n', h=self.nh)
            
        x, self.beta = _Att(q, k, v, mask)
        x = rearrange(x, 'b h p n -> b p (h n)', h=self.nh)
        x = self.p(x)
        
        return x
        
    def reset(self):
        self.k_saved = None
        self.v_saved = None
    
    
class FeedForwardLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.dense1 = nn.Linear(n, 4*n)
        self.dense2 = nn.Linear(4*n, n)
        self.dropout = nn.Dropout(0.1)
        self.ln = LayerNorm(n)
    def forward(self, x):
        y = x
        y = self.ln(y)
        y = self.dense1(y)
        y = gelu(y)
        y = self.dropout(y)
        y = self.dense2(y)
        y = self.dropout(y)
        return x + y
        

class EncoderBlock(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.mha = MultiHeadAttention(n, m, nh)
        self.ln = LayerNorm(n)
        self.ff = FeedForwardLayer(n)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, mask=None):
        y = self.ln(x)
        x = x + self.dropout(self.mha(y, y, y, mask))
        x = self.ff(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.mha1 = MultiHeadAttention(n, m, nh)
        self.ln1 = LayerNorm(n)
        self.mha2 = MultiHeadAttention(n, m, nh)
        self.ln2 = LayerNorm(n)
        self.ff = FeedForwardLayer(n)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, y, mask=None):
        z = self.ln1(x)
        x = x + self.dropout(self.mha1(z, z, z, mask))
        z = self.ln2(x)
        x = x + self.dropout(self.mha2(z, y, y, mask))
        return self.ff(x)
    def reset(self):
        self.mha1.reset()
    def cat_mode(mode):
        self.mha1.cat_mode = mode

    
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
        x = x + self.dense2(gelu(self.dense1(x)))
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
    
    

class Seed(nn.Module):
    def __init__(self, n, M):
        super().__init__()
        self.emb = nn.Linear(M, n, bias=False)
        self.M = M

    def forward(self, x):
        b, c, p = x.shape
        oh = torch.eye(self.M).repeat(b, 1, 1) # shape b, M, M
        oh = oh.to(x.device)
        y = self.emb(oh)
        return y


class ISAB2(nn.Module):
    def __init__(self, n, m, nh):
        super().__init__()
        self.mab1 = MultiAttentionBlock(n, m, nh)
        self.mab2 = MultiAttentionBlock(n, m, nh)

    def forward(self, x, y):
        y = self.mab1(y, x)
        x = self.mab2(x, y)
        return x, y
    



class Net(nn.Module):
    def __init__(self, n=256):
        super().__init__()
        NBUFF = 100
        self.dense1 = nn.Linear(NBUFF, n)
        self.posemb = PositionalEncoding(n)
        self.posemb2d = PositionalEncoding2d(n)
        self.ln1 = LayerNorm(n)
        self.ln2 = LayerNorm(n)
        self.ln3 = LayerNorm(n)
        self.ln4 = LayerNorm(n)
        self.enc1 = EncoderBlock(n, n//4, 4)
        self.enc2 = EncoderBlock(n, n//4, 4)
        self.enc3 = EncoderBlock(n, n//4, 4)
        self.dec1 = DecoderBlock(n, n//4, 4)
        self.dec2 = DecoderBlock(n, n//4, 4)
        self.dec3 = DecoderBlock(n, n//4, 4)
        self.dense2 = nn.Linear(n, 3)
        self.cuda()
        self.n = n

    def forward(self, x):
        x = self.dense1(x)
        x = self.posemb(x)
        x = self.ln1(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x0 = self.ln2(x)
        
        b, p, c = x.shape
        SX = 10
        x = torch.zeros(b, SX, SX, c).cuda()
        
        x = self.posemb2d(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.ln3(x)
        x = self.dec1(x, x0)
        x = self.dec2(x, x0)
        x = self.dec3(x, x0)
        x = self.ln4(x)
        x = self.dense2(x)
        x = F.softmax(x, -1)
        return x
