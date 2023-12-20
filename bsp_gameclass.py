
from bs_unet import *
from bs_helpers import *
from bsp_helpers import *


SX = SY = 10

_probabilities = dict()
@torch.no_grad
def getprobability(s, net=unet):
    shash = GameClass.getHashable(s)
    if shash not in _probabilities:
        net.eval()
        xs = np2t([encode(s.hist, s.shipnum)])
        yp = t2np(net(xs)).reshape((10,10,3))
        _probabilities[shash] = yp
    return _probabilities[shash]
    
    
class GameState:
    def __init__(self, shipnum):
        self.shipnum = shipnum
        self.sea = np.zeros((SX, SY), 'uint8')
        self.det = np.zeros((SX, SY), 'uint8')
        self.hist = []
        
    def copy(self):
        other = GameState()
        other.shipnum = self.shipnum
        other.sea = self.sea.copy()
        other.det = self.det.copy()
        other.hist = self.hist.copy()
        return other


class GameClass:
    
    # numActions = SX*SY
    
    @staticmethod
    def getNextState(s, a, ships=None):
        s = s.copy()
        assert s.det[a] == 0, 'Square already discovered'
        if type(ships) is type(None):  # probablistic
            p = getprobabilities(s)[a]
            t = np.random.choice(range(3), p=p)  # 0: no hit, 1: hit, 2: hit and sunk
            s.det[a] = 1
            s.sea[a] = t>0
            self.hist.append((a, t>0, t>1))
        else: # deterministic
            sea = ships.sum(0)
            s.sea[a] = sea[a]
            s.det[a] = 1
            s.hist.append((a, sea[a], sinksShip(ships, s.det, a))
        return s
            
    @staticmethod
    def getValidActions(s):
        return [ (i, j) for i in range(SX) for j in range(SY) if not s.det[i,j] ]
            
    @staticmethod
    def getEnded(s):
        if np.sum(s.sea) >= sum(s.shipnum) or np.sum(det) == SX*SY:
            return np.sum(s.sea)/np.sum(s.det)
        return None
    
    @staticmethod
    def getHashable(s):
        return repr(s.hist)
    
    

def plot_state(s):
    plot_sea(s.sea, s.det)
    
    
    
    
    