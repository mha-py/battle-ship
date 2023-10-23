
from bs_unet import *
from bs_helpers import *


SX = SY = 10
nShipFields = 30



net = bship_unet()
net.eval()

with open('data/battleships_unet.dat', 'rb') as f:
    net.load_state_dict(torch.load(f))

_probabilities = dict()
def _getprobability(s):
    h = GameClass.getHashable(s)
    if not h in _probabilities:
        with torch.no_grad():
            prob = net.predict(encode_x(s.sea, s.det))
        _probabilities[h] = prob
    else:
        prob = _probabilities[h]
    plt.imshow(prob, vmin=0., vmax=1.)
    plt.show()
    return prob
    
    
class GameState:
    def __init__(self):
        self.sea = np.zeros((SX, SY), 'uint8')
        self.det = np.zeros((SX, SY), 'uint8')
        
    def copy(self):
        other = GameState()
        other.sea = self.sea.copy()
        other.det = self.det.copy()
        return other


class GameClass:
    
    numActions = SX*SY
    
    @staticmethod
    def getNextState(s, a, hidden=None):
        s = s.copy()
        if type(hidden) is type(None):
            # no information of ships, must be rolled out
            s.sea[a] = np.random.rand() < _getprobability(s)[a]
            s.det[a] = 1
        else:
            # information of ships is given in array `hidden`
            s.sea[a] = hidden[a]
            s.det[a] = 1
        return s
            
    @staticmethod
    def getValidActions(s):
        return [ (i, j) for i in range(SX) for j in range(SY) if not s.det[i,j] ]
    
    @staticmethod
    def getTurn(s):
        return 1
            
    @staticmethod
    def getEnded(s):
        if np.sum(s.sea * s.det) >= 30:
            return np.sum(s.sea)/np.sum(s.det)
        return None
    
    @staticmethod
    def getHashable(s):
        return s.sea.tobytes() + s.det.tobytes()
    
    @staticmethod
    def mirrorx_state(s):
        s.sea = s.sea[:, ::-1]
        s.det = s.det[:, ::-1]
    @staticmethod
    def mirrory_state(s):
        s.sea = s.sea[::-1]
        s.det = s.det[::-1]
    @staticmethod
    def mirrorx_action(a):
        i, j = a
        return SX-i, j
    @staticmethod
    def mirrory_action(a):
        i, j = a
        return i, SY-j
    
    

def plot_state(s):
    plot_sea(s.sea, s.det)
    
    
    
    
    