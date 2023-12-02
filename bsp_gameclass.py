
from bs_unet import *
from bs_helpers import *


SX = SY = 20
NSHIPS = 10



def calcdist(ships):
    dist = np.zeros_like(ships)
    done = np.zeros_like(ships)
    queue = [ (i,j) for i in range(SX) for j in range(SY) if ships[i,j].sum() > 0 ]
    d = 0
    while not np.all(done): ####
        next_queue = []
        while queue:
            i, j = queue.pop()
            # check if indices are inside the map
            if not ((0 <= i < SX) and (0 <= j < SY)):
                continue
            # check if this position was already handled
            if done[i,j]:
                continue
            # handle this position
            dist[i,j] = d
            # add neighbours into the queue
            [ next_queue.append((i+di, j+dj)) for di in [-1, 0, 1] for dj in [-1, 0, 1] ]
            # label this position as done
            done[i,j] = True
            
        queue = next_queue
        d += 1

    return  dist

    
class GameState:
    def __init__(self):
        self.hist = []
        self.det = np.zeros((SX, SY))
        
    def copy(self):
        other = GameState()
        other.hist = self.hist.copy()
        other.det = self.det.copy()
        return other


class GameClass:
    
    numActions = SX*SY
    
    @staticmethod
    def getNextState(s, a, hidden):
        s = s.copy()
        i, j = a
        s.det[i, j] = 1
        
        dist = s.dist[i,j]
        sea = hidden.sum(-1)
        sunken = False
        if dist == 0:
            for l in range(NSHIPS):
                if ships[i,j,l]:
                    sunken = not np.any(s.ships[:,:,l] * (1-s.det))
        s.hist.append((i, j, dist, sunken))

        if sunken:
            s.dist = calcdist(s.ships)
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
        return s
    @staticmethod
    def mirrory_state(s):
        s.sea = s.sea[::-1]
        s.det = s.det[::-1]
        return s
    @staticmethod
    def mirrorx_action(a):
        i, j = a
        return SX-1-i, j
    @staticmethod
    def mirrory_action(a):
        i, j = a
        return i, SY-1-j
    @staticmethod
    def mirror_transpose_state(s):
        s.sea = s.sea.T
        s.det = s.det.T
        return s
    @staticmethod
    def mirror_transpose_action(a):
        i, j = a
        return j, i
    
    

def plot_state(s):
    plot_sea(s.sea, s.det)
    
    
    
    
    