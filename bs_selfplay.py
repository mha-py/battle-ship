

from bs_gameclass import *
from tqdm.notebook import tqdm, trange


class TrivialPlayer:
    def findmove(self, s):
        va = GameClass.getValidActions(s)
        len(va)
        return va[np.random.choice(len(va))]
    def clear(self):
        pass
    
    
class UnetPlayer(TrivialPlayer):
    def findmove(self, s):
        prob = net.predict(encode_x(s.sea, s.det))
        prob[s.det > 0] = 0
        ij = prob.argmax()
        i, j = ij//SX, ij%SX
        return i, j
    
    
    
    

states = None
records = None
def selfplay_batched(ai, ngames=1000, verbose=0):
    
    bnum = ai.nparallel
    game_records = []
    
    ai.eta = 0.3

    if verbose>=1:
        pbar = tqdm(total=ngames)
        
    def newstate():
        return GameState(), create_sea()
            
    global states, records
    try:
        if len(states) != bnum:
            states, records = None, None
    except:
        pass
    if isinstance(states, type(None)):
        states = [ newstate() for _ in range(bnum) ]
    if isinstance(records, type(None)):
        records = [ [] for _ in range(bnum) ]
    
    
    completedgames = 0
    while completedgames < ngames:
        
        ####check_new_model(net, 'net_temp.dat')
        ai.clear()
        ####moves = ai.findmove(states, tau=None)
        moves = [ ai.findmove(s) for s, h in states ]
        for b in range(bnum):
            
            ###
            #if b==0:
            #    print_state(states[b])
            ###
            
            a = moves[b]
            records[b] += [ (states[b][0], a) ]
            s, hidden = states[b]
            states[b] = GameClass.getNextState(s, a, hidden), hidden
            
            r = GameClass.getEnded(states[b][0])  # reward at games end
            if r:
                record = [ (s, a, r) for (s, a) in records[b] ]
                game_records += record
                completedgames += 1
                records[b] = []
                states[b] = newstate()
                if verbose>=1:
                    pbar.update(1)
    
    if verbose>=1:
        pbar.close()
        
    return game_records

