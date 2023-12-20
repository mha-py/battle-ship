
import numpy as np
from numba import njit
import random


SX = SY = 10
NSHIPS = 6



def create_sea_ships(shipnum=(0, 0, 4, 3, 2, 1), seed=None): # diese funktion ist anders als die bei unet!!
    'Creates a sea with random ships on it'
    rng = random.Random(seed)
    sea = np.zeros((SX, SX))
    ships = np.zeros((sum(shipnum), SX, SX))
    k = 0
    for l in [5,4,3,2]: # Länge
        n = shipnum[l] # Anzahl
        for _ in range(n):
            # Boot mit Länge l platzieren
            while True:
                t = rng.random() < 0.5
                if t:
                    sea = sea.T # Transponieren
                    ships = ships.transpose(0, 2, 1)
                px = rng.randint(0, SX-l)
                py = rng.randint(0, SX-1)
                if sum(sea[px:px+l,py]) > 0:
                    continue
                sea[px:px+l, py] = 1
                ships[k, px:px+l, py] = 1
                if t:
                    sea = sea.T # Transponieren
                    ships = ships.transpose(0, 2, 1)
                k += 1
                break
    return sea, ships



@njit(cache=True)
def njit_create_sea_ships(shipnum=(0, 0, 4, 3, 2, 1)): # diese funktion ist anders als die bei unet!!
    'Creates a sea with random ships on it'
    sea = np.zeros((SX, SX))
    ships = np.zeros((sum(shipnum), SX, SX))
    k = 0
    for l in [5,4,3,2]: # Länge
        n = shipnum[l] # Anzahl
        for _ in range(n):
            # Boot mit Länge l platzieren
            while True:
                t = np.random.rand() < 0.5
                if t:
                    sea = sea.T # Transponieren
                    ships = ships.transpose(0, 2, 1)
                px = np.random.randint(0, SX+1-l)
                py = np.random.randint(0, SX)
                if sum(sea[px:px+l,py]) > 0:
                    continue
                sea[px:px+l, py] = 1
                ships[k, px:px+l, py] = 1
                if t:
                    sea = sea.T # Transponieren
                    ships = ships.transpose(0, 2, 1)
                k += 1
                break
    return sea, ships






@njit(cache=True)
def findShipNum(nships):
    'Random numbers for the count of ships of length 2 to 5'
    shipnum = np.random.multinomial(nships, 6*[1/6]) # schiffe verteilen auf 6 mögliche Längen
    shipnum[:2] = 0 # Länge null und Länge eins sollen nicht vorkommen
    return shipnum
    

@njit(cache=True)
def sinksShip(ships, det, ij):
    det = det.copy()
    det[ij] = 1
    sn = 0
    for k in range(len(ships)):
        if ships[k][ij]:
            if np.sum(ships[k,:,:] * (1-det)) == 0:
                sn = 1
    return sn

@njit(cache=True)
def play(nmoves):
    'Plays a game and returns the game history and the sea'
    shipnum= findShipNum(NSHIPS)
    sea, ships = njit_create_sea_ships(shipnum)
    det = np.zeros_like(sea)
    moves = []
    hits = []
    sunken = []
    for _ in range(nmoves):
        vms = [ (i,j) for i in range(SX) for j in range(SY) if (1-det[i,j]) ]
        k = np.random.randint(len(vms))
        ij = vms[k]
        det[ij] = 1
        moves.append(ij)
        hits.append(sea[ij])
        sunken.append(sinksShip(ships, det, ij))
    
    # target is 1 if a ship is at ij, and is 2 if a shot at ij sinks a ship
    target = np.zeros((SX, SX))
    for i in range(SX):
        for j in range(SX):
            target[i,j] = sea[i,j]
            if target[i,j]:
                target[i,j] += sinksShip(ships, det, (i,j))
    return list(zip(moves, hits, sunken)), target, shipnum#, det, ships



def encode_shipnum(shipnum):
    'Encodes the number of ships on the sea into a one hot format'
    z = np.zeros((NSHIPS+1)*6)
    for k in range(6):
        i = shipnum[k]
        z[(NSHIPS+1)*k + i] = 1.
    return z
    

def encode(hist, shipnum):
    'Encodes the visible information to an input to the neural network'
    'First data point is the number of ships followed by the coordinates'
    'of the shots and the information of a ship was hit and if it sank'
    def bits(k):
        return [ float((k>>i)&1) for i in range(8) ]
    
    #x = np.zeros((SX**2, 2*SX+2+8))
    NBUFF = 100
    x = np.zeros((1+SX**2, NBUFF))
    x[0, :6*(NSHIPS+1)] = encode_shipnum(shipnum)
    
    for k in range(len(hist)):
        (i,j),r,sn = hist[k]
        x[k+1, i] = 1
        x[k+1, SX+j] = 1
        x[k+1, 2*SX] = r
        x[k+1, 2*SX+1] = sn
        x[k+1, -8:] = np.asarray(bits(k+1))
    return x

