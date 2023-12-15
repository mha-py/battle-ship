# Erstellt August 2020
# (c) mha


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import random


SX = 10
NSHIPS = 10


def argmax2d(arr):
    n, m = arr.shape
    ij = arr.argmax()
    i, j = ij//m, ij%m
    return i, j


def create_sea(seed=None):
    'Creates a sea with random ships on it'
    rng = random.Random(seed)
    sea = np.zeros((10,10))
    #for l in [5, 4, 3, 2]: # Länge
    for l in [5,4,3,2]: # Länge
        n = 6-l # Anzahl
        for _ in range(n):
            # Boot mit Länge l platzieren
            while True:
                t = rng.random() < 0.5
                if t: sea = sea.T # Transponieren
                px = rng.randint(0, 10-l)
                py = rng.randint(0, 9)
                if sum(sea[px:px+l,py]) > 0:
                    continue
                sea[px:px+l, py] = 1
                if t: sea = sea.T # Transponieren
                break
    return sea



@njit(cache=True)
def njit_create_sea():
    'Creates a sea with random ships on it'
    sea = np.zeros((10,10))
    for l in [5,4,3,2]: # Länge
        n = 6-l # Anzahl
        for _ in range(n):
            # Boot mit Länge l platzieren
            while True:
                t = np.random.rand() < 0.5
                if t: sea = sea.T # Transponieren
                px = np.random.randint(0, 11-l)  # unterschied zur funktion oben ist wichtig
                py = np.random.randint(0, 10)
                if np.sum(sea[px:px+l,py]) > 0:
                    continue
                sea[px:px+l, py] = 1
                if t: sea = sea.T # Transponieren
                break
    return sea



def create_detection(n=None):
    'Creates a random detected array (for test purposes)'
    if n is None:
        n = np.random.randint(0, 100)
    det = -1
    while np.sum(det) != n:
        det = np.random.rand(10,10) < n/100
    return det


def newrandomstate(t=1.):
    from bs_gameclass import GameState
    s = GameState()
    s.det = np.random.rand(10,10) < np.random.rand()*t
    s.sea = create_sea()
    s.sea *= s.det
    return s



def visualize(sea, detection):
    'Erstellt eine Veranschaulichung, 0 bzw. 4 sind detektiertes Wasser bzw. Schiff, 1 und 2 sind undetektiert.'
    return sea + sea*detection + 1 - ((1-sea)*detection)


def plot_sea(sea, det, ax=None):
    if ax is None: ax = plt.gca()
    #ax.imshow(visualize(sea, det), vmin=-2, cmap='plasma')
    ax.imshow(visualize(sea, det), vmin=-1, vmax=3.15, cmap='cividis')
    ax.axis('off')
    
    
    
def getStartEndPoints(ship):
    T = (ship.sum(0)>0).sum()==1
    if T:
        ship = ship.T
    l = ship.sum(1).argmax()
    i = ship.sum(0).argmax()
    j = i + ship.sum()-1
    if not T:
        return (i, l), (j, l)
    else:
        return (l, i), (l, j)

def plot_ships(ships, det, ax=None, oldversion=False):
    if oldversion: # Achsen wurden vertauscht in neue version
        ships = ships.transpose(2, 0, 1)
    if ax is None: ax = plt.gca()
    background = np.zeros((SX, SX, 3))
    background[:,:] = (0.1, 0.3, 0.8)
    ax.imshow(background, zorder=-50)
     
    for k in range(len(ships)):
        pt1, pt2 = getStartEndPoints(ships[k])
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0.6,0.6,0.6), lw=15, zorder=-20)
    ones = np.ones_like(det).astype(float)*0
    det = np.stack([ones, ones, ones, (1-det)/3], -1)
    ax.imshow(det, cmap='gray')
    ax.axis('off')