# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 01:14:20 2025

@author: Romain
"""

import sys
import math
from collections import deque
import time
import numpy as np

from simulation import Game
import matplotlib.pyplot as plt

game=Game()
plt.ion() 
ITER=0

import random as rd

MOVES =["UP","DOWN","RIGHT","LEFT"]

while not game.is_end():

    n= game.Nplayers
    p = ITER%game.Nplayers
    

    for i in range(n):
        x0, y0, x1, y1 = game.getData(i)


    game.step(MOVES[rd.randint(0,3)],p)
    
    ITER+=1

