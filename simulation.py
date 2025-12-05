# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 01:13:54 2025

@author: Romain
"""


import sys
import math
from collections import deque
import time
import numpy as np
import random as rd

import matplotlib.pyplot as plt

WIDTH = 30
HEIGHT = 20
TIME=0.1

MOVES = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}

class Moto:
    def __init__(self, id,x=-1,y=-1,x0=-1,y0=-1):
        self.id = id
        self.x = x
        self.y = y
        self.x0 = x0
        self.y0 = y0
        self.alive = True
        self.parcours=[(x,y)]
class Game:
    
    def __init__(self):
        self.grid=[[0]*WIDTH for _ in range(HEIGHT)]
 
        self.Nplayers=rd.randint(2,4)
 
        Coord=[]
        i=0
        while len(Coord)<self.Nplayers:
            x=rd.randint(0,WIDTH-1)
            y=rd.randint(0,HEIGHT-1)
 
            if (x,y) not in Coord:
                Coord.append(Moto(i,x=x,y=y,x0=x,y0=y))

                self.grid[y][x]=-1
                i+=1
                self.players=Coord

 
    def getData(self,idx):

        player=self.players[idx]
        if player.alive:return player.x0,player.y0,player.x,player.y
        return -1,-1,-1,-1
    
    def draw(self):
        plt.clf()
        plt.imshow(self.grid, cmap="gray")
        plt.pause(0.001)

    def step(self, action,idx):
        
        player=self.players[idx]
        if not player.alive:
            return
        x,y=player.x,player.y

        xm,ym=MOVES[action]

        nx,ny=x+xm,y+ym

        if 0<=nx<WIDTH and 0<=ny<HEIGHT and self.grid[ny][nx]!=-1:
            player.x=nx
            player.y=ny
            self.grid[ny][nx]=-1
            player.parcours.append((nx,ny))
   
        else:
            player.alive=False
            for x,y in player.parcours:
                self.grid[y][x]=0
        
        self.draw()
    
    def is_end(self):
        s=sum(p.alive for p in self.players)
        if s<2: return True
        return False


            
            
            