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
# --- CONFIGURATION & CONSTANTES ---
WIDTH = 30
HEIGHT = 20
EMPTY = 0
WALL = -1

# Directions possibles
MOVES = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}
C_PARAM = 1.0 / math.sqrt(2)


class Player:
    def __init__(self, id,x=-1,y=-1,parcours=[]):
        self.id = id
        self.x = x
        self.y = y
        self.alive = True
        self.parcours=parcours
    
    def copy(self):
        return Player(self.id,self.x,self.y,self.parcours.copy())

def is_valid(x, y, grid):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT and grid[y][x] == EMPTY

class GameState:
    def __init__(self, grid, players_data, current_player_id, current_turn):
        self.grid = [row[:] for row in grid]
        self.players =players_data.copy()

        self.current_player_id = current_player_id
        self.current_turn = current_turn

    def get_moves(self):
        p = self.players[self.current_player_id]
        if not p.alive:
            return []
        possible_moves = []
        for direction in MOVES:
            dx, dy = MOVES[direction]
            nx, ny = p.x + dx, p.y + dy
            #print(nx,ny,direction,p.x,p.y, file=sys.stderr)
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and self.grid[ny][nx] == EMPTY:
                possible_moves.append(direction)
        return possible_moves

    def apply_move(self, direction):
        p = self.players[self.current_player_id]
        if not p.alive:
            next_state = GameState(self.grid, self.players, (self.current_player_id + 1) % len(self.players), self.current_turn)
            return next_state

        dx, dy = MOVES[direction]
        nx, ny = p.x + dx, p.y + dy
        
        new_players = [Player(p.id) for p in self.players]
        for i, p_data in enumerate(self.players):
            new_players[i].x = p_data.x
            new_players[i].y = p_data.y
            new_players[i].alive = p_data.alive
            new_players[i].parcours = p_data.parcours[:]

        p_new = new_players[self.current_player_id]
        
        # Le déplacement est validé par get_moves, on procède à la mise à jour
        new_grid = [row[:] for row in self.grid]
        new_grid[ny][nx] = WALL
        p_new.x, p_new.y = nx, ny
        p_new.parcours.append((nx, ny))

        next_player_id = (self.current_player_id + 1) % len(self.players)
        
        next_state = GameState(new_grid, new_players, next_player_id, self.current_turn + 1)
        return next_state

    def is_terminal(self):
        alive_count = sum(1 for p in self.players if p.alive)
        return alive_count <= 1

    def get_winner(self):
        for p in self.players:
            if p.alive:
                return p.id
        return -1 # Match nul ou tous morts
class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.score = 0      # Q
        self.visits = 0    # N
        self.children = []
        self.untried_moves = state.get_moves()

    def uct_select(self):
        best_child = None
        best_score = -float('inf')
        
        N_parent = self.visits
        
        for child in self.children:
            Q_child = child.score
            N_child = child.visits
            
            # Équilibre entre Exploitation (Q/N) et Exploration (sqrt(ln N_p / N_i))
            exploitation = Q_child / N_child
            exploration = C_PARAM * math.sqrt(math.log(N_parent) / N_child)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.apply_move(move)
        child_node = Node(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.score += result  # result est 1 pour victoire, 0 pour défaite
        if self.parent:
            self.parent.backpropagate(1 - result) # Résultat inversé pour le parent (adversaire)

decision={"LEFT":("UP","DOWN","LEFT"),"UP":("RIGHT","LEFT","UP"),"RIGHT":("DOWN","UP","RIGHT"),"DOWN":("LEFT","RIGHT","DOWN"),"":("LEFT","RIGHT","DOWN","UP")}


def MCTS(Players,grid,tours,p):
    rstate=GameState(grid,Players,p,tours)
    root=Node(rstate)
    start_time = time.time()
    """
    for move in root.state.get_moves():
        nstate=root.state.apply_move(move)
        child_node = Node(nstate, parent=root, move=move)
        root.children.append(child_node)
        score=calculate_voronoi_score(nstate.grid, p, nstate.players)
        child_node.backpropagate(score)
    """
    #print(f"Moves evaluated: ", file=sys.stderr)
    while time.time() - start_time < 0.09:
        node = root
        
        # 1. Selection
        while node.untried_moves == [] and node.children != [] and not node.state.is_terminal():
            node = node.uct_select()
        
        # 2. Expansion
        if node.untried_moves != [] and not node.state.is_terminal():
            node = node.expand()

        # 3. Simulation (Rollout)
        ngrid=node.state.grid
        nPlay=node.state.players
        pid=node.state.current_player_id
        score=calculate_voronoi_score(ngrid, pid, nPlay)
        
        # 4. Backpropagation
        node.backpropagate(score)
    #print(f"Moves evaluqvqs", file=sys.stderr)
    
    
    # 1. Selection



    return  max(root.children, key=lambda x:x.score/x.visits if x.visits else 0)
# --- ALGORITHME DE VORONOÏ (MULTI-SOURCE BFS) ---
# Cette fonction calcule combien de cases chaque joueur peut atteindre AVANT les autres.
def calculate_voronoi_score(grid, my_player_index, players):
    owner_grid = [[-1 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    queue = deque()
    
    active_players = []
    for i, p in enumerate(players):
        if p.alive:
            queue.append((p.x, p.y, i))
            owner_grid[p.y][p.x] = i
            active_players.append(i)

    while queue:
        cx, cy, owner_id = queue.popleft()
        
        for move in MOVES.values():
            nx, ny = cx + move[0], cy + move[1]
            
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and grid[ny][nx] == EMPTY:
                if owner_grid[ny][nx] == -1:
                    owner_grid[ny][nx] = owner_id
                    queue.append((nx, ny, owner_id))
    
    scores = {pid: 0 for pid in active_players}
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if owner_grid[y][x] != -1:
                scores[owner_grid[y][x]] += 1
    
    denom=0.001
    for pid in active_players : denom+= scores[pid]
                
    return np.log(0.001+scores.get(my_player_index, 0)**2 /denom)


grid = [[EMPTY for _ in range(WIDTH)] for _ in range(HEIGHT)]
players = [Player(i) for i in range(4)]
first_turn = True
tours=-1
last_move=""
# --- BOUCLE DE JEU ---
for i in range(100):
    tours+=1

    N= game.Nplayers
    P = i%game.Nplayers
    

    for i in range(N):
        x0, y0, x1, y1 = game.getData(P)

        if first_turn:
            players[i].x = x0
            players[i].y = y0
            grid[y0][x0] = WALL # Marquer le point de départ
            players[i].parcours.append((x0,y0))
        
        # Gestion d'un joueur mort
        if x1 == -1:
            if players[i].alive:

                players[i].alive = False
                for x,y in players[i].parcours:grid[y][x]=EMPTY
        else:
            players[i].x = x1
            players[i].y = y1
            players[i].alive = True
            grid[y1][x1] = WALL # Marquer la position actuelle comme mur
            players[i].parcours.append((x1,y1))
    
    first_turn = False
    """
    # --- PRISE DE DÉCISION ---
    best_move = "LEFT" # Valeur par défaut
    max_score = -1
    
    me = players[P]
    
    # On teste les 4 directions possibles
    valid_moves = []
    for direction  in decision[last_move]:
        dx, dy=MOVES[direction]
        nx, ny = me.x + dx, me.y + dy
        
        if is_valid(nx, ny, grid):
            # SIMULATION : On joue le coup temporairement
            grid[ny][nx] = WALL # On pose un mur
            me.x, me.y = nx, ny # On déplace le joueur
            
            # ÉVALUATION : On lance le Voronoï
            score = calculate_voronoi_score(grid, P, players)
            
            # ANNULATION : On remet comme avant (Backtracking)
            me.x, me.y = nx - dx, ny - dy
            grid[ny][nx] = EMPTY
            
            valid_moves.append((score, direction))
    

    if valid_moves:
        valid_moves.sort(key=lambda x: x[0], reverse=True)
        best_move = valid_moves[0][1]
        
        # Debug
        print(f"Moves evaluated: {valid_moves}", file=sys.stderr)
    else:
        print("No moves valid, I die.", file=sys.stderr)
        best_move = "UP" 
    last_move=best_move
    """
    node=MCTS(players,grid,tours,P)
    game.step(node.move,P)

