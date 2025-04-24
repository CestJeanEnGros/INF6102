import copy 
import numpy as np
import random as rd
import time as t
from itertools import combinations, product
from math import exp, comb 
from typing import List, Tuple
from eternity_puzzle import EternityPuzzle
from collections import deque

####################
##   CONSTANTES   ##
####################

GRAY = 0
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

####################
##   PARAMETRES   ##
####################

# Affichage
LOG = True

# Border Tabu Search
BORDER_ITER_BAN = 10
BORDER_RATIO_VOISINS = 0.5

# Global Tabu Search
GLOBAL_ITER_BAN = 1000
GLOBAL_RATIO_VOISINS = 0.5
GLOBAL_MAX_ITER = 50000

def solve_advanced(puzzle : EternityPuzzle):
    # Seeding 
    r = 0.18062110343039572
    print(f'Seed: {r}')
    rd.seed(r)      

    # Initialisation
    solver = Solver(puzzle, t.time())

    # Greedy

    
    

    # Retour de la solution
    s = solver.format()
    return s, puzzle.get_total_n_conflict(s)
    

class Solver:
    def __init__(self, puzzle: EternityPuzzle, startTime):
        self.puzzle = puzzle
        self.n = puzzle.board_size
        self.size = self.n ** 2
        self.startTime = startTime
        self.T = 3600
        
        self.pieces = puzzle.piece_list
        self.corner_pieces = [piece for piece in self.pieces if piece.count(0) == 2]
        self.edge_pieces = [piece for piece in self.pieces if piece.count(0) == 1]
        self.border_pieces = [piece for piece in self.pieces if piece.count(0) >= 1]
        self.inner_pieces = [piece for piece in self.pieces if piece.count(0) == 0]

        self.corner_cells = [(0, 0), (self.n - 1, 0), (0, self.n - 1), (self.n - 1, self.n - 1)]
        self.edge_cells = [(x, y) for x in range(self.n) for y in range(self.n) if (x in [0, self.n - 1] or y in [0, self.n - 1]) and (x, y) not in self.corner_cells]
        self.border_cells = [(x, y) for x in range(self.n) for y in range(self.n) if x == 0 or x == self.n - 1 or y == 0 or y == self.n - 1]
        self.inner_cells = [(x, y) for x in range(1, self.n - 1) for y in range(1, self.n - 1)]
        self.cells = [(x, y) for x in range(self.n) for y in range(self.n )]

        self.used_pieces = {}
        self.all_rotations = {p: puzzle.generate_rotation(p) for p in self.pieces}
        self.all_config = set([self.all_rotations[p][i] for p in self.pieces for i in range(4)])
        self.board = [[None for _ in range(self.n)] for _ in range(self.n)]

        self.best_sol = [None, np.inf]
        self.best_sols = []

    ################
    ##   Greedy   ##
    ################

    def build_greedy(self):
        rd.shuffle(self.corner_pieces)
        rd.shuffle(self.edge_pieces)
        rd.shuffle(self.inner_pieces)

        for s in range(2*self.n - 1):  # s = i + j
            for x in range(s, -1, -1):  # i descend
                y = s - x
                if x < self.n and y < self.n:
                    if self.board[x][y] == None:
                        p, o, _ = self.best_piece_greedy(x, y)
                        self.place_piece(x, y, p, o)   
    
    def best_piece_greedy(self, x, y):
        pieces = self.valid_pieces(x, y)

        best = [None, None, np.inf]

        for p in pieces:
            if p in self.used_pieces:
                continue
            for o in self.all_rotations[p]:
                value = self.predict_conflict_greedy(x, y, o)
                if value < best[2]:
                    best = [p, o, value]
                if value == 0:
                    return best
                
        return best

    def valid_pieces(self, x, y):
        if (x, y) in self.corner_cells:
            return self.corner_pieces
        elif (x, y) in self.edge_cells:
            return self.edge_pieces
        else:
            return self.inner_pieces
    
    def predict_conflict_greedy(self, x, y, o):
        nb_conflict = 0

        if x-1 < 0: south_col = GRAY
        elif self.board[x-1][y] == None: south_col = None
        else: south_col = self.board[x-1][y]["o"][NORTH]

        if x+1 >= self.n: north_col = GRAY
        elif self.board[x+1][y] == None: north_col = None
        else: north_col = self.board[x+1][y]["o"][SOUTH]
        
        if y-1 < 0: west_col = GRAY
        elif self.board[x][y-1] == None: west_col = None
        else: west_col = self.board[x][y-1]["o"][EAST]

        if y+1 >= self.n: east_col = GRAY
        elif self.board[x][y+1] == None: east_col = None
        else: east_col = self.board[x][y+1]["o"][WEST]

        if south_col != None and south_col != o[SOUTH]: nb_conflict += 1
        if north_col != None and north_col != o[NORTH]: nb_conflict += 1
        if west_col != None and west_col != o[WEST]: nb_conflict += 1
        if east_col != None and east_col != o[EAST]: nb_conflict += 1
        
        return nb_conflict

    ########################
    ## Global Tabu Search ##
    ########################

    def globalTabuSearch(self):
        print("Global Tabu Search")
        if (t.time() - self.startTime) > self.T:
                return 
        
        tabu_dict = deque(maxlen=GLOBAL_ITER_BAN)

        value = self.totalConflicts()
        best = [copy.deepcopy(self.board), value]

        iter = 0
        iter_without_best = 0
        while (t.time() - self.startTime) < self.T and iter_without_best < GLOBAL_MAX_ITER:
            # print(tabu_dict)
            iter += 1 
            iter_without_best += 1 
            if iter_without_best % 1000 == 0: print(f" Iter withotu best : {iter_without_best}")
            if iter % 100 == 0: print(f" Iter : {iter}")
            neighbors = self.neighborsGlobal(tabu_dict, iter, best[1])
            neighbors.sort(key=lambda x: x[2])

            if len(neighbors) == 0:
                print("no neigh")
                continue

            neigh = neighbors[0]
            value += neigh[2]
            
            
            type = neighbors[0][3]
            x1, y1, p1, o1 = neigh[0] 
            x2, y2, p2, o2 = neigh[1]


            if type == "swap":
                self.place_piece(x1, y1, p1, o1)
                self.place_piece(x2, y2, p2, o2)
                tabu_dict.append(copy.deepcopy(self.board))
            
            if type == "rot":
                self.place_piece(x1, y1, p1, o1)
                tabu_dict.append(copy.deepcopy(self.board))


            if value != self.puzzle.get_total_n_conflict(self.format()):
                print(value, self.puzzle.get_total_n_conflict(self.format()))
                print(type)


            if value < best[1]:
                iter_without_best = 0
                best = [copy.deepcopy(self.board), value]
                print(value)

            if value == 0:
                break
        
        self.board = best[0]
        self.best_sols.append([copy.deepcopy(best[0]), best[1]])
        return
    
    
    def neighborsGlobal(self, tabu_dict, iter, best_value):
        neighbors = []

        for (x1, y1), (x2, y2) in combinations(self.inner_cells, 2):
            if (t.time() - self.startTime) > self.T:
                return neighbors
            
            # Swap 2 sans rotation
            if rd.random() <= GLOBAL_RATIO_VOISINS:
                p1 = self.board[x1][y1]["p"]
                old_o1 = self.board[x1][y1]["o"]
                p2 = self.board[x2][y2]["p"]
                old_o2 = self.board[x2][y2]["o"] 

                if self.homogeneous(p1, p2):

                    old_conflicts = self.predict_conflict_greedy(x1, y1, old_o1) + self.predict_conflict_greedy(x2, y2, old_o2)
                    if self.correction_double_conflicts(x1, y1, old_o1, x2, y2, old_o2): old_conflicts -= 1

                    self.place_piece(x1, y1, p2, old_o2)
                    self.place_piece(x2, y2, p1, old_o1)

                    new_conflicts = self.predict_conflict_greedy(x1, y1, old_o2) + self.predict_conflict_greedy(x2, y2, old_o1)
                    if self.correction_double_conflicts(x1, y1, old_o2, x2, y2, old_o1): new_conflicts -= 1
                    
                    if self.board not in tabu_dict:
                        self.place_piece(x1, y1, p1, old_o1)
                        self.place_piece(x2, y2, p2, old_o2)

                        delta = new_conflicts - old_conflicts
                        
                        neighbors.append([(x1, y1, p2, old_o2), (x2, y2, p1, old_o1), delta, "swap"])
                    else:
                        self.place_piece(x1, y1, p1, old_o1)
                        self.place_piece(x2, y2, p2, old_o2)
            
            ## Rotation d'une pièce           
            if rd.random() < GLOBAL_RATIO_VOISINS:
                p1 = self.board[x1][y1]["p"]
                old_o1 = self.board[x1][y1]["o"]
                
                for new_o1 in self.all_rotations[p1]:
                    if new_o1 != old_o1:
                        old_conflicts = self.predict_conflict_greedy(x1, y1, old_o1)
                        self.place_piece(x1, y1, p1, new_o1)

                        if self.board not in tabu_dict:

                            new_conflicts = self.predict_conflict_greedy(x1, y1, new_o1)
                            self.place_piece(x1, y1, p1, old_o1)

                            delta = new_conflicts - old_conflicts

                            neighbors.append([(x1, y1, p1, new_o1), (x1, y1, p1, new_o1), delta, "rot"])
                        else:
                            self.place_piece(x1, y1, p1, old_o1)

        return neighbors
    
    #######################
    ## Inner Tabu Search ##
    #######################

    def innerTabuSearch(self):
        print("Inner Tabu Search")
        if (t.time() - self.startTime) > self.T:
                return 
        
        tabu_dict1 = {(o1, o2): 0 for o1, o2 in product(self.all_config, self.all_config)}
        tabu_dict2 = {o: 0 for o in self.all_config}
        tabu_dict = {**tabu_dict1, **tabu_dict2}

        value = self.totalConflicts()
        best = [copy.deepcopy(self.board), value]

        k = 1000
        iter = 0
        iter_without_best = 0
        while (t.time() - self.startTime) < self.T and iter_without_best < 1000:
            iter += 1 
            iter_without_best += 1 
            neighbors = self.neighborsInner(tabu_dict, iter, best[1])
            neighbors.sort(key=lambda x: x[2])

            new_value = neighbors[0][2]
            type = neighbors[0][3]
            x1, y1, p1, o1 = neighbors[0][0] 
            x2, y2, p2, o2 = neighbors[0][1]

            if type == "swap":
                self.place_piece(x1, y1, p1, o1)
                self.place_piece(x2, y2, p2, o2)
                tabu_dict[(o1, o2)] = iter + k
                tabu_dict[(o1, o2)] = iter + k
            
            if type == "rot":
                self.place_piece(x1, y1, p1, o1)
                tabu_dict[o1] = iter + k*new_value

            if new_value < best[1]:
                iter_without_best = 0
                best = [copy.deepcopy(self.board), new_value]
                print(new_value)

            if value == 0:
                break
        
        self.board = best[0]
        self.best_sols.append([copy.deepcopy(best[0]), best[1]])
        return
    
    
    def neighborsInner(self, tabu_dict, iter, best_value):
        neighbors = []

        for (x1, y1), (x2, y2) in combinations(self.inner_cells, 2):
            if (t.time() - self.startTime) > self.T:
                return neighbors
            
            if rd.random() > 0.1:
                continue
            
            # Echange de deux pièces sans rotations
            p1 = self.board[x1][y1]["p"]
            old_o1 = self.board[x1][y1]["o"]
            p2 = self.board[x2][y2]["p"]
            old_o2 = self.board[x2][y2]["o"]           

            self.place_piece(x1, y1, p2, old_o2)
            self.place_piece(x2, y2, p1, old_o1)
            new_value = self.totalConflicts()
            self.place_piece(x1, y1, p1, old_o1)
            self.place_piece(x2, y2, p2, old_o2)

            if tabu_dict[(old_o2, old_o2)] < iter or new_value < best_value:
                neighbors.append([(x1, y1, p2, old_o2), (x2, y2, p1, old_o1), new_value, "swap"])

            # # Rotation d'une pièce
            for new_o1 in self.all_rotations[p1]:
                if new_o1 != old_o1:
                    self.place_piece(x1, y1, p1, new_o1)
                    new_value = self.totalConflicts()
                    self.place_piece(x1, y1, p1, old_o1)

                    if tabu_dict[new_o1] < iter or new_value < best_value:
                        neighbors.append([(x1, y1, p1, new_o1), (x1, y1, p1, new_o1), new_value, "rot"])

        return neighbors
    
    ########################
    ## Border Tabu Search ##
    ########################

    def borderTabuSearch(self):
        print("Random Border Tabu Search")
        if (t.time() - self.startTime) > self.T:
            return
        
        tabu_dict = {(x1, y1, x2, y2): 0 for (x1, y1), (x2, y2) in combinations(self.border_cells, 2)}
        total_border_conflicts = self.totalBorderConflicts()
        
        iter = 0
        while (t.time() - self.startTime) < self.T:
            iter += 1  
            neighbors = self._neighborsBorder(tabu_dict, iter)
            neighbors.sort(key=lambda x: x[2])

            neigh = neighbors[0]
            delta = neigh[2]
            x1, y1, p1, o1 = neigh[0] 
            x2, y2, p2, o2 = neigh[1]

            self.place_piece(x1, y1, p1, o1)
            self.place_piece(x2, y2, p2, o2)

            total_border_conflicts += delta
            tabu_dict[(x1, y1, x2, y2)] = iter + BORDER_ITER_BAN
            
            if total_border_conflicts == 0:
                break
    
    def _neighborsBorder(self, tabu_dict, iter):
        neighbors = []

        for (x1, y1), (x2, y2) in combinations(self.border_cells, 2):
            if (t.time() - self.startTime) > self.T:
                return neighbors

            if tabu_dict[(x1, y1, x2, y2)] >= iter or rd.random() > BORDER_RATIO_VOISINS:
                continue

            p1 = self.board[x1][y1]["p"]
            old_o1 = self.board[x1][y1]["o"]
            p2 = self.board[x2][y2]["p"]
            old_o2 = self.board[x2][y2]["o"]

            if not self.homogeneous(p1, p2):
                continue

            for o in self.all_rotations[p1]:
                if self.fits_border(x2, y2, o):
                    new_o1 = o
                    break
            for o in self.all_rotations[p2]:
                if self.fits_border(x1, y1, o):
                    new_o2 = o
                    break
            
            old_conflicts = self.get_conflicts_p(x1, y1, "border") + self.get_conflicts_p(x2, y2, "border")
            if self.correction_double_conflicts(x1, y1, old_o1, x2, y2, old_o2): old_conflicts -= 1

            self.place_piece(x1, y1, p2, new_o2)
            self.place_piece(x2, y2, p1, new_o1)

            new_conflicts = self.get_conflicts_p(x1, y1, "border") + self.get_conflicts_p(x2, y2, "border")
            if self.correction_double_conflicts(x1, y1, new_o2, x2, y2, new_o1): new_conflicts -= 1
            
            self.place_piece(x1, y1, p1, old_o1)
            self.place_piece(x2, y2, p2, old_o2)

            delta = new_conflicts - old_conflicts

            neighbors.append([(x1, y1, p2, new_o2), (x2, y2, p1, new_o1), delta])
    
        return neighbors
    
    ############
    ## Random ##
    ############

    def randomBorder(self):       
        print("Random Border")
        rd.shuffle(self.corner_pieces)
        rd.shuffle(self.edge_pieces)
        
        for x, y in self.corner_cells:
            for piece in self.corner_pieces:
                if piece in self.used_pieces:
                    continue

                for rot in self.all_rotations[piece]:
                    if self.fits_border(x, y, rot):
                        self.place_piece(x, y, piece, rot)
                        break
                break

        for x, y in self.edge_cells:
            for piece in self.edge_pieces:
                if piece in self.used_pieces:
                    continue

                for rot in self.all_rotations[piece]:
                    if self.fits_border(x, y, rot):
                        self.place_piece(x, y, piece, rot)
                        break
                break

    def randomInside(self):
        print("Random Inside")
        rd.shuffle(self.inner_pieces)
        interior_iter = iter(self.inner_pieces)
        for x, y in self.inner_cells:
            piece = next(interior_iter)
            self.place_piece(x, y, piece, rd.choice(self.all_rotations[piece]))


    #################
    ## Utilitaires ##
    #################

    def correction_double_conflicts(self, x1, y1, old_o1, x2, y2, old_o2):
        if x1 == x2:
            if y1 + 1 == y2:
                if old_o1[EAST] != old_o2[WEST]: return True
            elif y2 + 1 == y1:
                if old_o1[WEST] != old_o2[EAST]: return True
        elif y1 == y2:
            if x1 + 1 == x2:
                if old_o1[NORTH] != old_o2[SOUTH]: return True
            elif x2 + 1 == x1:
                if old_o1[SOUTH] != old_o2[NORTH]: return True
        
        return False

    def place_piece(self, x, y, piece, oriented):
        self.board[x][y] = {"p" : piece, "o": oriented}
        self.used_pieces[piece] = oriented

    def format(self):
        # Construction de la solution en liste plate : ordre "de bas en haut", chaque ligne de gauche à droite.
        flat_solution = []
        for x in range(self.n):
            for y in range(self.n):
                flat_solution.append(self.board[x][y]["o"])
        
        return flat_solution
    
    def get_conflicts_p(self, x, y, type):
        conflicts_of_p = 0
        o = self.board[x][y]["o"]

        if type == "border":
            if x == 0 or x == self.n - 1:
                if x == 0 and o[SOUTH] != GRAY:
                    conflicts_of_p += 1
                if x == self.n - 1 and o[NORTH] != GRAY:
                    conflicts_of_p += 1
                if y-1 >= 0 and self.board[x][y-1]["o"][EAST] != o[WEST]:
                    conflicts_of_p += 1
                if y+1 < self.n and self.board[x][y+1]["o"][WEST] != o[EAST]:
                    conflicts_of_p += 1

            if y == 0 or y == self.n - 1:
                if y == 0 and o[WEST] != GRAY:
                    conflicts_of_p += 1
                if y == self.n - 1 and o[EAST] != GRAY:
                    conflicts_of_p += 1
                if x-1 >= 0 and self.board[x-1][y]["o"][NORTH] != o[SOUTH]:
                    conflicts_of_p += 1
                if x+1 < self.n and self.board[x+1][y]["o"][SOUTH] != o[NORTH]:
                    conflicts_of_p += 1
    
        return conflicts_of_p

    def totalConflicts(self):
        total_conflict = 0

        for (x, y) in self.inner_cells:

            if self.board[x][y]["o"][WEST] != self.board[x][y-1]["o"][EAST]:
                total_conflict += 1

            if y == self.n - 2 and self.board[x][y]["o"][EAST] != self.board[x][y+1]["o"][WEST]:
                total_conflict += 1

            if self.board[x][y]["o"][SOUTH] != self.board[x-1][y]["o"][NORTH]:
                total_conflict += 1

            if x == self.n - 2 and self.board[x][y]["o"][NORTH] != self.board[x+1][y]["o"][SOUTH]:
                total_conflict += 1


        return total_conflict

    def totalBorderConflicts(self):
        total_border_conflicts = 0

        for x, y in self.border_cells:
            total_border_conflicts += self.get_conflicts_p(x, y, "border")/2
       
        return total_border_conflicts

    def homogeneous(self, p1, p2):
        return p1.count(0) == p2.count(0)

    def fits_border(self, x, y, oriented):
        return (oriented[SOUTH] == GRAY if x == 0 else True) and \
               (oriented[NORTH] == GRAY if x == self.n - 1 else True) and \
               (oriented[WEST] == GRAY if y == 0 else True) and \
               (oriented[EAST] == GRAY if y == self.n - 1 else True)

    def hash_rep(self):
        hash_rep = (self.board[x][y]["o"] for x, y in self.cells)
        return hash_rep
        