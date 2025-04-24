import random as rd
import numpy as np
import sys
import copy
import time as t 
from collections import defaultdict
from itertools import product, combinations, product
from eternity_puzzle import EternityPuzzle
import heapq

GRAY = 0

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def solve_advanced(puzzle : EternityPuzzle):
    # Seed (pour test)
    r = 69
    rd.seed(r)

    # Initialisation
    solver = Solver(puzzle, t.time())
    solver._randomBorder()
    solver._randomInside()
    
    # Algo principal
    while (t.time() - solver.startTime) < solver.T:
        iter = 0
        while (t.time() - solver.startTime) < solver.T and iter < 5:
            print(f"Iter n : {iter}")
            iter += 1
            solver.best_sol = [None, np.inf]
            
            solver._randomPerturbations()
            solver._borderTabuSearch()
            solver._innerTabuSearch()
            solver._simulatedAnnealing()

            print(f" Dernière meilleure sol : {solver.best_sols[-1][1]}")
        
        solver.board = min(solver.best_sols, key=lambda x: x[1])[0]
        solver.best_sols = []

    # Retourne la meilleure solution
    solver.board = min(solver.best_sols, key=lambda x: x[1])[0]
    s = solver._format()
    return s, puzzle.get_total_n_conflict(s)

class Solver():
    def __init__(self, puzzle: EternityPuzzle, startTime: float):
        self.puzzle = puzzle
        self.n = puzzle.board_size
        self.size = self.n ** 2
        self.startTime = startTime
        self.T = 600
        
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

        
## RECUIT SIMULER

    def _simulatedAnnealing(self):
        print("Simulated Annealing")
        temp = 100
        alpha = 0.99
        value = self._totalConflicts()

        best = [None, np.inf]

        iter = 0
        iter_without_best = 0
        while (t.time() - self.startTime) < self.T and iter_without_best < 500:
            iter += 1 
            iter_without_best += 1
            neighbors = self._neighborsSimulatedAnnealing(iter, value, best[1])
            neighbors.sort(key=lambda x: x[2])

            neighs = rd.choice(neighbors[:10])
            delta = neighs[2] - best[1]

            if delta < 0:
                x1, y1, p1, o1 = neighbors[0][0] 
                x2, y2, p2, o2 = neighbors[0][1]
                self._place_piece(x1, y1, p1, o1)
                self._place_piece(x2, y2, p2, o2)
                
                iter_without_best = 0
                best = [copy.deepcopy(self.board), neighs[2]]
                print(neighs[2])     

            elif delta >= 0 and rd.random() < np.exp(-delta/temp):
                x1, y1, p1, o1 = neighbors[0][0] 
                x2, y2, p2, o2 = neighbors[0][1]
                self._place_piece(x1, y1, p1, o1)
                self._place_piece(x2, y2, p2, o2)

                if neighs[2] < best[1]:
                    best = [copy.deepcopy(self.board), neighs[2]]
                    print(neighs[2])     

            if value == 0:
                break
            
            temp = alpha*temp
            

        self.board = best[0]
        self.best_sols.append([copy.deepcopy(best[0]), best[1]])
        return

    def _neighborsSimulatedAnnealing(self, iter, value, best):
        neighbors = []

        for (x1, y1), (x2, y2) in combinations(self.cells, 2):
            if (t.time() - self.startTime) > self.T:
                return neighbors
            
            if rd.random() > 0.1:
                continue
            
            p1 = self.board[x1][y1]["p"]
            old_o1 = self.board[x1][y1]["o"]
            p2 = self.board[x2][y2]["p"]
            old_o2 = self.board[x2][y2]["o"]

            # if rd.random() > 1:
            #     continue

            if not self._homogeneous(p1, p2):
                continue

            self._place_piece(x1, y1, p2, old_o2)
            self._place_piece(x2, y2, p1, old_o1)
            new_value = self._totalConflicts()
            self._place_piece(x1, y1, p1, old_o1)
            self._place_piece(x2, y2, p2, old_o2)

            neighbors.append([(x1, y1, p2, old_o2), (x2, y2, p1, old_o1), new_value])

            for new_o1 in self.all_rotations[p1]:
                if new_o1 != old_o1:
                    self._place_piece(x1, y1, p1, new_o1)
                    new_value = self._totalConflicts()
                    self._place_piece(x1, y1, p1, old_o1)

                    neighbors.append([(x1, y1, p1, new_o1), (x2, y2, p2, old_o2), new_value])

        return neighbors
    

## INNER TABU SEARCH

    def _randomInside(self):
        print("Random Inside")
        rd.shuffle(self.inner_pieces)
        interior_iter = iter(self.inner_pieces)
        for x, y in self.inner_cells:
            piece = next(interior_iter)
            self._place_piece(x, y, piece, rd.choice(self.all_rotations[piece]))
            
    def _innerTabuSearch(self):
        print("Inner Tabu Search")
        if (t.time() - self.startTime) > self.T:
                return 
        
        tabu_dict1 = {(x1, y1, p1, x2, y2, p2): 0 for (x1, y1), (x2, y2) in combinations(self.cells, 2) for p1, p2 in product(self.pieces, self.pieces)}
        tabu_dict2 = {(x, y, o): 0 for x, y in self.cells for o in self.all_config}
        tabu_dict = {**tabu_dict1, **tabu_dict2}

        value = self._totalConflicts()
        best = [copy.deepcopy(self.board), value]

        k = 100
        iter = 0
        iter_without_best = 0
        while (t.time() - self.startTime) < self.T and iter_without_best < 1000:
            iter += 1 
            iter_without_best += 1 
            neighbors = self._neighborsInner(tabu_dict, iter, best[1])
            neighbors.sort(key=lambda x: x[2])

            new_value = neighbors[0][2]
            type = neighbors[0][3]
            x1, y1, p1, o1 = neighbors[0][0] 
            x2, y2, p2, o2 = neighbors[0][1]

            if type == "swap":
                self._place_piece(x1, y1, p1, o1)
                self._place_piece(x2, y2, p2, o2)
                tabu_dict[(x1, y1, p1, x2, y2, p2)] = iter + k*new_value
                tabu_dict[(x2, y2, p2, x1, y1, p1)] = iter + k*new_value
            
            if type == "rot":
                self._place_piece(x1, y1, p1, o1)
                tabu_dict[(x1, y1, o1)] = iter + k*new_value

            if new_value < best[1]:
                iter_without_best = 0
                best = [copy.deepcopy(self.board), new_value]
                print(new_value)

            if value == 0:
                break
        
        self.board = best[0]
        self.best_sols.append([copy.deepcopy(best[0]), best[1]])
        return
    
    
    def _neighborsInner(self, tabu_dict, iter, best_value):
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

            self._place_piece(x1, y1, p2, old_o2)
            self._place_piece(x2, y2, p1, old_o1)
            new_value = self._totalConflicts()
            self._place_piece(x1, y1, p1, old_o1)
            self._place_piece(x2, y2, p2, old_o2)

            if tabu_dict[(x1, y1, p2, x2, y2, p1)] < iter or new_value < best_value:
                neighbors.append([(x1, y1, p2, old_o2), (x2, y2, p1, old_o1), new_value, "swap"])

            # # Rotation d'une pièce
            for new_o1 in self.all_rotations[p1]:
                if new_o1 != old_o1:
                    self._place_piece(x1, y1, p1, new_o1)
                    new_value = self._totalConflicts()
                    self._place_piece(x1, y1, p1, old_o1)

                    if tabu_dict[(x1, y1, new_o1)] < iter or new_value < best_value:
                        neighbors.append([(x1, y1, p1, new_o1), (x1, y1, p1, new_o1), new_value, "rot"])

        return neighbors


## BORDER TABU SEARCH

    def _randomBorder(self):       
        print("Random Border")
        rd.shuffle(self.corner_pieces)
        rd.shuffle(self.edge_pieces)
        
        for x, y in self.corner_cells:
            for piece in self.corner_pieces:
                if piece in self.used_pieces:
                    continue

                for rot in self.all_rotations[piece]:
                    if self._fits_border(x, y, rot):
                        self._place_piece(x, y, piece, rot)
                        break
                break

        for x, y in self.edge_cells:
            for piece in self.edge_pieces:
                if piece in self.used_pieces:
                    continue

                for rot in self.all_rotations[piece]:
                    if self._fits_border(x, y, rot):
                        self._place_piece(x, y, piece, rot)
                        break
                break
    
    def _borderTabuSearch(self):
        print("Random Border Tabu Search")
        if (t.time() - self.startTime) > self.T:
            return
        
        tabu_dict = {(p1, p2): 0 for p1 in self.border_pieces for p2 in self.border_pieces}
        
        total_border_conflicts = self._totalBorderConflicts()

        k = 30
        iter = 0
        while (t.time() - self.startTime) < self.T:
            iter += 1  
            neighbors = self._neighborsBorder(tabu_dict, iter, total_border_conflicts)
            neighbors.sort(key=lambda x: x[2])
            delta = neighbors[0][2]
            x1, y1, p1, o1 = neighbors[0][0] 
            x2, y2, p2, o2 = neighbors[0][1]
            self._place_piece(x1, y1, p1, o1)
            self._place_piece(x2, y2, p2, o2)
            total_border_conflicts += delta
            tabu_dict[(p1, p2)] = iter + k
            tabu_dict[(p2, p1)] = iter + k
            
            # print(total_border_conflicts)
            if total_border_conflicts == 0:
                break
    
    def _neighborsBorder(self, tabu_dict, iter, old_conflicts):
        neighbors = []

        for (x1, y1), (x2, y2) in combinations(self.border_cells, 2):
            if (t.time() - self.startTime) > self.T:
                return neighbors
            
            p1 = self.board[x1][y1]["p"]
            old_o1 = self.board[x1][y1]["o"]
            p2 = self.board[x2][y2]["p"]
            old_o2 = self.board[x2][y2]["o"]

            if not self._homogeneous(p1, p2) or tabu_dict[(p1, p2)] >= iter:
                continue

            for o in self.all_rotations[p1]:
                if self._fits_border(x2, y2, o):
                    new_o1 = o
                    break
            for o in self.all_rotations[p2]:
                if self._fits_border(x1, y1, o):
                    new_o2 = o
                    break
            
            self._place_piece(x1, y1, p2, new_o2)
            self._place_piece(x2, y2, p1, new_o1)
            new_conflicts = self._totalBorderConflicts()
            self._place_piece(x1, y1, p1, old_o1)
            self._place_piece(x2, y2, p2, old_o2)

            delta = new_conflicts - old_conflicts

            neighbors.append([(x1, y1, p2, new_o2), (x2, y2, p1, new_o1), delta])
    
        return neighbors
    

## PERTURBATIONS

    def _randomPerturbations(self):
        print("Perturbations")
        gamma = 0.5
        number_of_pairs = int(self.size*gamma)
        beta = 0.5
        theta = 0.5

        for _ in range(int(number_of_pairs*beta)):
            if (t.time() - self.startTime) > self.T:
                return 
            
            ## Swap de 2 pièces interieur
            (x1, y1), (x2, y2) = rd.sample(self.inner_cells, 2)
            p1 = self.board[x1][y1]["p"]
            o1 = self.board[x1][y1]["o"]
            p2 = self.board[x2][y2]["p"]
            o2 = self.board[x2][y2]["o"]
            self._place_piece(x1, y1, p2, o2)
            self._place_piece(x2, y2, p1, o1)

        for _ in range(int(number_of_pairs*(1-beta))):
            if (t.time() - self.startTime) > self.T:
                return 
            
            ## Swap de 2 pièces edges
            (x1, y1), (x2, y2) = rd.sample(self.edge_cells, 2)
            p1 = self.board[x1][y1]["p"]
            p2 = self.board[x2][y2]["p"]

            for o in self.all_rotations[p1]:
                if self._fits_border(x2, y2, o):
                    new_o1 = o
                    break
            for o in self.all_rotations[p2]:
                if self._fits_border(x1, y1, o):
                    new_o2 = o
                    break
            
            self._place_piece(x1, y1, p2, new_o2)
            self._place_piece(x2, y2, p1, new_o1)

        ## Swap de 2 coins avec une proba theta
        if rd.random() < theta:
            (x1, y1), (x2, y2) = rd.sample(self.corner_cells, 2)
            p1 = self.board[x1][y1]["p"]
            p2 = self.board[x2][y2]["p"]

            for o in self.all_rotations[p1]:
                if self._fits_border(x2, y2, o):
                    new_o1 = o
                    break
            for o in self.all_rotations[p2]:
                if self._fits_border(x1, y1, o):
                    new_o2 = o
                    break
            
            self._place_piece(x1, y1, p2, new_o2)
            self._place_piece(x2, y2, p1, new_o1)
    

## UTILITAIRE
    
    def _get_conflicts_p(self, x, y, type):
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

    def _totalConflicts(self):
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

    def _totalBorderConflicts(self):
        total_border_conflicts = 0

        for x, y in self.border_cells:
            total_border_conflicts += self._get_conflicts_p(x, y, "border")/2
       
        return total_border_conflicts

    def _homogeneous(self, p1, p2):
        return p1.count(0) == p2.count(0)

    def _fits_border(self, x, y, oriented):
        return (oriented[SOUTH] == GRAY if x == 0 else True) and \
               (oriented[NORTH] == GRAY if x == self.n - 1 else True) and \
               (oriented[WEST] == GRAY if y == 0 else True) and \
               (oriented[EAST] == GRAY if y == self.n - 1 else True)

    def _place_piece(self, x, y, piece, oriented):
        self.board[x][y] = {"p" : piece, "o": oriented}
        self.used_pieces[piece] = oriented

    def _format(self):
        # Construction de la solution en liste plate : ordre "de bas en haut", chaque ligne de gauche à droite.
        flat_solution = []
        for x in range(self.n):
            for y in range(self.n):
                flat_solution.append(self.board[x][y]["o"])
        
        return flat_solution