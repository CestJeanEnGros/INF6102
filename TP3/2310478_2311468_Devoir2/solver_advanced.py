from utils import Instance, Solution
from typing import Iterable
import random as rd
import numpy as np
import time as t
import copy
from math import exp
from itertools import combinations, product
from colorama import Fore, Style, init 
from collections import deque
import matplotlib.pyplot as plt

###################################################################################################################
###########                                       Configuration                                         ########### 
###################################################################################################################

LOG = True
SEED = True 
SEED_VALUE = 42

INSTANCES = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "X": 6}
LOWER_BOUND = [1471, 10340, 25076, 18098, 16127, 18289, 0]
UPPER_BOUND = [2270, 17309, 38477, 27348, 22233, 24448, 100000]

''' 
ATENTION : Il faut indiqué la lettre de l'instance ci-dessous et le temps accordé pour que l'algorithme fonctionne correctement. Pour l'instance caché, laissez X.
'''
INSTANCE = "X"
TEMPS_MAX = 600

START_TIME = 0

###################################################################################################################
###########                                     HyperParametres                                         ########### 
###################################################################################################################

NEIGH_RATIO = 0.1
TABU_LENGHT = 50

###################################################################################################################
###########                                          Main Aglo                                          ########### 
###################################################################################################################


def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """
    global START_TIME
    START_TIME = t.time()

    if SEED:
        print(f'Seed fixé: {SEED_VALUE}')
        rd.seed(SEED_VALUE)
    else:
        seed = rd.random()
        print(f'Seed random: {seed}')
        rd.seed(seed)

    solver = Solver(instance)
        

    # Solution itéré + tabu + restart
    solver.initial_iterated_search()
    solver.tabu_search()

    # Décommenté pour afficher un graphe d'évolution d ucout en fonction du nombre d'itération
    # solver.plot_score_per_iter()
    
    return Solution(solver.solution)


###################################################################################################################
###########                                            Solver                                           ########### 
###################################################################################################################


class Solver:
    def __init__(self, instance: Instance):
        # Parametres de l'instance
        self.instance = instance
        self.J = self.instance.J 
        self.C = self.instance.C 
        self.H = self.instance.H
        self.start_time = t.time()

        # Etat de la solution
        self.solution = []
        self.cost = None
        self.transition_cost = 0 
        self.storage_cost = 0 

        # Structure de données utiles
        self.total_order = {c: {j: sum(self.instance.order(j_prime, c) for j_prime in range(j + 1)) for j in range(self.J)} for c in range(self.C) } 
        self.total_prod = {c: {j: 0 for j in range(self.J)} for c in range(-1, self.C)} 

        self.saved_sol = deque([[None, None, np.inf] for _ in range(10)], maxlen=10)

        # Attributs pour tracer la courbe d'evolution du score en fonction du nombre d'iterations
        self.iterations = []
        self.score_value = [] 


    ###################################################################################################################
    ###########                                      Solution Initiale                                      ########### 
    ###################################################################################################################

    def initial_iterated_search(self):
        '''
        Renvoie une solution initiale itérée
        '''
        self.solution = [-1 for _ in range(self.J)]

        for j in range(self.J):
            for c in range(self.C):
                if self.instance.order(j,c):
                    possible_days = [i for i, v in enumerate(self.solution[:min(j+1, self.J)]) if v == -1]
                    day = rd.choice(possible_days)
                    self.solution[day] = c
                    for p in range(day, self.J):
                        self.total_prod[c][p] += 1
                    
            
            self.cost = self.instance.solution_cost(Solution(self.solution))
            self.transition_cost = self.instance.solution_transition_cost(Solution(self.solution))
            self.storage_cost = self.instance.solution_storage_cost(Solution(self.solution))

            iter = 0
            ratio = 0.1
            while True:
                iter += 1
                neighbors = self.neighbors_2_swap(iter, ratio=ratio)
                neighbors.sort(key=lambda x: x[2])         

                if len(neighbors) == 0:
                    break                 

                neigh = neighbors[0] 

                if neigh[1] < self.cost:
                    self.do_swap(neigh)
                else:
                    break

    def initial_sol(self):
        '''
        Renvoie une solution initiale naive
        '''
        sol = [-1 for _ in range(self.J)]
        pos = 0

        for j in range(self.J):
            for c in range(self.C):
                if self.instance.order(j,c):
                    sol[pos] = c
                    for p in range(pos, self.J):
                        self.total_prod[c][p] += 1

                    pos += 1

        self.solution = sol
        self.cost = self.instance.solution_cost(Solution(self.solution))
        self.transition_cost = self.instance.solution_transition_cost(Solution(self.solution))
        self.storage_cost = self.instance.solution_storage_cost(Solution(self.solution))


    def random_initial_sol(self):
        '''
        Renvoie une solution initiale aléatorie mais valide
        '''
        sol = [-1 for _ in range(self.J)]

        for j in range(self.J):
            for c in range(self.C):
                if self.instance.order(j,c):
                    possible_days = [i for i, v in enumerate(sol[:min(j+1, len(sol))]) if v == -1]
                    day = rd.choice(possible_days)
                    sol[day] = c
                    for p in range(day, self.J):
                        self.total_prod[c][p] += 1
                    
        
        self.solution = sol
        self.cost = self.instance.solution_cost(Solution(self.solution))
        self.transition_cost = self.instance.solution_transition_cost(Solution(self.solution))
        self.storage_cost = self.instance.solution_storage_cost(Solution(self.solution))


    ###################################################################################################################
    ###########                                        Tabu Search                                          ########### 
    ###################################################################################################################


    def tabu_search(self):
        # Parametres de la Tabu et Initialisation
        tabu_dict = {(-k, i, j, c_i, c_j): 0 for k in range(1,3) for (i, j) in combinations(range(self.J), 2) for (c_i, c_j) in product(range(-1, self.C), range(-1, self.C))}

        best_cost = self.cost
        best_overall = [None, self.cost]
        first_save = True

        k = TABU_LENGHT
        ratio = NEIGH_RATIO

        print(f"Valeur Initial : {best_cost}")

        # Tabu Search
        iter = 0
        iter_without_best = 0
        while (t.time() - START_TIME) < TEMPS_MAX:
            
            iter += 1
            iter_without_best += 1

            neighbors = self.neighbors_2_swap(iter, ratio, tabu_dict, best_cost)
            neighbors.sort(key=lambda x: x[2])

            if len(neighbors) == 0:
                print("no more neigh")
                continue

            neigh = neighbors[0]                
            
            if iter % 1000 == 0: print(f"Iter : {iter, neigh[1]}")

            # Mouvement + mise à jour du tabu_dict
            self.do_swap(neigh)
            tabu_dict[neigh[0]] = iter + k           

            # Mise à jour de la meilleure solution locale
            if neigh[1] < best_cost:
                iter_without_best = 0
                print(f"New best : {neigh[1], iter}")
                self.iterations.append(iter)
                self.score_value.append(neigh[1])
                best_cost = neigh[1]

            # Mise à jour des 10 meilleures solutions globales
            if neigh[1] < best_overall[1]:
                print(f"New best overall : {neigh[1], iter}")
                best_overall = [copy.copy(self.solution), neigh[1]]
                print(f"Save sol mode {first_save}")
                if first_save:
                    self.saved_sol = sorted(self.saved_sol, key=lambda x: x[2], reverse=True)
                    self.saved_sol.append([copy.copy(self.solution), copy.deepcopy(self.total_prod), self.cost])
                    first_save= False
                else:
                    self.saved_sol[-1] = [copy.copy(self.solution), copy.deepcopy(self.total_prod), self.cost]
                

            # Mécaisme de Restart
            if iter_without_best > 10000:
                print("Restart")
                iter_without_best = 0
                restart_sol = copy.deepcopy(rd.choice(list(filter(lambda x: x[1] is not None, self.saved_sol))))
                
                self.solution = restart_sol[0]
                self.total_prod = restart_sol[1]
                self.cost = self.instance.solution_cost(Solution(self.solution))
                self.transition_cost = self.instance.solution_transition_cost(Solution(self.solution))
                self.storage_cost = self.instance.solution_storage_cost(Solution(self.solution))

                best_cost = self.cost
                iter += 50

                first_save = True
                
        self.solution = best_overall[0]
        return best_overall

    
    def neighbors_2_swap(self, iter, ratio, tabu_dict=None, best_cost=None):
        '''
        Renvoie le voisinages
        '''
        neighbors = []

        for (i, j) in combinations(range(self.J), 2):
            if (t.time() - START_TIME) > TEMPS_MAX:
                return neighbors
            
            if rd.random() > ratio: continue
            if self.solution[i] == self.solution[j]: continue

            new_sol = copy.copy(self.solution)
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]

            if self.valid_swap(i, j):
                if tabu_dict != None and tabu_dict[(-1, i, j, self.solution[i], self.solution[j])] > iter:
                    continue

                new_trans_cost = self.change_of_trans_cost(i, j)
                new_storage_stock = self.change_of_storage_cost(i, j)
                new_cost = new_storage_stock + new_trans_cost

                if new_cost < self.cost:
                    return [[(-1, i, j, self.solution[i], self.solution[j]), new_cost, new_storage_stock, new_trans_cost, "2-Swap"]]

                neighbors.append([(-1, i, j, self.solution[i], self.solution[j]), new_cost, new_storage_stock, new_trans_cost, "2-Swap"])

        return neighbors
    
    ###################################################################################################################
    ###########                                        Utilitaires                                          ########### 
    ###################################################################################################################

    # Effectue le mouvement et mets à jour la structure de données
    def do_swap(self, neigh):
        i = neigh[0][1]
        j = neigh[0][2]

        c_i = self.solution[i]
        if c_i != -1:
            for k in range(i, j):
                self.total_prod[c_i][k] -= 1
        
        c_j = self.solution[j]
        if c_j != -1:
            for k in range(i, j):
                self.total_prod[c_j][k] += 1

        self.solution[i], self.solution[j] = self.solution[j], self.solution[i]

        self.cost = neigh[1]
        self.storage_cost = neigh[2]
        self.transition_cost = neigh[3]
        
    # Verifie qu'un mvmt est valide
    def valid_swap(self, i, j):
        c_i = self.solution[i]
        
        if c_i == -1:
            return True
        
        for k in range(i, j):
            if self.total_prod[c_i][k] - 1 < self.total_order[c_i][k]:
                return False
        
        return True
    
    # Calcule efficacement le changement de cout de stockage du à un mouvement
    def change_of_storage_cost(self, i, j):
        storage_cost = self.storage_cost

        xi, xj = self.solution[i], self.solution[j]
        delta = 0

        # si on déplace une production de i -> j, on stocke (j-i) jours en plus (ou moins)
        if xi != -1:
            delta += self.H * (i - j)
        # idem pour la production de j -> i
        if xj != -1:
            delta += self.H * (j - i)

        return storage_cost + delta

    # Calcule efficacement le changement de cout de transition du à un mouvement
    def change_of_trans_cost(self, i, j):
        # Cout de transition de la solution actuelle
        transition_cost = self.transition_cost

        # Renvoie le plus proche voisin != -1 de idx dans la direction step (-1 ou +1)
        def neighbor(idx, step):
            k = idx + step
            while 0 <= k < self.J and self.solution[k] == -1:
                k += step
            return k if 0 <= k < self.J else None

        # Trouve les transitions à retirer
        old_trans_ind = set()
        for idx in (i, j):
            if self.solution[idx] != -1:
                p = neighbor(idx, -1)
                if p is not None:
                    old_trans_ind.add((min(p, idx), max(p, idx)))
                q = neighbor(idx, +1)
                if q is not None:
                    old_trans_ind.add((min(idx, q), max(idx, q)))
            else:
                p = neighbor(idx, -1)
                q = neighbor(idx, +1)
                if p != None and q != None:
                    old_trans_ind.add((min(p, q), max(p, q)))

        # Calcul la somme des couts de tranistion à retirer
        old = sum(self.instance.transition_cost(self.solution[a], self.solution[b]) for a, b in old_trans_ind)

        # Se rend sur la solution étudiée
        self.solution[i], self.solution[j] = self.solution[j], self.solution[i]

        # Trouve les transitions à ajouter
        new_trans_ind = set()
        for idx in (i, j):
            if self.solution[idx] != -1:
                p = neighbor(idx, -1)
                if p is not None:
                    new_trans_ind.add((min(p, idx), max(p, idx)))
                q = neighbor(idx, +1)
                if q is not None:
                    new_trans_ind.add((min(idx, q), max(idx, q)))
            else:
                p = neighbor(idx, -1)
                q = neighbor(idx, +1)
                if p != None and q != None:
                    new_trans_ind.add((min(p, q), max(p, q)))
        
        # Calcul la somme des couts de tranistion à ajouter
        new = sum(self.instance.transition_cost(self.solution[a], self.solution[b]) for a, b in new_trans_ind)

        # Reviens sur la solution actuel
        self.solution[i], self.solution[j] = self.solution[j], self.solution[i]

        return transition_cost - old + new
    

    # Trace un jolie graphe
    def plot_score_per_iter(self):
        plt.plot(self.iterations, self.score_value, color='blue', label='Score obtenu')
        plt.xscale('log')
        plt.axhline(y=LOWER_BOUND[INSTANCES[INSTANCE]], color='red', linestyle='--', label=f'Borne Inférieure du score de l\'instance')
        plt.axhline(y=UPPER_BOUND[INSTANCES[INSTANCE]], color='green', linestyle='--', label=f'Borne Supérieure du score de l\'instance')
        plt.title("Évolution du score en fonction du nombre d'itérations")
        plt.xlabel("Itérations")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()

