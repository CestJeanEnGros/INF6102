import copy 
import random 
import time 
from math import exp, comb
from solver_heuristic import get_spiral
#####################
# Constantes Pieces #
#####################
GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3
############################
# Configuration Algorithme #
############################
NUMBER_RESTARTS = 50
TABU_LIST = []
MAX_TABU_SIZE = 360
TEMP_0 = 100.0
TEMP_F = 0.05
COOLING_FACTOR = 0.99
MAX_ITER_1 = 400
MAX_ITER_2 = 400
MAX_ITER_3 = 1400
MAX_TIME_1 = 60
MAX_TIME_2 = 3600
BETA_1 = 0.5
BETA_2 = 6000
GAMMA = 0.5
GAMMA_BORDER = 0.2
###################
# Classe Solution #
###################
class Solution:
    def __init__(self, eternity_puzzle, sol, tabu_list=[], max_tabu_size=MAX_TABU_SIZE):
        self.__eternity_puzzle = eternity_puzzle
        self.__sol = sol 
        self.__tabu_list = tabu_list
        self.__tabu_size = max_tabu_size

        # Separation des pieces
        self.all_index = [i for i in range(eternity_puzzle.n_piece)]
        self.spiral_index = get_spiral(eternity_puzzle)
        self.corner_index = [0, eternity_puzzle.board_size-1, eternity_puzzle.n_piece-eternity_puzzle.board_size, eternity_puzzle.n_piece-1]
        self.border_index = [i for i in range(1,eternity_puzzle.board_size-1)] + [i for i in range(eternity_puzzle.n_piece-eternity_puzzle.board_size+1, eternity_puzzle.n_piece-1)] + [i*eternity_puzzle.board_size for i in range(1, eternity_puzzle.board_size-1)] + [(i+1)*eternity_puzzle.board_size-1 for i in range(1, eternity_puzzle.board_size-1)]
        self.inner_index = list((set(self.all_index).difference(set(self.corner_index))).difference(set(self.border_index)))
        self.homogenous_pieces = comb(4, 2) + comb(4*(eternity_puzzle.board_size-2), 2) + comb((eternity_puzzle.board_size-2)**2, 2)
        self.__piece2conflict, self.__conflict2piece = self.generate_conflicts()

    def get_puzzle(self):
        return self.__eternity_puzzle
    def get_sol(self):
        return self.__sol
    def get_piece2conflict(self):
        return self.__piece2conflict
    def get_conflict2piece(self):
        return self.__conflict2piece
    def get_tabu(self):
        return self.__tabu_list
    def add_tabu(self, elem):
        self.__tabu_list.append(elem)
    def clear_tabu(self):
        while len(self.__tabu_list) > self.__tabu_size:
            self.__tabu_list.pop(0)
    def reset_tabu(self):
        self.__tabu_list = []
    def generate_conflicts(self):
        piece2conflict = {i: 0 for i in range(self.__eternity_puzzle.n_piece)}
        conflict2piece = {k: [] for k in range(5)}
        for k in self.all_index:
            piece2conflict[k] = count_conflict(self, k)
            conflict2piece[count_conflict(self, k)].append(k)
        return piece2conflict, conflict2piece

# Commpte le nombre de conflits pour une piece a la position k dans le puzzle
def count_conflict(solution: Solution, k: int) -> int:
    puzzle = solution.get_puzzle()
    i = k % puzzle.board_size
    j = k // puzzle.board_size
    sol = solution.get_sol()
    num_conflict = 0
    if sol[k] == 0:
        return num_conflict
    if i > 0 and sol[puzzle.board_size*j + (i-1)] != 0:
        if sol[k][WEST] != sol[puzzle.board_size*j + (i-1)][EAST]:
            num_conflict += 1
    if i < puzzle.board_size-1 and sol[puzzle.board_size*j + (i+1)] != 0:
        if sol[k][EAST] != sol[puzzle.board_size*j + (i+1)][WEST]:
            num_conflict += 1
    if j > 0 and sol[puzzle.board_size*(j-1) + i] != 0:
        if sol[k][SOUTH] != sol[puzzle.board_size*(j-1) + i][NORTH]:
            num_conflict += 1
    if j < puzzle.board_size-1 and sol[puzzle.board_size*(j+1) + i] != 0:
        if sol[k][NORTH] != sol[puzzle.board_size*(j+1) + i][SOUTH]:
            num_conflict += 1
    nb_gray = sol[k].count(GRAY)
    if nb_gray == 1:
        if i == 0:
            if sol[k][WEST] != GRAY:
                num_conflict += 1
        if i == puzzle.board_size-1:
            if sol[k][EAST] != GRAY:
                num_conflict += 1
        if j == 0:
            if sol[k][SOUTH] != GRAY:
                num_conflict += 1
        if j == puzzle.board_size-1:
            if sol[k][NORTH] != GRAY:
                num_conflict += 1
    if nb_gray == 2:
        if i == 0 and j == 0:
            if sol[k][WEST] != GRAY:
                num_conflict += 1
            if sol[k][SOUTH] != GRAY:
                num_conflict += 1
        if i == puzzle.board_size-1 and j == 0:
            if sol[k][EAST] != GRAY:
                num_conflict += 1
            if sol[k][SOUTH] != GRAY:
                num_conflict += 1
        if i == puzzle.board_size-1 and j == puzzle.board_size-1:
            if sol[k][EAST] != GRAY:
                num_conflict += 1
            if sol[k][NORTH] != GRAY:
                num_conflict += 1
        if i == 0 and j == puzzle.board_size-1:
            if sol[k][WEST] != GRAY:
                num_conflict += 1
            if sol[k][NORTH] != GRAY:
                num_conflict += 1
    return num_conflict

# Compte le nombre total de conflits entre les pieces du bord et seulement les pieces du bord 
def conflicts_on_border(solution: Solution) -> int:
    temp_sol = [0]*solution.get_puzzle().n_piece
    tot_conflicts = 0
    for i in range(4*solution.get_puzzle().board_size-4):
        temp_sol[solution.spiral_index[i]] = solution.get_sol()[solution.spiral_index[i]]
    temp_sol = Solution(solution.get_puzzle(), temp_sol)
    for i in range(4*solution.get_puzzle().board_size-4):
        tot_conflicts += count_conflict(temp_sol, temp_sol.spiral_index[i])
    return tot_conflicts
####################
### NEIGHBORHOOD ###
####################
# Tourne chaque piece pour trouver une meilleure configuration 
# 3(n-2)^2 si on ne regarde que l'interieur
# n^2 si on regarde toutes les pieces '
# type_piece: 0 -> inner; 1 -> border; 2 -> all 
def rotate_inner(solution: Solution, greedy=False, is_tabu=True, type_piece=0):
    eternity_puzzle = solution.get_puzzle()
    sol = solution.get_sol()
    sol_conflicts = eternity_puzzle.get_total_n_conflict(sol)

    match type_piece:
        case 0:
            indices = solution.inner_index
        case 1:
            indices = solution.corner_index + solution.border_index
        case 2:
            indices = solution.all_index

    best_rotation = None 
    best_piece = None 
    best_change = 0
    num_conflict = 4
    while num_conflict >= 0:
        for piece in solution.get_conflict2piece()[num_conflict]:
            if piece in indices:
                rotations = eternity_puzzle.generate_rotation(sol[piece])
                temp = sol[piece]
                for r in range(1, len(rotations)):
                    sol[piece] = rotations[r]
                    conflicts = count_conflict(solution, piece)
                    if conflicts - num_conflict < best_change:
                        if (not is_tabu) or ((temp, r, sol_conflicts) not in solution.get_tabu()):
                            best_change = conflicts - num_conflict
                            best_rotation = r
                            best_piece = piece 
                            if greedy:
                                sol[piece] = rotations[0]
                                return best_piece, best_rotation, sol_conflicts, best_change
                                # if is_tabu:
                                #     solution.add_tabu((best_piece, best_rotation, sol_conflicts))
                                # sol[best_piece] = eternity_puzzle.generate_rotation(sol[best_piece])[best_rotation]
                                # solution.generate_conflicts()
                                # return solution, best_change

                sol[piece] = rotations[0]
        num_conflict -= 1

    # if best_change != 0:
    #     if is_tabu:
    #         solution.add_tabu((best_piece, best_rotation, sol_conflicts))
    #     sol[best_piece] = eternity_puzzle.generate_rotation(sol[best_piece])[best_rotation]
    #     solution.generate_conflicts()

    return best_piece, best_rotation, sol_conflicts, best_change

# Tente d'echanger avec rotations toutes les paires de pieces homogenes = meme type (corner avec corner, inner avec inner, border avec border)
# 16(n-2)^4 si on ne regarde que l'interieur 
# type_piece: 0 -> inner; 1 -> border; 2 -> all 
def swap_2(solution: Solution, greedy=False, is_tabu=True, type_piece=0, random_swap=False, altering=False):
    eternity_puzzle = solution.get_puzzle()
    sol = solution.get_sol()
    if is_tabu:
        if type_piece == 1:
            sol_conflicts = conflicts_on_border(solution)
        else:
            sol_conflicts = eternity_puzzle.get_total_n_conflict(sol)
    else:
        sol_conflicts = eternity_puzzle.get_total_n_conflict(sol)

    # On garde une trace de la meilleure nouvelle configuration trouvee
    best_piece_a = None 
    best_rot_a = None
    best_piece_b = None
    best_rot_b = None  
    best_change = 0 
    num_conflict = 4 

    # On garde egalement une trace de la meilleure configuration alternative dans le cas ou aucune amelioration n'est trouvee
    best_alter_a = None
    best_alter_rot_a = None 
    best_alter_b = None 
    best_alter_rot_b = None 
    best_alter_change = 42

    if type_piece == 0:
        indices = solution.inner_index
    elif type_piece == 1:
        indices = solution.corner_index + solution.border_index
    else:
        indices = solution.all_index

    if random_swap:
        if type_piece == 0:
            piece_a = 0
            piece_b = 0
            while piece_a == piece_b:
                piece_a = random.choice(indices)
                piece_b = random.choice(indices)
                r_a = random.choice([0, 1, 2, 3])
                r_b = random.choice([0, 1, 2, 3])
            # temp = sol[piece_a]
            # sol[piece_a] = eternity_puzzle.generate_rotation(sol[piece_b])[r_a]
            # sol[piece_b] = eternity_puzzle.generate_rotation(temp)[r_b]
            # solution.generate_conflicts()
            # return solution, best_change
            return piece_a, r_a, piece_b, r_b, sol_conflicts, -1
        elif type_piece == 1:
            piece_a = 0
            piece_b = 0
            while piece_a == piece_b or (piece_a in solution.border_index and piece_b in solution.corner_index) or (piece_a in solution.corner_index and piece_b in solution.border_index):
                piece_a = random.choice(solution.corner_index)
                piece_b = random.choice(solution.corner_index)
                # r_a = random.choice([0, 1, 2, 3])
                # r_b = random.choice([0, 1, 2, 3])
            nb_gray = sol[piece_a].count(GRAY)
            r_a = 0
            r_b = 0
            if nb_gray == 1:
                if sol[piece_a][0] == GRAY:
                    orientation_a = 0
                elif sol[piece_a][1] == GRAY:
                    orientation_a = 1
                elif sol[piece_a][2] == GRAY:
                    orientation_a = 2
                elif sol[piece_a][3] == GRAY:
                    orientation_a = 3

                if sol[piece_b][0] == GRAY:
                    orientation_b = 0
                elif sol[piece_b][1] == GRAY:
                    orientation_b = 1
                elif sol[piece_b][2] == GRAY:
                    orientation_b = 2
                elif sol[piece_b][3] == GRAY:
                    orientation_b = 3

                rotations_a = eternity_puzzle.generate_rotation(sol[piece_a])
                rotations_b = eternity_puzzle.generate_rotation(sol[piece_b])
                while rotations_a[r_a][orientation_b] != GRAY:
                    r_a += 1
                while rotations_b[r_b][orientation_a] != GRAY:
                    r_b += 1

            elif nb_gray == 2:
                orientation_a_1 = None
                orientation_a_2 = None 
                for i in range(4):
                    if sol[piece_a][i] == GRAY:
                        if orientation_a_1 != None:
                            orientation_a_2 = i 
                            break 
                        else:
                            orientation_a_1 = i

                orientation_b_1 = None
                orientation_b_2 = None 
                for i in range(4):
                    if sol[piece_b][i] == GRAY:
                        if orientation_b_1 != None:
                            orientation_b_2 = i 
                            break 
                        else:
                            orientation_b_1 = i               

                rotations_a = eternity_puzzle.generate_rotation(sol[piece_a])
                rotations_b = eternity_puzzle.generate_rotation(sol[piece_b])
                while rotations_a[r_a][orientation_b_1] != GRAY or rotations_a[r_a][orientation_b_2] != GRAY:
                    r_a += 1
                while rotations_b[r_b][orientation_a_1] != GRAY or rotations_b[r_b][orientation_a_2] != GRAY:
                    r_b += 1

            # temp = sol[piece_a]
            # sol[piece_a] = eternity_puzzle.generate_rotation(sol[piece_b])[r_a]
            # sol[piece_b] = eternity_puzzle.generate_rotation(temp)[r_b]
            # solution.generate_conflicts()
            # return solution, best_change
            return piece_a, r_b, piece_b, r_a, sol_conflicts, -1
        else:
            piece_a = 0
            piece_b = 0
            while piece_a == piece_b or not ((piece_a in solution.border_index and piece_b in solution.border_index) or (piece_a in solution.corner_index and piece_b in solution.corner_index) or (piece_a in solution.inner_index and piece_b in solution.inner_index)):
                piece_a = random.choice(indices)
                piece_b = random.choice(indices)
                r_a = random.choice([0, 1, 2, 3])
                r_b = random.choice([0, 1, 2, 3])
            # temp = sol[piece_a]
            # sol[piece_a] = eternity_puzzle.generate_rotation(sol[piece_b])[r_a]
            # sol[piece_b] = eternity_puzzle.generate_rotation(temp)[r_b]
            # solution.generate_conflicts()
            # return solution, best_change
            return piece_a, r_a, piece_b, r_b, sol_conflicts, -1

    pairs_checked = 0
    random.shuffle(indices)
    while num_conflict >= 0 and pairs_checked <= max(BETA_1*solution.homogenous_pieces, BETA_2) :
        random.shuffle(solution.get_conflict2piece()[num_conflict])
        for piece_a in solution.get_conflict2piece()[num_conflict]:
            temp_a = sol[piece_a]
            if piece_a in indices:
                for piece_b in indices:
                    if piece_a != piece_b:
                        # Check si les pieces sont homogenes = du meme type
                        if ((piece_a in solution.corner_index and piece_b in solution.corner_index) or (piece_a in solution.border_index and piece_b in solution.border_index) or ((piece_a in solution.inner_index and piece_b in solution.inner_index))):
                            pairs_checked += 1
                            # Compte les conflits impliquant ces 2 pieces en retirant les doublons 
                            curr_conflict_a = count_conflict(solution, piece_a)
                            curr_conflict_b = count_conflict(solution, piece_b) 
                            curr_conflicts = curr_conflict_a + curr_conflict_b
                            i_a = piece_a % eternity_puzzle.board_size
                            j_a = piece_a // eternity_puzzle.board_size
                            if i_a > 0 and piece_b == piece_a-1:
                                if sol[piece_b][EAST] != sol[piece_a][WEST]:
                                    curr_conflicts -= 1
                            elif i_a < eternity_puzzle.board_size-1 and piece_b == piece_a+1:
                                if sol[piece_b][WEST] != sol[piece_a][EAST]:
                                    curr_conflicts -= 1
                            elif j_a > 0 and piece_b == piece_a-eternity_puzzle.board_size:
                                if sol[piece_b][NORTH] != sol[piece_a][SOUTH]:
                                    curr_conflicts -= 1
                            elif j_a < eternity_puzzle.board_size-1 and piece_b == piece_a+eternity_puzzle.board_size:
                                if sol[piece_b][SOUTH] != sol[piece_a][NORTH]:
                                    curr_conflicts -= 1
                            temp_a = sol[piece_a]
                            temp_b = sol[piece_b]
                            for r_a in [0, 1, 2, 3]:
                                for r_b in [0, 1, 2, 3]:
                                    sol[piece_a] = eternity_puzzle.generate_rotation(temp_b)[r_a]
                                    sol[piece_b] = eternity_puzzle.generate_rotation(temp_a)[r_b]
                                    new_conflict_a = count_conflict(solution, piece_a)
                                    new_conflict_b = count_conflict(solution, piece_b)
                                    new_conflicts = new_conflict_a + new_conflict_b
                                    if i_a > 0 and piece_b == piece_a-1:
                                        if sol[piece_b][EAST] != sol[piece_a][WEST]:
                                            new_conflicts -= 1
                                    elif i_a < eternity_puzzle.board_size-1 and piece_b == piece_a+1:
                                        if sol[piece_b][WEST] != sol[piece_a][EAST]:
                                            new_conflicts -= 1
                                    elif j_a > 0 and piece_b == piece_a-eternity_puzzle.board_size:
                                        if sol[piece_b][NORTH] != sol[piece_a][SOUTH]:
                                            new_conflicts -= 1
                                    elif j_a < eternity_puzzle.board_size-1 and piece_b == piece_a+eternity_puzzle.board_size:
                                        if sol[piece_b][SOUTH] != sol[piece_a][NORTH]:
                                            new_conflicts -= 1
                                    if (sol_conflicts+new_conflicts-curr_conflicts) == 0:
                                        best_change = new_conflicts - curr_conflicts
                                        best_piece_a = piece_a
                                        best_piece_b = piece_b
                                        best_rot_a = r_a 
                                        best_rot_b = r_b
                                        return best_piece_a, best_rot_a, best_piece_b, best_rot_b, sol_conflicts, best_change
                                    if (new_conflicts - curr_conflicts < best_change):
                                        if (not is_tabu) or ((temp_a, r_a, temp_b, r_b, sol_conflicts, sol_conflicts+new_conflicts-curr_conflicts) not in solution.get_tabu() and (temp_b, r_b, temp_a, r_a, sol_conflicts, sol_conflicts+new_conflicts-curr_conflicts) not in solution.get_tabu()):
                                            best_change = new_conflicts - curr_conflicts
                                            best_piece_a = piece_a
                                            best_piece_b = piece_b
                                            best_rot_a = r_a 
                                            best_rot_b = r_b
                                            if greedy:
                                                sol[piece_b] = temp_b
                                                sol[piece_a] = temp_a
                                                # if is_tabu:
                                                #     solution.add_tabu((best_piece_a, best_rot_a, best_piece_b, best_rot_b, sol_conflicts))
                                                # temp = sol[best_piece_a]
                                                # sol[best_piece_a] = eternity_puzzle.generate_rotation(sol[best_piece_b])[best_rot_a]
                                                # sol[best_piece_b] = eternity_puzzle.generate_rotation(temp)[best_rot_b]
                                                # solution.generate_conflicts()
                                                # return solution, best_change
                                                return best_piece_a, best_rot_a, best_piece_b, best_rot_b, sol_conflicts, best_change


                                    if (new_conflicts - curr_conflicts < best_alter_change) :
                                        if (not is_tabu) or ((temp_a, r_a, temp_b, r_b, sol_conflicts, sol_conflicts+new_conflicts-curr_conflicts) not in solution.get_tabu() and (temp_b, r_b, temp_a, r_a, sol_conflicts, sol_conflicts+new_conflicts-curr_conflicts) not in solution.get_tabu()):
                                            # print("TEST")
                                            # print(solution.get_tabu())
                                            # print((piece_a, r_a, piece_b, r_b, sol_conflicts, sol_conflicts+new_conflicts-curr_conflicts))
                                            # print((piece_a, (4-r_a)%4, piece_b, (4-r_b)%4, sol_conflicts+new_conflicts-curr_conflicts, sol_conflicts))
                                            if len(solution.get_tabu()) == 0 or (((temp_a, (4-r_a)%4, temp_b, (4-r_b)%4, sol_conflicts+new_conflicts-curr_conflicts, sol_conflicts) != solution.get_tabu()[-1]) and ((temp_b, (4-r_b)%4, temp_a, (4-r_a)%4, sol_conflicts+new_conflicts-curr_conflicts, sol_conflicts) != solution.get_tabu()[-1])):
                                                best_alter_change = new_conflicts - curr_conflicts
                                                best_alter_a = piece_a
                                                best_alter_b = piece_b
                                                best_alter_rot_a = r_a 
                                                best_alter_rot_b = r_b 
                            sol[piece_b] = temp_b
                            sol[piece_a] = temp_a
        num_conflict -= 1
    
    if best_change != 0:
        # if is_tabu:
        #     solution.add_tabu((best_piece_a, best_rot_a, best_piece_b, best_rot_b, sol_conflicts))
        # temp = sol[best_piece_a]
        # sol[best_piece_a] = eternity_puzzle.generate_rotation(sol[best_piece_b])[best_rot_a]
        # sol[best_piece_b] = eternity_puzzle.generate_rotation(temp)[best_rot_b]
        # solution.generate_conflicts()
        return best_piece_a, best_rot_a, best_piece_b, best_rot_b, sol_conflicts, best_change
    
    else:
        if altering and best_alter_change != 42:
            # if is_tabu:
            #     solution.add_tabu((best_alter_a, best_alter_rot_a, best_alter_b, best_alter_rot_b, sol_conflicts))
            # temp = sol[best_alter_a]
            # sol[best_alter_a] = eternity_puzzle.generate_rotation(sol[best_alter_b])[best_alter_rot_a]
            # sol[best_alter_b] = eternity_puzzle.generate_rotation(temp)[best_alter_rot_b]
            # solution.generate_conflicts()
            return best_alter_a, best_alter_rot_a, best_alter_b, best_alter_rot_b, sol_conflicts, best_alter_change
        else:
            return None, None, None, None, None, 0

###############################
### Build Init with Phase 1 ###
###############################
# Use of build_greedy from solver_heuristic
# Decomposition in 2 phases
# Tabu Search as Phase 1 on border pieces and corner pieces
# Then Greedy to build inner
# Tabu Search as Phase 2 on inner pieces
def phase_1(eternity_puzzle):
    solution = [0]*eternity_puzzle.n_piece
    sol = Solution(eternity_puzzle, solution)

    corner_pieces = [] 
    border_pieces = [] 
    inner_pieces  = []

    # Separation des pièces en 3 categories
    for piece in eternity_puzzle.piece_list:
        nb_gray = piece.count(GRAY)
        if nb_gray == 2:
            corner_pieces.append(piece)
        elif nb_gray == 1:
            border_pieces.append(piece)
        else:
            inner_pieces.append(piece)

    # Placement des coins et orientation
    index_corner = 0
    random.shuffle(corner_pieces)
    for k in sol.corner_index:
        solution[k] = corner_pieces[index_corner]
        if k == sol.corner_index[0]:
            while (solution[k][SOUTH] != GRAY or solution[k][WEST] != GRAY):
                solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]
        if k == sol.corner_index[1]:
            while (solution[k][SOUTH] != GRAY or solution[k][EAST] != GRAY):
                solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]

        if k == sol.corner_index[2]:
            while (solution[k][NORTH] != GRAY or solution[k][WEST] != GRAY):
                solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]

        if k == sol.corner_index[3]:
            while (solution[k][NORTH] != GRAY or solution[k][EAST] != GRAY):
                solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]
        index_corner += 1

    # Placement des bordures et orientation en minimisant les conflits
    index_visited = []
    random.shuffle(border_pieces)
    for k in sol.all_index:
        if k in sol.border_index:
            index_best = None
            best_orientation = None
            min_conflict = 4
            for b in range(len(border_pieces)):
                if b not in index_visited:
                    i = k % eternity_puzzle.board_size
                    j = k // eternity_puzzle.board_size
                    r = 0
                    solution[k] = border_pieces[b]
                    if j == 0:
                        while (solution[k][SOUTH] != GRAY):
                            r += 1
                            solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]
                    elif j == eternity_puzzle.board_size-1:
                        while (solution[k][NORTH] != GRAY):
                            r += 1
                            solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]
                    elif i == 0:
                        while (solution[k][WEST] != GRAY):
                            r += 1
                            solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]
                    elif i == eternity_puzzle.board_size-1:
                        while (solution[k][EAST] != GRAY):
                            r += 1
                            solution[k] = eternity_puzzle.generate_rotation(solution[k])[1]

                    if index_best == None:
                        index_best = b
                        best_orientation = r
                    if count_conflict(sol, k) < min_conflict:
                        min_conflict = count_conflict(sol, k)
                        index_best = b
                        best_orientation = r
            solution[k] = eternity_puzzle.generate_rotation(border_pieces[index_best])[best_orientation]
            index_visited.append(index_best)

    return Solution(eternity_puzzle, solution)
    
def phase_2(solution: Solution):
    eternity_puzzle = solution.get_puzzle()
    sol = solution.get_sol()

    corner_pieces = [] 
    border_pieces = [] 
    inner_pieces  = []

    # Separation des pièces en 3 categories
    for piece in eternity_puzzle.piece_list:
        nb_gray = piece.count(GRAY)
        if nb_gray == 2:
            corner_pieces.append(piece)
        elif nb_gray == 1:
            border_pieces.append(piece)
        else:
            inner_pieces.append(piece)

    # Placement des pieces internes et orientation en minimisant les conflits
    index_visited = []
    random.shuffle(inner_pieces)
    for k in solution.all_index:
        if k in solution.inner_index:
            index_best = None
            best_orientation = None
            min_conflict = 4
            for i in range(len(inner_pieces)):
                if i not in index_visited:
                    rotations = [0, 1, 2, 3]
                    random.shuffle(rotations)
                    for r in range(4):
                        if index_best == None:
                            index_best = i 
                            best_orientation = r
                        sol[k] = eternity_puzzle.generate_rotation(inner_pieces[i])[r]
                        if count_conflict(solution, k) < min_conflict:
                            min_conflict = count_conflict(solution, k)
                            index_best = i 
                            best_orientation = r
            sol[k] = eternity_puzzle.generate_rotation(inner_pieces[index_best])[best_orientation]
            index_visited.append(index_best)    
    
    return solution

def phase_2_random(solution: Solution):
    eternity_puzzle = solution.get_puzzle()
    sol = solution.get_sol()

    corner_pieces = [] 
    border_pieces = [] 
    inner_pieces  = []

    # Separation des pièces en 3 categories
    for piece in eternity_puzzle.piece_list:
        nb_gray = piece.count(GRAY)
        if nb_gray == 2:
            corner_pieces.append(piece)
        elif nb_gray == 1:
            border_pieces.append(piece)
        else:
            inner_pieces.append(piece)

    # Placement des pieces internes et orientation en minimisant les conflits
    index_visited = []
    random.shuffle(inner_pieces)
    index = 0
    for k in solution.all_index:
        if k in solution.inner_index:
            orientation = random.choice([0,1,2,3])
            sol[k] = eternity_puzzle.generate_rotation(inner_pieces[index])[orientation]
            index += 1  
    
    return solution

def solve_advanced(eternity_puzzle):
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    ##################
    # Configuration
    ##################
    # r = random.random()
    # r = 0.23293498870400287
    # r = 0.23293498870400287
    # r = 0.1077730560307194
    # r = 0.4859014510305725
    # r = 0.604053790924224
    # print(f'Seed: {r}')
    # random.seed(r)
    ##################
    # Phase 1
    sol_init = phase_1(eternity_puzzle)

    # Tabu Search after Phase 1 for borders + corners
    tot_conflicts = conflicts_on_border(sol_init)
    start_tabu_phase_1 = time.time()
    iteration_1 = 0
    while tot_conflicts != 0 and (time.time() - start_tabu_phase_1) < MAX_TIME_1 and iteration_1 < MAX_ITER_1:
        iteration_1 += 1
        swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_change = swap_2(sol_init, greedy=False, is_tabu=True, type_piece=1, random_swap=False, altering=True)
        if swap_BP_a != None:
            sol_init.add_tabu((sol_init.get_sol()[swap_BP_a], swap_BR_a, sol_init.get_sol()[swap_BP_b], swap_BR_b, swap_base_conflicts))
            temp = sol_init.get_sol()[swap_BP_a]
            sol_init.get_sol()[swap_BP_a] = eternity_puzzle.generate_rotation(sol_init.get_sol()[swap_BP_b])[swap_BR_a]
            sol_init.get_sol()[swap_BP_b] = eternity_puzzle.generate_rotation(temp)[swap_BR_b]
            sol_init.generate_conflicts()
        tot_conflicts = conflicts_on_border(sol_init)
        sol_init.clear_tabu()

    # Phase 2
    sol_init = phase_2(sol_init)
    print(f'Score before Phase 2: {eternity_puzzle.get_total_n_conflict(sol_init.get_sol())}')
    # return sol_init.get_sol(), eternity_puzzle.get_total_n_conflict(sol_init.get_sol())
    best_sol_overall = Solution(eternity_puzzle, copy.deepcopy(sol_init.get_sol()))
    best_sol = Solution(eternity_puzzle, copy.deepcopy(sol_init.get_sol()))
    curr_sol = Solution(eternity_puzzle, copy.deepcopy(sol_init.get_sol()))
    start_time = time.time()
    iteration_2 = 0
    iteration_3 = 0
    while eternity_puzzle.get_total_n_conflict(best_sol_overall.get_sol()) != 0 and (time.time() - start_time) < MAX_TIME_2:
        print(f'Iteration without Improvements: {iteration_2}')
        rot_BP, rot_BR, rot_base_conflicts, rot_change = rotate_inner(curr_sol, greedy=False, is_tabu=True, type_piece=0)
        swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_change = swap_2(curr_sol, greedy=False, is_tabu=True, type_piece=2, random_swap=False, altering=True)

        if rot_change != 0 and rot_change < swap_change:
            curr_sol.add_tabu((curr_sol.get_sol()[rot_BP], rot_BR, rot_base_conflicts))
            curr_sol.get_sol()[rot_BP] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[rot_BP])[rot_BR]
            curr_sol.generate_conflicts()
        else:
            curr_sol.add_tabu((curr_sol.get_sol()[swap_BP_a], swap_BR_a, curr_sol.get_sol()[swap_BP_b], swap_BR_b, swap_base_conflicts, swap_base_conflicts+swap_change))
            temp = curr_sol.get_sol()[swap_BP_a]
            curr_sol.get_sol()[swap_BP_a] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[swap_BP_b])[swap_BR_a]
            curr_sol.get_sol()[swap_BP_b] = eternity_puzzle.generate_rotation(temp)[swap_BR_b]
            curr_sol.generate_conflicts()

        curr_sol.clear_tabu()
        new_conflicts = eternity_puzzle.get_total_n_conflict(curr_sol.get_sol())

        if new_conflicts < eternity_puzzle.get_total_n_conflict(best_sol.get_sol()):
            print(new_conflicts)
            iteration_2 = 0
            iteration_3 = 0
            best_sol = Solution(eternity_puzzle, copy.deepcopy(curr_sol.get_sol()))
            if eternity_puzzle.get_total_n_conflict(best_sol.get_sol()) < eternity_puzzle.get_total_n_conflict(best_sol_overall.get_sol()):
                best_sol_overall = Solution(eternity_puzzle, copy.deepcopy(best_sol.get_sol()))
        else:
            iteration_2 += 1
            iteration_3 += 1

        # Reset init solution
        if iteration_3 >= MAX_ITER_3 and (time.time() - start_time) < MAX_TIME_2:
            print("SWAP SOL INIT")
            iteration_1 = 0
            iteration_2 = 0 
            iteration_3 = 0
            
            for _ in range(int(GAMMA_BORDER*GAMMA*eternity_puzzle.board_size**2)):
                swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_change = swap_2(curr_sol, greedy=False, is_tabu=True, type_piece=1, random_swap=True, altering=True)
                temp = curr_sol.get_sol()[swap_BP_a]
                curr_sol.get_sol()[swap_BP_a] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[swap_BP_b])[swap_BR_a]
                curr_sol.get_sol()[swap_BP_b] = eternity_puzzle.generate_rotation(temp)[swap_BR_b]

            for _ in range(int((1-GAMMA_BORDER)*GAMMA*eternity_puzzle.board_size**2)):
                swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_change = swap_2(curr_sol, greedy=False, is_tabu=True, type_piece=0, random_swap=True, altering=True)
                temp = curr_sol.get_sol()[swap_BP_a]
                curr_sol.get_sol()[swap_BP_a] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[swap_BP_b])[swap_BR_a]
                curr_sol.get_sol()[swap_BP_b] = eternity_puzzle.generate_rotation(temp)[swap_BR_b]

            best_sol = Solution(eternity_puzzle, copy.deepcopy(curr_sol.get_sol()))
            curr_sol = Solution(eternity_puzzle, copy.deepcopy(curr_sol.get_sol()))
            print("END SWAP SOL INIT")
        
        # SA to help with local minima if stuck 
        elif iteration_2 >= MAX_ITER_2 and (time.time() - start_time) < MAX_TIME_2:
            print("SIMULATED ANNEALING")
            iteration_2 = 0
            temperature = TEMP_0
            curr_sol.reset_tabu()
            while temperature >= TEMP_F and (time.time() - start_time) < MAX_TIME_2:
                rot_BP, rot_BR, rot_base_conflicts, rot_change = rotate_inner(curr_sol, greedy=False, is_tabu=True, type_piece=0)
                swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_change = swap_2(curr_sol, greedy=False, is_tabu=True, type_piece=2, random_swap=False, altering=True)
                if rot_change < 0 or swap_change < 0:
                    if rot_change != 0 and rot_change < swap_change:
                        curr_sol.reset_tabu()
                        curr_sol.add_tabu((rot_BP, rot_BR, rot_base_conflicts))
                        curr_sol.get_sol()[rot_BP] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[rot_BP])[rot_BR]
                        curr_sol.generate_conflicts()
                    else:
                        curr_sol.reset_tabu()
                        curr_sol.add_tabu((swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_base_conflicts+swap_change))
                        temp = curr_sol.get_sol()[swap_BP_a]
                        curr_sol.get_sol()[swap_BP_a] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[swap_BP_b])[swap_BR_a]
                        curr_sol.get_sol()[swap_BP_b] = eternity_puzzle.generate_rotation(temp)[swap_BR_b]
                        curr_sol.generate_conflicts()
                else:
                    delta = swap_change
                    proba_selection = random.uniform(0, 1)
                    if proba_selection <= exp((-delta)/temperature):
                        curr_sol.reset_tabu()
                        curr_sol.add_tabu((swap_BP_a, swap_BR_a, swap_BP_b, swap_BR_b, swap_base_conflicts, swap_base_conflicts+swap_change))
                        temp = curr_sol.get_sol()[swap_BP_a]
                        curr_sol.get_sol()[swap_BP_a] = eternity_puzzle.generate_rotation(curr_sol.get_sol()[swap_BP_b])[swap_BR_a]
                        curr_sol.get_sol()[swap_BP_b] = eternity_puzzle.generate_rotation(temp)[swap_BR_b]
                        curr_sol.generate_conflicts()
                
                new_conflicts = eternity_puzzle.get_total_n_conflict(curr_sol.get_sol())
                if new_conflicts < eternity_puzzle.get_total_n_conflict(best_sol.get_sol()):
                    print("BETTER IN SA")
                    iteration_2 = 0
                    iteration_3 = 0
                    best_sol = Solution(eternity_puzzle, copy.deepcopy(curr_sol.get_sol()))
                    if eternity_puzzle.get_total_n_conflict(best_sol.get_sol()) < eternity_puzzle.get_total_n_conflict(best_sol_overall.get_sol()):
                        best_sol_overall = Solution(eternity_puzzle, copy.deepcopy(best_sol.get_sol()))
        
                temperature = temperature*COOLING_FACTOR
            print("END SIMULATED ANNEALING")

    print(len(curr_sol.get_tabu()))
    print(curr_sol.get_tabu())

    return best_sol_overall.get_sol(), eternity_puzzle.get_total_n_conflict(best_sol_overall.get_sol())