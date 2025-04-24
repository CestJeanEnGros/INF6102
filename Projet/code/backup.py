 # Benchmark
    T_moy = 0
    Best_10 = []
    for i in range(10):
        print(f"Grande boucle {i}")
        solver = Solver(puzzle)
        solver.greedy()
        greedy_board = copy.deepcopy(solver.board)
        greedy_board_pieces = copy.deepcopy(solver.board_pieces)

        v_moy = 0
        for j in range(1):
            print(f"Petite boucle {j}")
            solver.board = copy.deepcopy(greedy_board)
            solver.board_pieces = copy.deepcopy(greedy_board_pieces)
            start = t.time()
            solver.tabu_search()
            T_moy += t.time() - start
            v_moy += puzzle.get_total_n_conflict(solver.format())
        Best_10.append(v_moy/1)
    
    print(f"Best 10 {Best_10}")
    print(f"Best global moy {np.sum(Best_10)/10}")
    print(f"T global moyen {T_moy/100}")



for c_d in conflict_pieces_to_remove:
    self.board[c_d["cell"]] = (0, 0, 0, 0)

for u_d in unconflict_pieces_to_remove:
    self.board[u_d["cell"]] = (0, 0, 0, 0)

    # ################
    # ##   Tabu     ##
    # ################

    # def crown_tabu_search(self, i):
    #     print(f"Crown Tabu Search : {i}")
    #     if (t.time() - self.startTime) > self.T:
    #             return 
        
    #     tabu_list = deque(maxlen=CROWN_ITER_BAN)
    #     crown_cells = self.get_cells_crown(i)
    #     inner_crown_cells = self.get_cells_inside_crown(i)

    #     value = self.get_crown_conflicts(i)
    #     print(value)
    #     best = [copy.deepcopy(self.board), value]

    #     iter = 0
    #     iter_without_best = 0
    #     if i == 0: debug = 1000
    #     else: debug = 50
    #     while (t.time() - self.startTime) < self.T:
    #     # while (t.time() - self.startTime) < self.T and iter < debug:

    #         if i > 0 and iter_without_best > CROWN_MAX_ITER:
    #             break
            
    #         iter += 1 
    #         # if iter % 100 == 0: print(f"Iter {iter}")
            
    #         iter_without_best += 1 

    #         if i == 0:
    #             neighbors = self.neighbors_ext_crown_tabu_search(tabu_list, crown_cells, value, best[1])
    #         else:
    #             neighbors = self.neighbors_crown_tabu_search(tabu_list, i, crown_cells, inner_crown_cells, iter, value, best[1])
    #         neighbors.sort(key=lambda x: x[2])

    #         if len(neighbors) == 0:
    #             continue


    #         # if i >= 1: self.puzzle.display_solution(self.format(), f"pre {iter}")
    #         # print(f"Score avant chgt : {self.get_crown_conflicts(i), value}")

    #         neigh = neighbors[0]
    #         value = neigh[2]
    #         type = neighbors[0][3]

    #         if type == "2-Swap":
    #             x1, y1, p1, o1 = neigh[0] 
    #             x2, y2, p2, o2 = neigh[1]
    #             self.place_piece(x1, y1, p1, o1)
    #             self.place_piece(x2, y2, p2, o2)
    #             tabu_list.append(copy.deepcopy(self.board))
            
    #         elif type == "Rot":
    #             x1, y1, p1, o1 = neigh[0] 
    #             self.place_piece(x1, y1, p1, o1)
    #             tabu_list.append(copy.deepcopy(self.board))
            
    #         if value < best[1]:
    #             iter_without_best = 0
    #             best = [copy.deepcopy(self.board), value]
    #             print(value)
            
    #         # if i >= 1 : self.puzzle.display_solution(self.format(), f"post {iter}")
           
    #         # print(f"Chgt : {x1, y1, p1, o1, x2, y2, p2, o2, value, neigh[-1]}")
    #         # print(f"Score après chgt : {self.get_crown_conflicts(i), value}")

    #         if value == 0:
    #             break
        
    #     return


    # def neighbors_crown_tabu_search(self, tabu_list, i, crown_cells, inner_crown_cells, iter, value, best_value):
    #     neighbors = []

    #     for (x1, y1) in crown_cells:
    #         if (t.time() - self.startTime) > self.T:
    #             return neighbors
            
    #         # Rotation pièce unique
    #         p1 = self.board_pieces[(x1, y1)]
    #         old_o1 = self.board[(x1, y1)]
    #         for new_o1 in self.all_rotations[p1]:
    #             if new_o1 == old_o1 or rd.random() > CROWN_RATIO_VOISINS:
    #                 continue

    #             ### Possible à écnonomiser si on stocke le nombre de conflit de chaque pièce de la solution acutel   
    #             old_conflicts = self.get_crown_conflicts_c(x1, y1, i) 
    #             ##################################################

    #             self.place_piece(x1, y1, p1, new_o1)

    #             new_conflicts = self.get_crown_conflicts_c(x1, y1, i) 

    #             delta = new_conflicts - old_conflicts

    #             if value + delta < best_value or self.board not in tabu_list :    
    #                 neighbors.append([(x1, y1, p1, new_o1), (x1, y1, p1, new_o1), value + delta, "Rot"])

    #             self.place_piece(x1, y1, p1, old_o1)

    #         for (x2, y2) in inner_crown_cells:
    #             if (x2, y2) == (x1, y1):
    #                 continue
                
    #             p2 = self.board_pieces[(x2, y2)]
    #             old_o2 = self.board[(x2, y2)]

    #             for new_o2 in self.all_rotations[p2]:
    #                 if rd.random() > CROWN_RATIO_VOISINS:
    #                     continue
                    
    #                 ### Possible à écnonomiser si on stocke le nombre de conflit de chaque pièce de la solution acutel
    #                 if (x2, y2) in crown_cells:
    #                     old_conflicts = self.get_crown_conflicts_c(x1, y1, i) + self.get_crown_conflicts_c(x2, y2, i)
    #                     if self.correction_double_conflicts(x1, y1, old_o1, x2, y2, old_o2): old_conflicts -= 1
    #                 else:
    #                     old_conflicts = self.get_crown_conflicts_c(x1, y1, i) 
    #                 ##################################################

    #                 self.place_piece(x1, y1, p2, new_o2)
    #                 self.place_piece(x2, y2, p1, old_o1)

    #                 if (x2, y2) in crown_cells:
    #                     new_conflicts = self.get_crown_conflicts_c(x1, y1, i) + self.get_crown_conflicts_c(x2, y2, i)
    #                     if self.correction_double_conflicts(x1, y1, new_o2, x2, y2, old_o1): new_conflicts -= 1
    #                 else:
    #                     new_conflicts = self.get_crown_conflicts_c(x1, y1, i) 

    #                 delta = new_conflicts - old_conflicts

    #                 if value + delta < best_value or self.board not in tabu_list:
    #                     neighbors.append([(x1, y1, p2, new_o2), (x2, y2, p1, old_o1), value + delta, "2-Swap"])

    #                 self.place_piece(x1, y1, p1, old_o1)
    #                 self.place_piece(x2, y2, p2, old_o2)

    #     return neighbors

    # def neighbors_ext_crown_tabu_search(self, tabu_list, crown_cells, value, best_value):
    #     neighbors = []
    #     for (x1, y1), (x2, y2) in combinations(crown_cells, 2):
    #         if (t.time() - self.startTime) > self.T:
    #             return neighbors

    #         if rd.random() > BORDER_RATIO_VOISINS:
    #             continue

    #         p1 = self.board_pieces[(x1, y1)]
    #         old_o1 = self.board[(x1, y1)]
    #         p2 = self.board_pieces[(x2, y2)]
    #         old_o2 = self.board[(x2, y2)]

    #         if not self.homogeneous(p1, p2):
    #             continue
            
    #         for o in self.all_rotations[p1]:
    #             if self.fits_border(x2, y2, o):
    #                 new_o1 = o
    #                 break
    #         for o in self.all_rotations[p2]:
    #             if self.fits_border(x1, y1, o):
    #                 new_o2 = o
    #                 break
            
    #         old_conflicts = self.get_conflicts_p(x1, y1, "border") + self.get_conflicts_p(x2, y2, "border")
    #         if self.correction_double_conflicts(x1, y1, old_o1, x2, y2, old_o2): old_conflicts -= 1
            
    #         self.place_piece(x1, y1, p2, new_o2)
    #         self.place_piece(x2, y2, p1, new_o1)
            
    #         new_conflicts = self.get_conflicts_p(x1, y1, "border") + self.get_conflicts_p(x2, y2, "border")
    #         if self.correction_double_conflicts(x1, y1, new_o2, x2, y2, new_o1): new_conflicts -= 1

    #         delta = new_conflicts - old_conflicts

    #         if value + delta < best_value or self.board not in tabu_list:
    #             neighbors.append([(x1, y1, p2, new_o2), (x2, y2, p1, new_o1), value + delta, "2-Swap"])

    #         self.place_piece(x1, y1, p1, old_o1)
    #         self.place_piece(x2, y2, p2, old_o2)
    
    #     return neighbors

   
    
    


def predict_conflict_greedy(self, x, y, o):
        nb_conflict = 0

        if x-1 < 0: south_col = GRAY
        elif self.board[(x-1, y)] == None: south_col = None
        else: south_col = self.board[(x-1, y)][NORTH]

        if x+1 >= self.n: north_col = GRAY
        elif self.board[(x+1, y)] == None: north_col = None
        else: north_col = self.board[(x+1, y)][SOUTH]
        
        if y-1 < 0: west_col = GRAY
        elif self.board[(x, y-1)] == None: west_col = None
        else: west_col = self.board[(x, y-1)][EAST]

        if y+1 >= self.n: east_col = GRAY
        elif self.board[(x, y+1)] == None: east_col = None
        else: east_col = self.board[(x, y+1)][WEST]

        if south_col != None and south_col != o[SOUTH]: nb_conflict += 1
        if north_col != None and north_col != o[NORTH]: nb_conflict += 1
        if west_col != None and west_col != o[WEST]: nb_conflict += 1
        if east_col != None and east_col != o[EAST]: nb_conflict += 1
        
        return nb_conflict
    

    def homogeneous(self, p1, p2):
        return p1.count(0) == p2.count(0)


def get_crown_conflicts_c1_c2(self, x1, y1, x2, y2):
        if x1 == x2:
            if self.board[(x1, y1)][EAST] != self.board[(x2, y2)][WEST]:
                return 1
        else:
            if self.board[(x1, y1)][NORTH] != self.board[(x2, y2)][SOUTH]:
                return 1
            
        return 0

    def get_cells_crown(self, i):

        cells = []
        
        for c in range(i, self.n - i):
            cells.append((i, c))

        if self.n - i - 1 != i:
            for c in range(i, self.n - i):
                cells.append((self.n - i - 1, c))
        
        for r in range(i + 1, self.n - i - 1):
            cells.append((r, i))

        if self.n - i - 1 != i:
            for r in range(i + 1, self.n - i - 1):
                cells.append((r, self.n - i - 1))
        
        return cells
    
    def get_cells_inside_crown(self, i ):
        cells = [(r, c) for r in range(i, self.n - i) for c in range(i, self.n - i)]
        return cells

def totalBorderConflicts(self):
        total_border_conflicts = 0

        for x, y in self.border_cells:
            total_border_conflicts += self.get_conflicts_p(x, y, "border")/2
       
        return total_border_conflicts


def get_conflicts_p(self, x, y, type):
        conflicts_of_p = 0
        o = self.board[(x, y)]

        if type == "border":
            if x == 0 or x == self.n - 1:
                if x == 0 and o[SOUTH] != GRAY:
                    conflicts_of_p += 1
                if x == self.n - 1 and o[NORTH] != GRAY:
                    conflicts_of_p += 1
                if y-1 >= 0 and self.board[(x, y-1)][EAST] != o[WEST]:
                    conflicts_of_p += 1
                if y+1 < self.n and self.board[(x, y+1)][WEST] != o[EAST]:
                    conflicts_of_p += 1

            if y == 0 or y == self.n - 1:
                if y == 0 and o[WEST] != GRAY:
                    conflicts_of_p += 1
                if y == self.n - 1 and o[EAST] != GRAY:
                    conflicts_of_p += 1
                if x-1 >= 0 and self.board[(x-1, y)][NORTH] != o[SOUTH]:
                    conflicts_of_p += 1
                if x+1 < self.n and self.board[(x+1, y)][SOUTH] != o[NORTH]:
                    conflicts_of_p += 1
    
        return conflicts_of_p

def get_crown_rep(self, crown_cells):
        rep = []
        for x, y in crown_cells:
            rep.append(self.board[(x, y)])
    
        return rep
    
    def get_crown_conflicts_c(self, x, y, i):
        nb_conflicts = 0
        curr_crown = self.get_cells_crown(i)
        prec_crown = self.get_cells_crown(i-1)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (x + dx, y + dy) in prec_crown:
                if i == 0:
                    nb_conflicts += self.get_border_conflict_c1_c2(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy))
                else:
                    nb_conflicts += self.get_crown_conflicts_c1_c2(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy))
            elif (x + dx, y + dy) in curr_crown:
                nb_conflicts += self.get_crown_conflicts_c1_c2(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy))

        return nb_conflicts

def get_crown_conflicts(self, i):
        nb_conflicts = 0
        curr_crown = self.get_cells_crown(i)
        prec_crown = self.get_cells_crown(i-1)

        for x, y in curr_crown:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (x + dx, y + dy) in prec_crown:
                    if i == 0:
                        nb_conflicts += self.get_border_conflict_c1_c2(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy))
                    else:
                        nb_conflicts += self.get_crown_conflicts_c1_c2(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy))
                elif (x + dx, y + dy) in curr_crown:
                    nb_conflicts += self.get_crown_conflicts_c1_c2(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy))/2

        return nb_conflicts

   
    def get_border_conflict_c1_c2(self, x1, y1, x2, y2):
        if x1 == x2:
            if y1 < 0 and self.board[(x2, y2)][WEST] != GRAY:
                return 1
            elif y1 >= 0 and self.board[(x1, y1)][EAST] != GRAY:
                return 1
        else:
            if x1 < 0 and self.board[(x2, y2)][SOUTH] != GRAY:
                return 1
            elif x1 >= 0 and  self.board[(x1, y1)][NORTH] != GRAY:
                return 1
        
        return 0


    

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

    def random_inner_crown(self, i):
        inner_crown_cells = self.get_cells_inside_crown(i)
        inner_crown_pieces = [self.board_pieces[x, y] for (x, y) in inner_crown_cells]
        rd.shuffle(inner_crown_pieces)
        interior_iter = iter(inner_crown_pieces)
        for x, y in inner_crown_cells:
            piece = next(interior_iter)
            self.place_piece(x, y, piece, rd.choice(self.all_rotations[piece]))









            # Algo

    
        